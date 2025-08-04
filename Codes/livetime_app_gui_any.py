#!/usr/bin/env python3
import sys, os, random
import cv2
import numpy as np
import tensorflow as tf
from collections import deque, Counter
from PyQt6 import QtCore, QtWidgets
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
from PyQt6.QtMultimediaWidgets import QVideoWidget
from PyQt6.QtCore import QUrl
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model

# 설정
EMOTIONS       = ['neutral', 'sad', 'fear', 'happy']
SEG_SEC        = 10          # 버퍼 길이 (초)
CAL_EPOCHS     = 5
CAL_BATCH_SIZE = 8
CAL_LR         = 1e-5
IMG_SIZE       = 224         # 모델 기대 입력 크기

class WorkerThread(QtCore.QThread):
    progress = QtCore.pyqtSignal(str)
    finished = QtCore.pyqtSignal(str)
    video_to_play = QtCore.pyqtSignal(str)

    def __init__(self, mode, data_dir, model_dir):
        super().__init__()
        self.mode      = mode
        self.data_dir  = data_dir
        self.model_dir = model_dir

        # 1) 어떤 .keras 모델이라도 로드
        model_path = os.path.join(self.model_dir, '10s_combined_model.keras')
        self.model = load_model(model_path)
        self.model.trainable = True  # fine-tune 가능하도록

    def run(self):
        try:
            if self.mode == 'calibrate':
                msg = self.calibrate()
            else:
                msg = self.evaluate()
        except Exception as e:
            msg = f"Error: {e}"
        self.finished.emit(msg)

    def crop_center_face(self, frame):
        """ 중앙 정사각형 크롭 → IMG_SIZE×IMG_SIZE """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb.shape
        side = min(h, w)
        cy, cx = h//2, w//2
        crop = rgb[cy-side//2:cy+side//2, cx-side//2:cx+side//2]
        if crop.shape[:2] != (IMG_SIZE, IMG_SIZE):
            crop = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))
        return crop

    def calibrate(self):
        self.progress.emit("[Calibration] 시작")
        Xc, yc = [], []

        for label, emo in enumerate(EMOTIONS):
            emo_dir = os.path.join(self.data_dir, emo)
            vids = [f for f in os.listdir(emo_dir)
                    if f.lower().endswith(('.mp4','.avi'))]
            if not vids:
                self.progress.emit(f"[Calibration] 영상 없음: {emo}")
                continue

            vid_path = os.path.join(emo_dir, random.choice(vids))
            self.progress.emit(f"[Calibration] 처리중: {emo}/{os.path.basename(vid_path)}")
            cap = cv2.VideoCapture(vid_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            buf_len = int(fps * SEG_SEC)
            buf = deque(maxlen=buf_len)

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                buf.append(frame)
                if len(buf) == buf_len:
                    # segment 중간 프레임 추출
                    mid_frame = buf[buf_len//2]
                    buf.clear()
                    img = self.crop_center_face(mid_frame)
                    img = preprocess_input(img.astype('float32'))
                    Xc.append(img)
                    yc.append(label)
            cap.release()

        if not Xc:
            return "[Calibration] 수집된 데이터 없음"

        Xc = np.stack(Xc, axis=0)  # (N, IMG_SIZE, IMG_SIZE, 3)
        yc = np.array(yc, dtype=np.int32)
        self.progress.emit(f"[Calibration] 샘플 수: {len(yc)}")

        # 모델 컴파일 & 학습
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(CAL_LR),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        self.model.fit(
            Xc, yc,
            epochs=CAL_EPOCHS,
            batch_size=CAL_BATCH_SIZE,
            verbose=1
        )
        # 저장
        save_path = os.path.join(self.model_dir, '10s_combined_model.keras')
        self.model.save(save_path)
        return "[Calibration] 완료 & 저장됨"

    def evaluate(self):
        self.progress.emit("[Evaluate] 시작")
        votes = []
        # 평가용 랜덤 감정 선택
        emo = random.choice(EMOTIONS)
        vids = [f for f in os.listdir(os.path.join(self.data_dir, emo))
                if f.lower().endswith(('.mp4','.avi'))]
        if not vids:
            return f"[Evaluate] {emo} 영상 없음"

        vid_path = os.path.join(self.data_dir, emo, random.choice(vids))
        self.progress.emit(f"[Evaluate] 재생: {emo}/{os.path.basename(vid_path)}")
        self.video_to_play.emit(vid_path)

        cap_v = cv2.VideoCapture(vid_path)
        cap_c = cv2.VideoCapture(0)
        fps = cap_v.get(cv2.CAP_PROP_FPS) or 30
        buf_len = int(fps * SEG_SEC)
        buf = deque(maxlen=buf_len)

        while True:
            rv, fv = cap_v.read()
            rc, fc = cap_c.read()
            if not rv:
                break
            if rc:
                buf.append(fc)
            if len(buf) == buf_len:
                mid = buf[buf_len//2]
                buf.clear()
                img = self.crop_center_face(mid)
                img = preprocess_input(img.astype('float32'))
                pred = self.model.predict(img[None,...], verbose=0)
                idx  = pred.argmax(axis=1)[0]
                votes.append(idx)
                self.progress.emit(f"[Evaluate] seg 예측 → {EMOTIONS[idx]}")
        cap_v.release(); cap_c.release()

        if votes:
            final = Counter(votes).most_common(1)[0][0]
            return f"[Evaluate] 실제={emo}, 예측={EMOTIONS[final]}"
        else:
            return "[Evaluate] 예측할 데이터 없음"

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Livetime GUI")
        self.resize(1000, 700)

        # 비디오 재생 위젯
        self.video_widget = QVideoWidget()
        self.player = QMediaPlayer(self)
        self.audio = QAudioOutput(self)
        self.player.setVideoOutput(self.video_widget)
        self.player.setAudioOutput(self.audio)

        # 로그
        self.log = QtWidgets.QTextEdit(readOnly=True)
        btn_cal = QtWidgets.QPushButton("Calibrate")
        btn_eval= QtWidgets.QPushButton("Evaluate")

        # 레이아웃
        lay = QtWidgets.QVBoxLayout()
        lay.addWidget(self.video_widget, 3)
        lay.addWidget(btn_cal)
        lay.addWidget(btn_eval)
        lay.addWidget(self.log, 2)
        w = QtWidgets.QWidget(); w.setLayout(lay)
        self.setCentralWidget(w)

        btn_cal.clicked.connect(lambda: self.start('calibrate'))
        btn_eval.clicked.connect(lambda: self.start('evaluate'))
        self.worker = None

    def start(self, mode):
        if self.worker and self.worker.isRunning():
            return
        self.log.clear()
        # data_dir, model_dir는 절대경로로!
        self.worker = WorkerThread(
            mode,
            data_dir='data',
            model_dir='model'
        )
        self.worker.progress.connect(self.log.append)
        self.worker.finished.connect(self.log.append)
        self.worker.video_to_play.connect(self.play)
        self.worker.start()

    def play(self, path):
        self.player.stop()
        self.player.setSource(QUrl.fromLocalFile(path))
        self.player.play()

if __name__=='__main__':
    app = QtWidgets.QApplication(sys.argv)
    mw = MainWindow(); mw.show()
    sys.exit(app.exec())
