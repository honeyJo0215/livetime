#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys, os, random, time, cv2, numpy as np, tensorflow as tf
from collections import deque, Counter
from PyQt6 import QtCore, QtWidgets
from PyQt6.QtCore import QUrl
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
from PyQt6.QtMultimediaWidgets import QVideoWidget

# ───────────────────────────────────────────────────────────────
# 1. 설정
# ───────────────────────────────────────────────────────────────
EMOTIONS       = ['neutral','sad','fear','happy']
SEG_SEC        = 10             # 분절 길이 (초)
WINDOW_SIZE    = 10             # 샘플링할 프레임 개수
CAL_EPOCHS     = 50
CAL_BATCH_SIZE = 32
CAL_LR         = 1e-5
IMG_SIZE       = 224

RESNET_WEIGHTS  = 'model/resnet50_ft.weights.h5'
CNN_MODEL_FILE  = 'model/cnn_10s_model.keras'


class WorkerThread(QtCore.QThread):
    progress      = QtCore.pyqtSignal(str)
    finished      = QtCore.pyqtSignal(str)
    video_to_play = QtCore.pyqtSignal(str)

    def __init__(self, mode, data_dir='data', model_dir='model'):
        super().__init__()
        self.mode     = mode
        self.data_dir = data_dir
        self.model_dir= model_dir

        # ResNet50 feature extractor
        self.resnet = tf.keras.applications.ResNet50(
            weights=None, include_top=False, pooling='avg',
            input_shape=(IMG_SIZE, IMG_SIZE, 3)
        )
        self.resnet.load_weights(RESNET_WEIGHTS)
        self.resnet.trainable = False

        # CNN classifier load (Calibration 후 저장할 모델)
        cnn_h5 = CNN_MODEL_FILE.replace('.keras', '.h5')
        self.cnn = tf.keras.models.load_model(cnn_h5)

    def crop_center(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h,w,_ = rgb.shape
        side  = min(h,w)
        cy, cx= h//2, w//2
        c = rgb[cy-side//2:cy+side//2, cx-side//2:cx+side//2]
        if c.shape[:2] != (IMG_SIZE, IMG_SIZE):
            c = cv2.resize(c, (IMG_SIZE, IMG_SIZE))
        return c

    def sample_frames(self, frames, num):
        if len(frames) < num:
            return []
        idxs = np.linspace(0, len(frames)-1, num, dtype=int)
        return [frames[i] for i in idxs]

    def run(self):
        try:
            if self.mode=='calibrate':
                msg = self.calibrate()
            else:
                msg = self.evaluate()
        except Exception as e:
            msg = f"Error: {e}"
        self.finished.emit(msg)

    def calibrate(self):
        self.progress.emit("[Calibration] 시작: 모델 로드 및 컴파일")
        # compile
        self.cnn.trainable = True
        self.cnn.compile(
            optimizer=tf.keras.optimizers.Adam(CAL_LR),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        # 웹캠 한 번만 열기
        cap_cam = cv2.VideoCapture(0)
        if not cap_cam.isOpened():
            return "[Calibration] Error: 웹캠 열기 실패"

        X_list, y_list = [], []

        # 감정별로
        for label, emo in enumerate(EMOTIONS):
            emo_dir = os.path.join(self.data_dir, emo)
            vids = [f for f in os.listdir(emo_dir) if f.lower().endswith(('.mp4','.avi'))]
            if not vids:
                self.progress.emit(f"[Calibration] No vids for {emo}")
                continue

            # 랜덤 영상 선택 & 재생
            vid  = random.choice(vids)
            path = os.path.join(emo_dir, vid)
            self.progress.emit(f"[Calibration] Playing {emo}/{vid}")
            self.video_to_play.emit(path)

            # 영상 분절 수 계산
            cap_v = cv2.VideoCapture(path)
            fps_v   = cap_v.get(cv2.CAP_PROP_FPS) or 30.0
            total_f = cap_v.get(cv2.CAP_PROP_FRAME_COUNT) or 0
            duration= total_f / fps_v
            segments= int(duration // SEG_SEC)
            cap_v.release()

            # 각 분절마다 웹캠에서 10초간 연속 캡처 → 10프레임 샘플링 → ResNet50 특성 추출
            for seg in range(segments):
                self.progress.emit(f"[Calibration] {emo} segment {seg+1}/{segments}")
                frames = []
                start_t = time.time()
                while time.time() - start_t < SEG_SEC:
                    ret, frame = cap_cam.read()
                    if ret:
                        frames.append(frame)
                    QtCore.QThread.msleep(5)

                samples = self.sample_frames(frames, WINDOW_SIZE)
                if len(samples) != WINDOW_SIZE:
                    self.progress.emit(f"[Calibration] Warning: segment{seg+1} 샘플 부족")
                    continue

                imgs = np.stack([self.crop_center(f) for f in samples], axis=0)
                imgs = tf.keras.applications.resnet50.preprocess_input(imgs.astype('float32'))
                feats = self.resnet.predict(imgs, verbose=0)  # (10,2048)

                X_list.append(feats)
                y_list.append(label)

        # 웹캠 해제
        cap_cam.release()

        if not X_list:
            return "[Calibration] 데이터 수집 실패"

        # 배열로 변환
        X = np.stack(X_list, axis=0)       # (N_segments,10,2048)
        y = np.array(y_list, dtype=int)    # (N_segments,)

        self.progress.emit(f"[Calibration] 총 {len(y)} samples → 모델 학습 시작")
        # 한 번에 학습
        self.cnn.fit(
            X, y,
            epochs=CAL_EPOCHS,
            batch_size=CAL_BATCH_SIZE,
            verbose=1
        )

        # 저장
        self.cnn.save(CNN_MODEL_FILE, save_format='keras_v3')
        return "[Calibration] 완료 & 저장"

    def evaluate(self):
        self.progress.emit("[Evaluate] 시작: 모델 로드")
        model = tf.keras.models.load_model(CNN_MODEL_FILE)
        model.trainable = False

        # 랜덤 영상 선택
        emo = random.choice(EMOTIONS)
        emo_dir = os.path.join(self.data_dir, emo)
        vids = [f for f in os.listdir(emo_dir) if f.lower().endswith(('.mp4','.avi'))]
        if not vids:
            return "[Evaluate] No vids for evaluation"
        vid = random.choice(vids)
        path = os.path.join(emo_dir, vid)
        self.progress.emit(f"[Evaluate] Playing {emo}/{vid}")
        self.video_to_play.emit(path)

        # 분절 수 계산
        cap_v = cv2.VideoCapture(path)
        fps_v   = cap_v.get(cv2.CAP_PROP_FPS) or 30.0
        total_f = cap_v.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        duration= total_f / fps_v
        segments= int(duration // SEG_SEC)
        cap_v.release()

        # 웹캠 한 번만 열기
        cap_cam = cv2.VideoCapture(0)
        if not cap_cam.isOpened():
            return "[Evaluate] Error: 웹캠 열기 실패"

        votes = []
        for seg in range(segments):
            self.progress.emit(f"[Evaluate] segment {seg+1}/{segments} 캡처 중")
            frames = []
            start_t = time.time()
            while time.time() - start_t < SEG_SEC:
                ret, frame = cap_cam.read()
                if ret:
                    frames.append(frame)
                QtCore.QThread.msleep(5)

            samples = self.sample_frames(frames, WINDOW_SIZE)
            if len(samples) != WINDOW_SIZE:
                continue

            imgs = np.stack([self.crop_center(f) for f in samples], axis=0)
            imgs = tf.keras.applications.resnet50.preprocess_input(imgs.astype('float32'))
            feats = self.resnet.predict(imgs, verbose=0)
            pred  = model.predict(feats[np.newaxis,...], verbose=0)
            idx   = int(np.argmax(pred, axis=1)[0])
            votes.append(idx)
            self.progress.emit(f"[Evaluate] segment{seg+1}→{EMOTIONS[idx]}")

        cap_cam.release()

        if not votes:
            return "[Evaluate] No Prediction"
        final = Counter(votes).most_common(1)[0][0]
        return f"[Evaluate] FinalPred={EMOTIONS[final]}"


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Livetime GUI")
        self.resize(1000,700)

        # 비디오 재생용
        self.video_widget = QVideoWidget()
        self.player       = QMediaPlayer(self)
        self.audio        = QAudioOutput(self)
        self.player.setVideoOutput(self.video_widget)
        self.player.setAudioOutput(self.audio)

        # 로그 + 버튼
        self.log     = QtWidgets.QTextEdit(readOnly=True)
        btn_cal     = QtWidgets.QPushButton("Calibrate")
        btn_eval    = QtWidgets.QPushButton("Evaluate")

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.video_widget, 3)
        layout.addWidget(btn_cal)
        layout.addWidget(btn_eval)
        layout.addWidget(self.log, 2)

        container = QtWidgets.QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        btn_cal.clicked.connect(lambda: self.start('calibrate'))
        btn_eval.clicked.connect(lambda: self.start('evaluate'))
        self.worker = None

    def start(self, mode):
        if self.worker and self.worker.isRunning():
            return
        self.log.clear()
        self.worker = WorkerThread(mode, data_dir='data', model_dir='model')
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
