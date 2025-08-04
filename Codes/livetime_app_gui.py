#!/usr/bin/env python3
import sys, os, random
import cv2
import numpy as np
import tensorflow as tf
from collections import deque, Counter
from concurrent.futures import ThreadPoolExecutor
from PyQt6 import QtCore, QtWidgets
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
from PyQt6.QtMultimediaWidgets import QVideoWidget
from PyQt6.QtCore import QUrl

# Face_feature.py에서는 extract_face_features_from_video만 있으므로,
# 프레임 리스트용 wrapper 작성
import tempfile
from Face_feature import extract_face_features_from_video

def extract_face_features_from_frames(frames, fps=30):
    with tempfile.NamedTemporaryFile(suffix='.avi', delete=False) as tmp:
        tmp_path = tmp.name
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(tmp_path, fourcc, fps, (w, h))
    for f in frames:
        out.write(f)
    out.release()
    feats, meta = extract_face_features_from_video(tmp_path)
    try:
        os.remove(tmp_path)
    except OSError:
        pass
    return feats, meta

# 설정
EMOTIONS       = ['neutral', 'sad', 'fear', 'happy']
SEG_SEC        = 1
CAL_EPOCHS     = 5
CAL_BATCH_SIZE = 8
CAL_LR         = 1e-5

class WorkerThread(QtCore.QThread):
    progress = QtCore.pyqtSignal(str)
    finished = QtCore.pyqtSignal(str)
    video_to_play = QtCore.pyqtSignal(str)

    def __init__(self, mode, data_dir, model_dir):
        super().__init__()
        self.mode = mode
        self.data_dir = data_dir
        self.model_dir = model_dir

    def run(self):
        try:
            if self.mode == 'calibrate':
                result = self.calibrate()
            else:
                result = self.evaluate()
        except Exception as e:
            result = f"Error: {e}"
        self.finished.emit(result)

    def calibrate(self):
        self.progress.emit("[Calibration] Loading model...")
        model_path = os.path.join(self.model_dir, '1s_best_face_model.keras')
        model = tf.keras.models.load_model(model_path)

        Xc_list, yc_list = [], []
        futures = []
        with ThreadPoolExecutor(max_workers=2) as exe:
            for label, emo in enumerate(EMOTIONS):
                emo_dir = os.path.join(self.data_dir, emo)
                vids = [f for f in os.listdir(emo_dir)
                        if f.lower().endswith(('.mp4', '.avi'))]
                if not vids:
                    self.progress.emit(f"[Calibration] No videos for {emo}")
                    continue

                vid_name = random.choice(vids)
                vid_path = os.path.join(emo_dir, vid_name)
                self.progress.emit(f"[Calibration] Playing {emo}: {vid_name}")
                # GUI에서 비디오 재생 요청
                self.video_to_play.emit(vid_path)

                cap_v = cv2.VideoCapture(vid_path)
                cap_c = cv2.VideoCapture(0)
                fps = cap_v.get(cv2.CAP_PROP_FPS) or 30
                buf_len = int(fps * SEG_SEC)
                buf = deque(maxlen=buf_len)

                while True:
                    ret_v, frame_v = cap_v.read()
                    ret_c, frame_c = cap_c.read()
                    if not ret_v:
                        break
                    if ret_c:
                        buf.append(frame_c)

                    if len(buf) == buf_len:
                        frames = list(buf)
                        buf.clear()
                        futures.append((exe.submit(
                            extract_face_features_from_frames,
                            frames, fps), label))
                        self.progress.emit(f"[Calibration] Submitted segment for extraction")

                cap_v.release()
                cap_c.release()

        # futures 결과 수집
        for fut, label in futures:
            feats, _ = fut.result()
            if feats.size:
                Xc_list.append(feats[:, 0])
                yc_list.append(label)
                self.progress.emit(f"[Calibration] Collected features shape: {feats.shape}")

        # 정규화
        Xc = np.stack(Xc_list, axis=0)
        yc = np.array(yc_list, dtype=np.int32)
        mean = Xc.mean(axis=0)
        std = Xc.std(axis=0) + 1e-6
        os.makedirs(self.model_dir, exist_ok=True)
        np.save(os.path.join(self.model_dir, 'mean.npy'), mean)
        np.save(os.path.join(self.model_dir, 'std.npy'), std)
        self.progress.emit("[Calibration] Normalization params saved")

        # 파인튜닝
        model.compile(
            optimizer=tf.keras.optimizers.Adam(CAL_LR),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        Xc_in = ((Xc - mean) / std)[..., None, :]
        self.progress.emit("[Calibration] Fine-tuning started")
        model.fit(Xc_in, yc,
                  epochs=CAL_EPOCHS,
                  batch_size=CAL_BATCH_SIZE,
                  verbose=0)
        model.save(model_path)
        return "[Calibration] Complete & Fine-tuned"

    def evaluate(self):
        self.progress.emit("[Evaluate] Loading model...")
        model = tf.keras.models.load_model(
            os.path.join(self.model_dir, '1s_best_face_model.keras'))
        mean = np.load(os.path.join(self.model_dir, 'mean.npy'))
        std = np.load(os.path.join(self.model_dir, 'std.npy'))

        emo = random.choice(EMOTIONS)
        test_dir = os.path.join(self.data_dir, emo)
        vids = [f for f in os.listdir(test_dir)
                if f.lower().endswith(('.mp4', '.avi'))]
        if not vids:
            raise RuntimeError(f"[Evaluate] No videos for {emo}")

        vid_name = random.choice(vids)
        vid_path = os.path.join(test_dir, vid_name)
        self.progress.emit(f"[Evaluate] Playing {emo}: {vid_name}")
        self.video_to_play.emit(vid_path)

        cap_v = cv2.VideoCapture(vid_path)
        cap_c = cv2.VideoCapture(0)
        fps = cap_v.get(cv2.CAP_PROP_FPS) or 30
        buf_len = int(fps * SEG_SEC)
        buf = deque(maxlen=buf_len)
        votes = []

        while True:
            ret_v, frame_v = cap_v.read()
            ret_c, frame_c = cap_c.read()
            if not ret_v:
                break
            if ret_c:
                buf.append(frame_c)

            if len(buf) == buf_len:
                feats, _ = extract_face_features_from_frames(
                    list(buf), fps)
                buf.clear()
                if feats.size:
                    x = (feats[:, 0] - mean) / std
                    pred = model.predict(x[None, None, :], verbose=0)
                    idx = pred.argmax(axis=1)[0]
                    votes.append(idx)
                    self.progress.emit(f"[Evaluate] Segment predicted: {EMOTIONS[idx]}")

        cap_v.release()
        cap_c.release()

        if votes:
            final = Counter(votes).most_common(1)[0][0]
            return f"[Evaluate] True: {emo}, Predicted: {EMOTIONS[final]}"
        return "[Evaluate] No Prediction"

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Livetime GUI")
        self.resize(1000, 700)

        # 위젯 설정
        self.video_widget = QVideoWidget()
        self.player = QMediaPlayer(self)
        self.audio_output = QAudioOutput(self)
        self.player.setVideoOutput(self.video_widget)
        self.player.setAudioOutput(self.audio_output)

        self.log = QtWidgets.QTextEdit(readOnly=True)
        btn_cal = QtWidgets.QPushButton("Calibrate & Fine-tune")
        btn_eval = QtWidgets.QPushButton("Evaluate Random Video")

        # 레이아웃
        central = QtWidgets.QWidget()
        vbox = QtWidgets.QVBoxLayout(central)
        vbox.addWidget(self.video_widget, 3)
        vbox.addWidget(btn_cal)
        vbox.addWidget(btn_eval)
        vbox.addWidget(self.log, 2)
        self.setCentralWidget(central)

        btn_cal.clicked.connect(lambda: self.start_task('calibrate'))
        btn_eval.clicked.connect(lambda: self.start_task('evaluate'))

        self.worker = None

    def start_task(self, mode):
        if self.worker and self.worker.isRunning():
            return
        self.log.clear()
        self.worker = WorkerThread(mode, 'data', 'model')
        self.worker.progress.connect(self.log.append)
        self.worker.finished.connect(self.log.append)
        self.worker.video_to_play.connect(self.play_video)
        self.worker.start()

    def play_video(self, path):
        # 로컬 파일 URL 설정 후 재생
        self.player.stop()
        url = QUrl.fromLocalFile(path)
        self.player.setSource(url)
        self.player.play()

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
