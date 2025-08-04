#!/usr/bin/env python3
import sys, os, random, tempfile, cv2, numpy as np, tensorflow as tf, time
from collections import deque, Counter
from concurrent.futures import ThreadPoolExecutor
from PyQt6 import QtCore, QtWidgets
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
from PyQt6.QtMultimediaWidgets import QVideoWidget
from PyQt6.QtCore import QUrl
import keras
keras.config.enable_unsafe_deserialization()
from concurrent.futures import as_completed, TimeoutError

# –– multimodal feature extractors
from Face_feature import extract_face_features_from_video
from EYE_feature import extract_features_from_video
from multimodal_training2 import SplitModality

# wrappers to turn frames into feature arrays

def extract_face_features_from_frames(frames, fps=30):
    with tempfile.NamedTemporaryFile(suffix='.avi', delete=False) as tmp:
        tmp_path = tmp.name
    h, w = frames[0].shape[:2]
    out = cv2.VideoWriter(tmp_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (w, h))
    for f in frames:
        out.write(f)
    out.release()
    feats, meta = extract_face_features_from_video(tmp_path)
    try: os.remove(tmp_path)
    except: pass
    return feats, meta


def extract_eye_features_from_frames(frames, fps=30):
    with tempfile.NamedTemporaryFile(suffix='.avi', delete=False) as tmp:
        tmp_path = tmp.name
    h, w = frames[0].shape[:2]
    out = cv2.VideoWriter(tmp_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (w, h))
    for f in frames:
        out.write(f)
    out.release()
    feats, meta = extract_features_from_video(tmp_path)
    try: os.remove(tmp_path)
    except: pass
    return feats, meta

# configuration
EMOTIONS       = ['neutral', 'sad', 'fear', 'happy']
SEG_SEC        = 1
CAL_EPOCHS     = 500
CAL_BATCH_SIZE = 8
CAL_LR         = 1e-4
MODEL_FILENAME = 'best_multi_model.keras'
MEAN_FACE_FILE = 'mean_face.npy'
STD_FACE_FILE  = 'std_face.npy'
MEAN_EYE_FILE  = 'mean_eye.npy'
STD_EYE_FILE   = 'std_eye.npy'

def path_join(*args):
    return os.path.join(*args)

class QtEpochCallback(tf.keras.callbacks.Callback):
            def __init__(self, thread):
                super().__init__()
                self.thread = thread
            def on_epoch_end(self, epoch, logs=None):
                loss = logs.get('loss')
                acc  = logs.get('accuracy')
                self.thread.progress.emit(
                    f"[Calibration] Epoch {epoch+1}/{CAL_EPOCHS} – loss: {loss:.4f}, acc: {acc:.4f}"
                )

class WorkerThread(QtCore.QThread):
    progress      = QtCore.pyqtSignal(str)
    finished      = QtCore.pyqtSignal(str)
    video_to_play = QtCore.pyqtSignal(str)

    def __init__(self, mode, data_dir, model_dir):
        super().__init__()
        self.mode     = mode
        self.data_dir = data_dir
        self.model_dir= model_dir

    def run(self):
        try:
            if self.mode == 'calibrate':
                res = self.calibrate()
            else:
                res = self.evaluate()
        except Exception as e:
            res = f"Error: {e}"
        self.finished.emit(res)

    def calibrate(self):
        self.progress.emit("[Calibration] Loading multimodal model...")
        model = tf.keras.models.load_model(
            path_join(self.model_dir, MODEL_FILENAME),
            custom_objects={'SplitModality': SplitModality},
            compile=False
        )
        self.progress.emit("[Calibration] Model loaded successfully")
        # Debug: show data directory
        self.progress.emit(f"[Calibration] Data directory: {self.data_dir}")

        # open live camera
        cap_c = None
        self.progress.emit("[Calibration] Trying to open camera...")
        for idx in range(4):
            cam = cv2.VideoCapture(idx)
            if cam.isOpened():
                cap_c = cam
                self.progress.emit(f"[Calibration] Opened camera at index {idx}")
                break
            cam.release()
        if cap_c is None:
            self.progress.emit("[Calibration] Error: Cannot open any camera (tried 0–3)")
            return "Camera open failed"

        Xf_list, Xe_list, y_list = [], [], []
        futures = []
        with ThreadPoolExecutor(max_workers=2) as exe:
            for label, emo in enumerate(EMOTIONS):
                emo_dir = path_join(self.data_dir, emo)
                # Debug: check folder exists
                exists = os.path.isdir(emo_dir)
                self.progress.emit(f"[Calibration] Checking folder for {emo}: {emo_dir}, exists={exists}")
                vids = []
                if exists:
                    vids = [f for f in os.listdir(emo_dir) if f.lower().endswith(('.mp4','.avi'))]
                # Debug: list of videos
                self.progress.emit(f"[Calibration] Found videos for {emo}: {vids}")
                if not vids:
                    self.progress.emit(f"[Calibration] No videos for {emo}")
                    continue

                vid_path = path_join(emo_dir, random.choice(vids))
                self.progress.emit(f"[Calibration] Playing {emo} video: {os.path.basename(vid_path)}")
                self.video_to_play.emit(vid_path)

                cap_v = cv2.VideoCapture(vid_path)
                fps   = cap_v.get(cv2.CAP_PROP_FPS) or 30
                buf_len = int(fps * SEG_SEC)
                buf = deque(maxlen=buf_len)

                while True:
                    ret_v, _ = cap_v.read()
                    ret_c, frame_c = cap_c.read()
                    if not ret_v:
                        break
                    if ret_c:
                        buf.append(frame_c)
                    if len(buf) == buf_len:
                        frames = list(buf)
                        buf.clear()
                        futures.append((
                            exe.submit(extract_face_features_from_frames, frames, fps),
                            exe.submit(extract_eye_features_from_frames,  frames, fps),
                            label
                        ))
                        self.progress.emit("[Calibration] Segment submitted")
                    time.sleep(1.0/fps)
                cap_v.release()                
        cap_c.release()

        if not futures:
            self.progress.emit("[Calibration] No segments were submitted—aborting.")
            return "Calibration aborted: no data"
        # process in completion order, with timeouts & error handling
        for f_fut, e_fut, label in futures:
            try:
                f_feats, _ = f_fut.result(timeout=60)   # wait max 60s per segment
                e_feats, _ = e_fut.result(timeout=60)
            except TimeoutError:
                self.progress.emit(f"[Calibration] Timeout extracting features for label {label}")
                continue
            except Exception as ex:
                self.progress.emit(f"[Calibration] Extraction error for label {label}: {ex}")
                continue

            if f_feats.size and e_feats.size:
                Xf_list.append(f_feats[:,0])
                Xe_list.append(e_feats[:,0])
                y_list.append(label)

        self.progress.emit("[Calibration] Collected all segments")

        # normalize & save params
        Xf = np.stack(Xf_list)
        Xe = np.stack(Xe_list)
        y  = np.array(y_list, dtype=np.int32)
        mean_face, std_face = Xf.mean(0), Xf.std(0)+1e-6
        mean_eye, std_eye   = Xe.mean(0), Xe.std(0)+1e-6
        os.makedirs(self.model_dir, exist_ok=True)
        np.save(path_join(self.model_dir, MEAN_FACE_FILE), mean_face)
        np.save(path_join(self.model_dir, STD_FACE_FILE),  std_face)
        np.save(path_join(self.model_dir, MEAN_EYE_FILE),  mean_eye)
        np.save(path_join(self.model_dir, STD_EYE_FILE),   std_eye)
        self.progress.emit("[Calibration] Normalization params saved")

        # fine-tune
        model.compile(
            optimizer=tf.keras.optimizers.Adam(CAL_LR),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        Xf_in = ((Xf - mean_face) / std_face)[:, None, :]
        Xe_in = ((Xe - mean_eye)  / std_eye)   [:, None, :]
        self.progress.emit("[Calibration] Fine-tuning started")
        model.fit([Xe_in, Xf_in], y,
                  epochs=CAL_EPOCHS,
                  batch_size=CAL_BATCH_SIZE,
                  verbose=0,
                  callbacks=[QtEpochCallback(self)])
        model.save(path_join(self.model_dir, MODEL_FILENAME))
        self.progress.emit("[Calibration] Complete & Fine-tuned")
        return "Calibration done"

    def evaluate(self):
        self.progress.emit("[Evaluate] Loading multimodal model...")
        model = tf.keras.models.load_model(
            path_join(self.model_dir, MODEL_FILENAME),
            custom_objects={'SplitModality': SplitModality},
            compile=False
        )
        self.progress.emit("[Evaluate] Model loaded successfully")

        mean_face = np.load(path_join(self.model_dir, MEAN_FACE_FILE))
        std_face  = np.load(path_join(self.model_dir, STD_FACE_FILE))
        mean_eye  = np.load(path_join(self.model_dir, MEAN_EYE_FILE))
        std_eye   = np.load(path_join(self.model_dir, STD_EYE_FILE))

        emo = random.choice(EMOTIONS)
        emo_dir = path_join(self.data_dir, emo)
        vids = [f for f in os.listdir(emo_dir) if f.lower().endswith(('.mp4','.avi'))]
        vid_path = path_join(emo_dir, random.choice(vids))
        self.progress.emit(f"[Evaluate] Playing {emo} video: {os.path.basename(vid_path)}")
        self.video_to_play.emit(vid_path)

        cap_v = cv2.VideoCapture(vid_path)
        cap_c = cv2.VideoCapture(0)
        if not cap_c.isOpened():
            self.progress.emit("[Evaluate] Warning: Cannot open camera")
        fps = cap_v.get(cv2.CAP_PROP_FPS) or 30
        buf = deque(maxlen=int(fps*SEG_SEC))
        votes = []

        while True:
            ret_v, _ = cap_v.read()
            ret_c, frame_c = cap_c.read()
            if not ret_v: break
            if ret_c: buf.append(frame_c)
            if len(buf) == buf.maxlen:
                frames=list(buf); buf.clear()
                f_feats,_ = extract_face_features_from_frames(frames, fps)
                e_feats,_ = extract_eye_features_from_frames(frames, fps)
                if f_feats.size and e_feats.size:
                    xf  = ((e_feats[:,0]-mean_eye)/std_eye)[None,None,:]
                    xf2 = ((f_feats[:,0]-mean_face)/std_face)[None,None,:]
                    pred=model.predict([xf,xf2],verbose=0)
                    votes.append(pred.argmax())
                    self.progress.emit(f"[Evaluate] Segment predicted: {EMOTIONS[votes[-1]]}")
            time.sleep(1.0/fps)

        cap_v.release(); cap_c.release()
        result = "No Prediction"
        if votes:
            final = Counter(votes).most_common(1)[0][0]
            result = f"True: {emo}, Predicted: {EMOTIONS[final]}"
        self.progress.emit(f"[Evaluate] {result}")
        return result

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Livetime GUI (Multimodal)")
        self.resize(1000,700)

        # video display
        self.video_widget = QVideoWidget()
        self.player       = QMediaPlayer(self)
        self.audio_output = QAudioOutput(self)
        self.player.setVideoOutput(self.video_widget)
        self.player.setAudioOutput(self.audio_output)

        # controls & log
        self.log      = QtWidgets.QTextEdit(readOnly=True)
        btn_cal      = QtWidgets.QPushButton("Calibrate & Fine-tune")
        btn_eval     = QtWidgets.QPushButton("Evaluate Random Video")

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.video_widget,3)
        layout.addWidget(btn_cal)
        layout.addWidget(btn_eval)
        layout.addWidget(self.log,2)
        container = QtWidgets.QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        btn_cal.clicked.connect(lambda: self.start_task('calibrate'))
        btn_eval.clicked.connect(lambda: self.start_task('evaluate'))

        self.worker = None

    def start_task(self, mode):
        if self.worker and self.worker.isRunning(): return
        self.log.clear()
        base = os.path.dirname(os.path.abspath(__file__))  # use script directory
        data_dir = path_join(base,'data')
        model_dir= path_join(base,'model')
        self.worker = WorkerThread(mode, data_dir, model_dir)
        self.worker.progress.connect(self.log.append)
        self.worker.finished.connect(self.log.append)
        self.worker.video_to_play.connect(self.play_video)
        self.worker.start()

    def play_video(self, path):
        self.player.stop()
        url = QUrl.fromLocalFile(path)
        self.player.setSource(url)
        self.player.play()

if __name__=='__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
