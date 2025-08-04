#!/usr/bin/env python3
import sys
import os
import random
import tempfile
import cv2
import numpy as np
import tensorflow as tf
import time
from collections import deque, Counter
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from PyQt6 import QtCore, QtWidgets
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
from PyQt6.QtMultimediaWidgets import QVideoWidget
from PyQt6.QtCore import QUrl
import keras
keras.config.enable_unsafe_deserialization()

# Multimodal 1s model imports
from multimodal_training2 import SplitModality
from Face_feature import extract_face_features_from_video
from EYE_feature import extract_features_from_video

# MicroExp3D 15s model imports
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input, Conv3D, MaxPool3D, Dropout, Flatten, Dense
from tensorflow.keras.models import Model as KerasModel
from tensorflow.keras.callbacks import Callback
import mediapipe as mp
from scipy.signal import medfilt

# --- Global Configuration ---
EMOTIONS         = ['neutral','sad','fear','happy']
# 1s model
SEG_SEC_1        = 1
CAL_EPOCHS_1     = 50
CAL_BATCH_SIZE_1 = 8
CAL_LR_1         = 1e-4
MODEL_FILE_1     = 'best_multi_rppg_model.keras'
MEAN_FACE_FILE   = 'mean_face.npy'
STD_FACE_FILE    = 'std_face.npy'
MEAN_EYE_FILE    = 'mean_eye.npy'
STD_EYE_FILE     = 'std_eye.npy'
MEAN_RPPG_FILE   = 'mean_rppg.npy'
STD_RPPG_FILE    = 'std_rppg.npy'
CALIB_1_PATH     = 'calib_data_1.npz'
# 15s model
SEG_SEC_15       = 15
FPS              = 30
BUFFER_SIZE_15   = SEG_SEC_15 * FPS
ONOFF_LEN        = FPS
CAL_EPOCHS_15    = 50
CAL_BATCH_15     = 4
CAL_LR_15        = 1e-5
MODEL_FILE_15    = 'final_microexp3dstcnn.keras'
CALIB_15_PATH    = 'calib_data_15.npz'
# FaceMesh & preprocessing params for 15s model
OUT_SIZE             = 112
PAD                  = 20
BRIGHTNESS_THRESHOLD = 40
N_LM                 = 478
FB_PARAMS = {
    'pyr_scale':0.5, 'levels':3, 'winsize':15,
    'iterations':3, 'poly_n':5, 'poly_sigma':1.2, 'flags':0
}

# MediaPipe Face Mesh for 15s
mp_face = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=False,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
last_bbox = None

# Utility
path_join = os.path.join

# --- 1s model feature extraction wrappers ---
def extract_face_features_from_frames_1s(frames, fps=30):
    with tempfile.NamedTemporaryFile(suffix='.avi', delete=False) as tmp:
        tmp_path = tmp.name
    h, w = frames[0].shape[:2]
    out = cv2.VideoWriter(tmp_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (w,h))
    for f in frames:
        out.write(f)
    out.release()
    feats, meta = extract_face_features_from_video(tmp_path)
    try: os.remove(tmp_path)
    except: pass
    return feats, meta


def extract_eye_features_from_frames_1s(frames, fps=30):
    with tempfile.NamedTemporaryFile(suffix='.avi', delete=False) as tmp:
        tmp_path = tmp.name
    h, w = frames[0].shape[:2]
    out = cv2.VideoWriter(tmp_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (w,h))
    for f in frames:
        out.write(f)
    out.release()
    feats, meta = extract_features_from_video(tmp_path)
    try: os.remove(tmp_path)
    except: pass
    return feats, meta

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def extract_rppg_features_from_frames_1s(frames, fps=30):
    gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,1.3,5)
    if len(faces)==0:
        x,y,w,h = 0,0,frames[0].shape[1],frames[0].shape[0]
    else:
        x,y,w,h = faces[0]
    ph, pw = h//4, w//5
    T = len(frames)
    raw = np.zeros((20, T), float)
    for t, f in enumerate(frames):
        roi = f[y:y+h, x:x+w]
        for idx in range(20):
            r, c = divmod(idx, 5)
            patch = roi[r*ph:(r+1)*ph, c*pw:(c+1)*pw]
            raw[idx, t] = np.nanmean(patch[:,:,1])
    old_t = np.arange(T)
    new_t = np.linspace(0, T-1, 50)
    up = np.vstack([np.interp(new_t, old_t, raw[ch]) for ch in range(20)])
    return up, None

# --- 15s model preprocessing & model builder ---
def align_frame_15(frame):
    global last_bbox
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = mp_face.process(rgb)
    det = bool(res.multi_face_landmarks)
    if not det:
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l,a,b = cv2.split(lab)
        cl = clahe.apply(l)
        merged = cv2.merge((cl,a,b))
        frame = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
        frame = cv2.bilateralFilter(frame,5,75,75)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = mp_face.process(rgb)
        det = bool(res.multi_face_landmarks)
    if not det and last_bbox is None:
        return np.zeros((OUT_SIZE,OUT_SIZE,3),np.uint8), False, np.full((N_LM,2),np.nan)
    if not det:
        x1,y1,x2,y2 = last_bbox
        crop = frame[y1:y2, x1:x2]
        aligned = cv2.resize(crop,(OUT_SIZE,OUT_SIZE))
        return aligned, False, np.full((N_LM,2),np.nan)
    lm = res.multi_face_landmarks[0].landmark
    xs = np.array([p.x*w for p in lm]); ys = np.array([p.y*h for p in lm])
    x1 = max(0,int(xs.min())-PAD); y1 = max(0,int(ys.min())-PAD)
    x2 = min(w,int(xs.max())+PAD); y2 = min(h,int(ys.max())+PAD)
    last_bbox = (x1,y1,x2,y2)
    crop = frame[y1:y2, x1:x2]
    aligned = cv2.resize(crop,(OUT_SIZE,OUT_SIZE))
    if aligned.mean()<BRIGHTNESS_THRESHOLD:
        return aligned, False, np.full((N_LM,2),np.nan)
    lm_arr = np.array([[p.x*w,p.y*h] for p in lm],dtype=np.float32)
    return aligned, True, lm_arr


def compute_combined_scores_15(frames, mask, lms,
                               n_base=5, w_pixel=1.0,
                               w_flow=0.5, w_landmark=0.5,
                               smooth_kernel=5):
    valid = np.where(mask)[0]
    if valid.size==0: return np.array([]), valid
    gray = np.dot(frames[valid][...,:3], [0.299,0.587,0.114])
    p = np.mean(np.abs(gray - ((gray[:n_base].mean(0)+gray[-n_base:].mean(0))/2)), axis=(1,2))
    f = np.zeros_like(p); prev = gray[0]
    for i,c in enumerate(gray[1:],1):
        flow = cv2.calcOpticalFlowFarneback(prev,c,None,**FB_PARAMS)
        f[i] = np.mean(np.linalg.norm(flow,axis=2)); prev = c
    lm_v = lms[valid]
    l = np.zeros_like(p); prev_l = lm_v[0]
    for i,cur in enumerate(lm_v[1:],1):
        l[i] = np.mean(np.linalg.norm(cur-prev_l,axis=1)); prev_l=cur
    def norm(x): return (x-x.min())/(x.max()-x.min()+1e-6)
    comb = w_pixel*norm(p) + w_flow*norm(f) + w_landmark*norm(l)
    if smooth_kernel>1 and smooth_kernel%2==1:
        comb = medfilt(comb, smooth_kernel)
    comb = np.convolve(comb, np.ones(5)/5, mode='same')
    return comb, valid


def find_top_segment_15(comb, valid, length=FPS):
    if comb.size==0: return 0,-1
    idx = np.argmax(comb); peak = valid[idx]; st = peak-length//2
    return int(st), int(peak)


def pad_or_crop_15(frames, start, length):
    T = frames.shape[0]; end = start+length
    s,e = max(start,0), min(end,T)
    clip = frames[s:e]
    pb,pa = max(0,-start), max(0,end-T)
    if pb or pa:
        clip = np.pad(clip, ((pb,pa),(0,0),(0,0),(0,0)), mode='constant')
    if clip.shape[0]>length: clip = clip[:length]
    if clip.shape[0]<length:
        clip = np.pad(clip, ((0,length-clip.shape[0]),(0,0),(0,0),(0,0)), mode='constant')
    return clip


def preprocess_clip_15(frames):
    arr = np.stack(frames).astype(np.float32)/255.0
    gray = np.dot(arr[...,:3], [0.299,0.587,0.114])
    out = np.zeros((gray.shape[0],64,64),dtype=np.float32)
    for i in range(gray.shape[0]):
        out[i] = cv2.resize(gray[i], (64,64))
    return out[...,np.newaxis]


def build_microexp3dstcnn(
    input_shape=(ONOFF_LEN*3,64,64,1),
    num_classes=len(EMOTIONS),
    dropout_rate=0.3
):
    inp = Input(shape=input_shape, name='video_in')
    x = Conv3D(32,(15,3,3),strides=(1,1,1),padding='valid',activation='relu')(inp)
    x = MaxPool3D((3,3,3),strides=(3,3,3),padding='valid')(x)
    x = Dropout(dropout_rate)(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    out = Dense(num_classes, activation='softmax')(x)
    model = KerasModel(inp, out)
    model.compile('adam', 'sparse_categorical_crossentropy', ['accuracy'])
    return model


def load_micro_model():
    if os.path.exists(MODEL_FILE_15):
        try:
            return load_model(MODEL_FILE_15, compile=False)
        except Exception:
            pass
    return build_microexp3dstcnn()

# Epoch callback for both models
class QtEpochCallback(tf.keras.callbacks.Callback):
    def __init__(self, thread):
        super().__init__()
        self.thread = thread
    def on_epoch_end(self, epoch, logs=None):
        loss = logs.get('loss')
        acc = logs.get('accuracy')
        self.thread.progress.emit(f"[Epoch {epoch+1}] loss:{loss:.4f}, acc:{acc:.4f}")

# WorkerThread handles both calibrate & evaluate for 1s and 15s
class WorkerThread(QtCore.QThread):
    progress      = QtCore.pyqtSignal(str)
    finished      = QtCore.pyqtSignal(str)
    video_to_play = QtCore.pyqtSignal(str)

    def __init__(self, mode, data_dir, model_dir):
        super().__init__()
        self.mode = mode; self.data_dir = data_dir; self.model_dir = model_dir

    def run(self):
        try:
            if self.mode == 'calibrate':
                self.calibrate()
                res = 'Calibration complete for both models'
            else:
                self.evaluate()
                res = 'Evaluation complete for both models'
        except Exception as e:
            res = f"Error: {e}"
        self.finished.emit(res)

    def calibrate(self):
        # Ensure tempvideo folder exists
        temp_folder = path_join(os.getcwd(), 'tempvideo')
        os.makedirs(temp_folder, exist_ok=True)
        # Prepare camera
        cap_c = cv2.VideoCapture(0)
        if not cap_c.isOpened():
            raise RuntimeError("Camera failed to open for calibration recording")
        # Frame size and settings
        ret, frame0 = cap_c.read()
        if not ret:
            cap_c.release()
            raise RuntimeError("Failed to read frame from camera")
        h, w = frame0.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        
        # Play each emotion video and record camera per emotion
        for label, emo in enumerate(EMOTIONS):
            emo_dir = path_join(self.data_dir, emo)
            vids = [f for f in os.listdir(emo_dir) if f.lower().endswith(('.mp4','.avi'))]
            if not vids:
                self.progress.emit(f"[Calibration] No videos for {emo}, skipping")
                continue
            vid_path = path_join(emo_dir, random.choice(vids))
            # Set up writer for this emotion
            out_path = path_join(temp_folder, f"{emo}.avi")
            out_cam = cv2.VideoWriter(out_path, fourcc, FPS, (w, h))
            # Play video
            self.video_to_play.emit(vid_path)
            time.sleep(0.5)
            cap_v = cv2.VideoCapture(vid_path)
            fps_v = cap_v.get(cv2.CAP_PROP_FPS) or FPS
            while cap_v.isOpened():
                rv_v, _ = cap_v.read()
                rv_c, cam_frame = cap_c.read()
                if not rv_v:
                    break
                if rv_c:
                    out_cam.write(cam_frame)
                time.sleep(1.0 / fps_v)
            cap_v.release()
            out_cam.release()
            self.progress.emit(f"[Calibration] Saved camera recording for {emo} at {out_path}")
        cap_c.release()
        
        # Load recorded camera videos and extract segments
        Xf, Xe, Xr, y1 = [], [], [], []
        X15, y15 = [], []
        for label, emo in enumerate(EMOTIONS):
            rec_path = path_join(temp_folder, f"{emo}.avi")
            cap_rec = cv2.VideoCapture(rec_path)
            recorded = []
            while True:
                ret, frm = cap_rec.read()
                if not ret:
                    break
                recorded.append(frm)
            cap_rec.release()
            total = len(recorded)
            # 1s segment
            start1 = 0
            end1 = int(SEG_SEC_1 * FPS)
            if total >= end1:
                seg = recorded[start1:end1]
                f_feats,_ = extract_face_features_from_frames_1s(seg, FPS)
                e_feats,_ = extract_eye_features_from_frames_1s(seg, FPS)
                r_feats,_ = extract_rppg_features_from_frames_1s(seg, FPS)
                if f_feats.size and e_feats.size and r_feats.size:
                    Xf.append(f_feats[:,0]); Xe.append(e_feats[:,0]); Xr.append(r_feats.T); y1.append(label)
            # 15s segment
            if total >= int(SEG_SEC_15 * FPS):
                clip = recorded[:int(SEG_SEC_15 * FPS)]
                arr = np.stack(clip)
                comb, valid = compute_combined_scores_15(arr, np.ones(len(arr),bool), np.zeros((len(arr),N_LM,2)))
                st, _ = find_top_segment_15(comb, valid, length=FPS)
                window = pad_or_crop_15(arr, st, ONOFF_LEN * 3)
                sample = preprocess_clip_15(window)[None,...]
                X15.append(sample[0]); y15.append(label)
        # Convert lists to arrays
        Xf = np.stack(Xf) if Xf else np.empty((0,0))
        Xe = np.stack(Xe) if Xe else np.empty((0,0))
        Xr = np.stack(Xr) if Xr else np.empty((0,0,0))
        y1 = np.array(y1) if y1 else np.array([])
        X15 = np.stack(X15) if X15 else np.empty((0, ONOFF_LEN*3, 64,64,1))
        y15 = np.array(y15) if y15 else np.array([])
        # Save calibration datasets
        os.makedirs(self.model_dir, exist_ok=True)
        np.savez(path_join(self.model_dir, CALIB_1_PATH), Xf=Xf, Xe=Xe, Xr=Xr, y=y1)
        np.savez(path_join(self.model_dir, CALIB_15_PATH), X=X15, y=y15)
        self.progress.emit(f"[Calibration] Saved calib data: {CALIB_1_PATH}, {CALIB_15_PATH}")
        
        # 1s model training with epoch accuracy
        self.progress.emit("[Calibration] Starting 1s model training")
        model1 = tf.keras.models.load_model(path_join(self.model_dir, MODEL_FILE_1),
                                            custom_objects={'SplitModality':SplitModality}, compile=False)
        mf, sf = Xf.mean(0), Xf.std(0)+1e-6
        me, se = Xe.mean(0), Xe.std(0)+1e-6
        flat_r = Xr.reshape(-1, Xr.shape[2]) if Xr.size else np.empty((0, Xr.shape[2]))
        mr, sr = flat_r.mean(0), flat_r.std(0)+1e-6
        model1.compile(optimizer=tf.keras.optimizers.Adam(CAL_LR_1),
                       loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model1.fit([
            ((Xe - me) / se)[:,None,:],
            ((Xf - mf) / sf)[:,None,:],
            ((Xr - mr) / sr)
        ], y1,
        epochs=CAL_EPOCHS_1, batch_size=CAL_BATCH_SIZE_1,
        callbacks=[QtEpochCallback(self)], verbose=0)
        model1.save(path_join(self.model_dir, MODEL_FILE_1))
        self.progress.emit("[Calibration] 1s model trained")
        # 15s model training with epoch accuracy
        if X15.size:
            self.progress.emit("[Calibration] Starting 15s model training")
            model2 = load_micro_model()
            model2.compile(optimizer=tf.keras.optimizers.Adam(CAL_LR_15),
                           loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            model2.fit(X15, y15,
                       epochs=CAL_EPOCHS_15, batch_size=CAL_BATCH_15,
                       callbacks=[QtEpochCallback(self)], verbose=0)
            model2.save(path_join(self.model_dir, MODEL_FILE_15))
            self.progress.emit("[Calibration] 15s model trained")
        else:
            self.progress.emit("[Calibration] No data for 15s model")
        self.progress.emit(f"[Calibration] Saved calib data: {CALIB_1_PATH}, {CALIB_15_PATH}")
    def evaluate(self):
        # Load models and stats
        model1 = tf.keras.models.load_model(
            path_join(self.model_dir, MODEL_FILE_1),
            custom_objects={'SplitModality': SplitModality}, compile=False
        )
        mf, sf = np.load(path_join(self.model_dir, MEAN_FACE_FILE)), np.load(path_join(self.model_dir, STD_FACE_FILE))
        me, se = np.load(path_join(self.model_dir, MEAN_EYE_FILE)), np.load(path_join(self.model_dir, STD_EYE_FILE))
        mr, sr = np.load(path_join(self.model_dir, MEAN_RPPG_FILE)), np.load(path_join(self.model_dir, STD_RPPG_FILE))
        model2 = load_micro_model()
        model2.compile()

        # Select a single video for evaluation
        emo = random.choice(EMOTIONS)
        emo_dir = path_join(self.data_dir, emo)
        vids = [f for f in os.listdir(emo_dir) if f.lower().endswith(('.mp4','.avi'))]
        if not vids:
            self.progress.emit(f"[Evaluate] No videos for {emo}")
            return
        vid = path_join(emo_dir, random.choice(vids))
        self.video_to_play.emit(vid)
        time.sleep(0.5)
        cap_v = cv2.VideoCapture(vid)
        cap_c = cv2.VideoCapture(0)
        if not cap_v.isOpened() or not cap_c.isOpened():
            self.progress.emit("[Evaluate] Camera or video failed")
            return

        fps = cap_v.get(cv2.CAP_PROP_FPS) or FPS
        buf1 = deque(maxlen=int(fps * SEG_SEC_1))
        buf15 = deque(maxlen=BUFFER_SIZE_15)
        votes1, votes2 = [], []
        count15 = 0
        start_time = time.time()
        # Collect and predict: two 15s cycles or until 70s
        # 1) 1초 모델 15회 예측
        for i in range(15):
            buf1 = deque(maxlen=int(fps * SEG_SEC_1))
            while len(buf1) < buf1.maxlen:
                rv_v, _ = cap_v.read()
                rv_c, frame = cap_c.read()
                if not rv_v or not rv_c:
                    continue
                buf1.append(frame)
            # 예측
            f_feats,_ = extract_face_features_from_frames_1s(list(buf1), fps)
            e_feats,_ = extract_eye_features_from_frames_1s(list(buf1), fps)
            r_feats,_ = extract_rppg_features_from_frames_1s(list(buf1), fps)
            pred1 = model1.predict([
                ((e_feats[:,0]-me)/se)[None,None,:],
                ((f_feats[:,0]-mf)/sf)[None,None,:],
                ((r_feats.T-mr)/sr)[None,:,:]
            ], verbose=0)
            label1 = pred1.argmax()
            votes1.append(label1)
            self.progress.emit(f"[Evaluate] 1s #{i+1}: {EMOTIONS[label1]}")

        # 2) 15초 모델 1회 예측
        buf15 = deque(maxlen=BUFFER_SIZE_15)
        while len(buf15) < buf15.maxlen:
            rv_v, _ = cap_v.read()
            rv_c, frame = cap_c.read()
            if not rv_v or not rv_c:
                continue
            buf15.append(frame)
        arr15 = np.stack(buf15)
        comb, valid = compute_combined_scores_15(arr15, np.ones(len(arr15),bool),
                                                np.zeros((len(arr15),N_LM,2)))
        st, _ = find_top_segment_15(comb, valid, length=FPS)
        clip15 = pad_or_crop_15(arr15, st, ONOFF_LEN*3)
        p15 = model2.predict(preprocess_clip_15(clip15)[None,...], verbose=0)
        votes2.append(p15.argmax())
        self.progress.emit(f"[Evaluate] 15s #1: {EMOTIONS[p15.argmax()]}")

        # 3) 1초 모델 추가 15회 예측
        for i in range(15):
            buf1 = deque(maxlen=int(fps * SEG_SEC_1))
            while len(buf1) < buf1.maxlen:
                rv_v, _ = cap_v.read()
                rv_c, frame = cap_c.read()
                if not rv_v or not rv_c:
                    continue
                buf1.append(frame)
            pred1 = model1.predict([
                ((extract_eye_features_from_frames_1s(list(buf1), fps)[0][:,0]-me)/se)[None,None,:],
                ((extract_face_features_from_frames_1s(list(buf1), fps)[0][:,0]-mf)/sf)[None,None,:],
                ((extract_rppg_features_from_frames_1s(list(buf1), fps)[0].T-mr)/sr)[None,:,:]
            ], verbose=0)
            label1 = pred1.argmax()
            votes1.append(label1)
            self.progress.emit(f"[Evaluate] 1s #16+{i+1}: {EMOTIONS[label1]}")

        # 4) 15초 모델 두 번째 예측
        buf15 = deque(maxlen=BUFFER_SIZE_15)
        while len(buf15) < buf15.maxlen:
            rv_v, _ = cap_v.read()
            rv_c, frame = cap_c.read()
            if not rv_v or not rv_c:
                continue
            buf15.append(frame)
        arr15 = np.stack(buf15)
        comb, valid = compute_combined_scores_15(arr15, np.ones(len(arr15),bool),
                                                np.zeros((len(arr15),N_LM,2)))
        st, _ = find_top_segment_15(comb, valid, length=FPS)
        clip15 = pad_or_crop_15(arr15, st, ONOFF_LEN*3)
        p15 = model2.predict(preprocess_clip_15(clip15)[None,...], verbose=0)
        votes2.append(p15.argmax())
        self.progress.emit(f"[Evaluate] 15s #2: {EMOTIONS[p15.argmax()]}")
       
        cap_v.release(); cap_c.release()

        # Emit final aggregated results for 1s model
        if votes1:
            final1 = Counter(votes1).most_common(1)[0][0]
            self.progress.emit(f"[Evaluate] 1s Final: {EMOTIONS[final1]}")
        else:
            self.progress.emit("[Evaluate] 1s Final: No predictions")
        # Emit both 15s cycle results
        if votes2:
            for idx, l2 in enumerate(votes2, 1):
                self.progress.emit(f"[Evaluate] 15s Cycle {idx} Final: {EMOTIONS[l2]}")
        else:
            self.progress.emit("[Evaluate] 15s Final: No predictions")

# --- GUI Application ---
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Dual Realtime Emotion GUI')
        self.resize(1000,700)
        self.video_widget = QVideoWidget()
        self.player = QMediaPlayer(self)
        self.audio_output = QAudioOutput(self)
        self.player.setVideoOutput(self.video_widget)
        self.player.setAudioOutput(self.audio_output)
        self.log = QtWidgets.QTextEdit(readOnly=True)
        btn_cal = QtWidgets.QPushButton('Calibrate Models')
        btn_eval= QtWidgets.QPushButton('Evaluate Models')
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.video_widget,3)
        layout.addWidget(btn_cal)
        layout.addWidget(btn_eval)
        layout.addWidget(self.log,2)
        container = QtWidgets.QWidget(); container.setLayout(layout)
        self.setCentralWidget(container)
        btn_cal.clicked.connect(lambda: self.start_task('calibrate'))
        btn_eval.clicked.connect(lambda: self.start_task('evaluate'))
        self.worker = None

    def start_task(self, mode):
        if self.worker and self.worker.isRunning(): return
        self.log.clear()
        base = os.path.dirname(os.path.abspath(__file__))
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

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())
