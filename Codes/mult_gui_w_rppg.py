#!/usr/bin/env python3
import sys, os, random, tempfile, cv2, numpy as np, tensorflow as tf, time
from collections import deque, Counter
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from PyQt6 import QtCore, QtWidgets
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
from PyQt6.QtMultimediaWidgets import QVideoWidget
from PyQt6.QtCore import QUrl
import keras
keras.config.enable_unsafe_deserialization()
from multimodal_training2 import SplitModality

# configuration
EMOTIONS        = ['neutral','sad','fear','happy']
SEG_SEC         = 1
CAL_EPOCHS      = 500
CAL_BATCH_SIZE  = 8
CAL_LR          = 1e-4
MODEL_FILENAME  = 'best_multi_rppg_model.keras'
MEAN_FACE_FILE  = 'mean_face.npy'
STD_FACE_FILE   = 'std_face.npy'
MEAN_EYE_FILE   = 'mean_eye.npy'
STD_EYE_FILE    = 'std_eye.npy'
MEAN_RPPG_FILE  = 'mean_rppg.npy'
STD_RPPG_FILE   = 'std_rppg.npy'

# Haar cascade for face ROI
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

def path_join(*args): return os.path.join(*args)

class QtEpochCallback(tf.keras.callbacks.Callback):
    def __init__(self, thread):
        super().__init__(); self.thread = thread
    def on_epoch_end(self, epoch, logs=None):
        loss = logs.get('loss'); acc = logs.get('accuracy')
        self.thread.progress.emit(
            f"[Calibration] Epoch {epoch+1}/{CAL_EPOCHS} – loss: {loss:.4f}, acc: {acc:.4f}"
        )

# extract face features wrapper
from Face_feature import extract_face_features_from_video
from EYE_feature  import extract_features_from_video

def extract_face_features_from_frames(frames, fps=30):
    with tempfile.NamedTemporaryFile(suffix='.avi', delete=False) as tmp:
        tmp_path = tmp.name
    h,w = frames[0].shape[:2]
    out = cv2.VideoWriter(tmp_path, cv2.VideoWriter_fourcc(*'XVID'), fps,(w,h))
    for f in frames: out.write(f)
    out.release()
    feats, meta = extract_face_features_from_video(tmp_path)
    try: os.remove(tmp_path)
    except: pass
    return feats, meta

def extract_eye_features_from_frames(frames, fps=30):
    with tempfile.NamedTemporaryFile(suffix='.avi', delete=False) as tmp:
        tmp_path = tmp.name
    h,w = frames[0].shape[:2]
    out = cv2.VideoWriter(tmp_path, cv2.VideoWriter_fourcc(*'XVID'), fps,(w,h))
    for f in frames: out.write(f)
    out.release()
    feats, meta = extract_features_from_video(tmp_path)
    try: os.remove(tmp_path)
    except: pass
    return feats, meta

def extract_rppg_features_from_frames(frames, fps=30):
    # 얼굴 ROI 검출
    gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,1.3,5)
    if len(faces)==0:
        x,y,w,h = 0,0,frames[0].shape[1],frames[0].shape[0]
    else:
        x,y,w,h = faces[0]
    # 5×4 그리드 → 20개 패치
    ph, pw = h//4, w//5
    T = len(frames)
    raw = np.zeros((20, T), float)
    for t, f in enumerate(frames):
        roi = f[y:y+h, x:x+w]
        for idx in range(20):
            r, c = divmod(idx, 5)
            patch = roi[r*ph:(r+1)*ph, c*pw:(c+1)*pw]
            raw[idx, t] = np.nanmean(patch[:,:,1])
    # 시간축을 50프레임으로 보간
    old_t = np.arange(T)
    new_t = np.linspace(0, T-1, 50)
    up = np.vstack([np.interp(new_t, old_t, raw[ch]) for ch in range(20)])
    return up, None   # shape (20, 50)

class WorkerThread(QtCore.QThread):
    progress      = QtCore.pyqtSignal(str)
    finished      = QtCore.pyqtSignal(str)
    video_to_play = QtCore.pyqtSignal(str)

    def __init__(self, mode, data_dir, model_dir):
        super().__init__()
        self.mode, self.data_dir, self.model_dir = mode, data_dir, model_dir

    def run(self):
        try:
            res = self.calibrate() if self.mode=='calibrate' else self.evaluate()
        except Exception as e:
            res = f"Error: {e}"
        self.finished.emit(res)

    def calibrate(self):
        # —————————————————————
        # 1) load existing calibration data (if any)
        # —————————————————————
        calib_path = path_join(self.model_dir, 'calib_data.npz')
        if os.path.exists(calib_path):
            data = np.load(calib_path)
            Xf_old, Xe_old, Xr_old, y_old = data['Xf'], data['Xe'], data['Xr'], data['y']
            self.progress.emit(f"[Calibration] Loaded {len(y_old)} prior segments")
        else:
            Xf_old = Xe_old = Xr_old = y_old = None

        self.progress.emit("[Calibration] Loading model...")
        model = tf.keras.models.load_model(
            path_join(self.model_dir, MODEL_FILENAME),
            custom_objects={'SplitModality':SplitModality}, compile=False
        )
        self.progress.emit("[Calibration] Model loaded")
        # open cam
        cap_c=None
        for idx in range(4):
            cam=cv2.VideoCapture(idx)
            if cam.isOpened(): cap_c=cam; break
            cam.release()
        if cap_c is None: return "Camera failed"

        Xf,Xe,Xr,y=[],[],[],[]
        futures=[]
        with ThreadPoolExecutor(max_workers=3) as exe:
            for label, emo in enumerate(EMOTIONS):
                emo_dir=path_join(self.data_dir,emo)
                vids=[f for f in os.listdir(emo_dir) if f.lower().endswith(('.mp4','.avi'))]
                if not vids: continue
                vid=path_join(emo_dir,random.choice(vids))
                self.video_to_play.emit(vid)
                cap_v=cv2.VideoCapture(vid)
                fps=cap_v.get(cv2.CAP_PROP_FPS) or 30
                buf=deque(maxlen=int(fps*SEG_SEC))
                while True:
                    rv,_=cap_v.read(); rc,frm=cap_c.read()
                    if not rv: break
                    if rc: buf.append(frm)
                    if len(buf)==buf.maxlen:
                        seg=list(buf); buf.clear()
                        futures.append((
                            exe.submit(extract_face_features_from_frames, seg,fps),
                            exe.submit(extract_eye_features_from_frames, seg,fps),
                            exe.submit(extract_rppg_features_from_frames, seg,fps),
                            label
                        ))
                    time.sleep(1.0/fps)
                cap_v.release()
        cap_c.release()

        for ff,ef,rf,label in futures:
            try:
                f_feats,_=ff.result(timeout=60)
                e_feats,_=ef.result(timeout=60)
                r_feats,_=rf.result(timeout=60)
            except TimeoutError:
                continue
            if f_feats.size and e_feats.size and r_feats.size:
                Xf.append(f_feats[:,0])       # face feature 벡터
                Xe.append(e_feats[:,0])       # eye feature 벡터
                Xr.append(r_feats.T)          # (50,20)로 쌓기
                y.append(label)

        Xf,Xe,Xr=np.stack(Xf),np.stack(Xe),np.stack(Xr)
        y=np.array(y)
        # concatenate with previous data
        if Xf_old is not None:
            Xf = np.concatenate([Xf_old, Xf], axis=0)
            Xe = np.concatenate([Xe_old, Xe], axis=0)
            Xr = np.concatenate([Xr_old, Xr], axis=0)
            y  = np.concatenate([y_old,  y ], axis=0)
        # save merged calibration set for next time
        os.makedirs(self.model_dir, exist_ok=True)
        np.savez(calib_path, Xf=Xf, Xe=Xe, Xr=Xr, y=y)
        # normalize & save
        mf,sf=Xf.mean(0),Xf.std(0)+1e-6
        me,se=Xe.mean(0),Xe.std(0)+1e-6
        # flatten across all samples & time to get channel‐wise stats
        flat_r = Xr.reshape(-1, Xr.shape[2])   # (N*50, 20)
        mr = flat_r.mean(axis=0)
        sr = flat_r.std(axis=0) + 1e-6
        os.makedirs(self.model_dir,exist_ok=True)
        np.save(path_join(self.model_dir,MEAN_FACE_FILE),mf)
        np.save(path_join(self.model_dir,STD_FACE_FILE),sf)
        np.save(path_join(self.model_dir,MEAN_EYE_FILE),me)
        np.save(path_join(self.model_dir,STD_EYE_FILE),se)
        np.save(path_join(self.model_dir,MEAN_RPPG_FILE),mr)
        np.save(path_join(self.model_dir,STD_RPPG_FILE),sr)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(CAL_LR),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        Xf_in = ((Xf - mf) / sf)[:, None, :]     # (N,1,F_face)
        Xe_in = ((Xe - me) / se)[:, None, :]     # (N,1,F_eye)
        Xr_in =  (Xr - mr) / sr                  # (N,50,20) — no extra time‐axis dimension
        model.fit([Xe_in,Xf_in,Xr_in],y,epochs=CAL_EPOCHS,
                  batch_size=CAL_BATCH_SIZE,verbose=0,
                  callbacks=[QtEpochCallback(self)])
        model.save(path_join(self.model_dir,MODEL_FILENAME))
        return "Calibration done"

    def evaluate(self):
        self.progress.emit("[Evaluate] Loading model...")
        model=tf.keras.models.load_model(
            path_join(self.model_dir,MODEL_FILENAME),
            custom_objects={'SplitModality':SplitModality},compile=False
        )
        mf,sf=np.load(path_join(self.model_dir,MEAN_FACE_FILE)),np.load(path_join(self.model_dir,STD_FACE_FILE))
        me,se=np.load(path_join(self.model_dir,MEAN_EYE_FILE)),np.load(path_join(self.model_dir,STD_EYE_FILE))
        mr,sr=np.load(path_join(self.model_dir,MEAN_RPPG_FILE)),np.load(path_join(self.model_dir,STD_RPPG_FILE))
        emo=random.choice(EMOTIONS)
        emo_dir=path_join(self.data_dir,emo)
        vids=[f for f in os.listdir(emo_dir) if f.lower().endswith(('.mp4','.avi'))]
        vid=path_join(emo_dir,random.choice(vids))
        self.video_to_play.emit(vid)
        cap_v,cap_c=cv2.VideoCapture(vid),cv2.VideoCapture(0)
        fps=cap_v.get(cv2.CAP_PROP_FPS) or 30
        buf=deque(maxlen=int(fps*SEG_SEC)); votes=[]
        while True:
            rv,_=cap_v.read(); rc,frm=cap_c.read()
            if not rv: break
            if rc: buf.append(frm)
            if len(buf)==buf.maxlen:
                seg=list(buf); buf.clear()
                f_feats,_=extract_face_features_from_frames(seg,fps)
                e_feats,_=extract_eye_features_from_frames(seg,fps)
                r_feats,_=extract_rppg_features_from_frames(seg,fps)
                if f_feats.size and e_feats.size and r_feats.size:
                    xf=((e_feats[:,0]-me)/se)[None,None,:]
                    xf2=((f_feats[:,0]-mf)/sf)[None,None,:]
                    xr  = ((r_feats.T - mr)/sr)[None,:,:]          # (1,50,20)
                    pred=model.predict([xf,xf2,xr],verbose=0)
                    votes.append(pred.argmax())
                    self.progress.emit(f"[Evaluate] Predicted: {EMOTIONS[votes[-1]]}")
            time.sleep(1.0/fps)
        cap_v.release(); cap_c.release()
        result="No Prediction"
        if votes:
            final=Counter(votes).most_common(1)[0][0]
            result=f"True: {emo}, Predicted: {EMOTIONS[final]}"
        return result

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Livetime GUI (Multimodal)")
        self.resize(1000,700)
        self.video_widget=QVideoWidget()
        self.player=QMediaPlayer(self)
        self.audio_output=QAudioOutput(self)
        self.player.setVideoOutput(self.video_widget)
        self.player.setAudioOutput(self.audio_output)
        self.log=QtWidgets.QTextEdit(readOnly=True)
        btn_cal=QtWidgets.QPushButton("Calibrate & Fine-tune")
        btn_eval=QtWidgets.QPushButton("Evaluate Random Video")
        layout=QtWidgets.QVBoxLayout()
        layout.addWidget(self.video_widget,3)
        layout.addWidget(btn_cal)
        layout.addWidget(btn_eval)
        layout.addWidget(self.log,2)
        c=QtWidgets.QWidget(); c.setLayout(layout)
        self.setCentralWidget(c)
        btn_cal.clicked.connect(lambda: self.start_task('calibrate'))
        btn_eval.clicked.connect(lambda: self.start_task('evaluate'))
        self.worker=None
    def start_task(self,mode):
        if self.worker and self.worker.isRunning(): return
        self.log.clear()
        base=os.path.dirname(os.path.abspath(__file__))
        data_dir=path_join(base,'data'); model_dir=path_join(base,'model')
        self.worker=WorkerThread(mode,data_dir,model_dir)
        self.worker.progress.connect(self.log.append)
        self.worker.finished.connect(self.log.append)
        self.worker.video_to_play.connect(self.play_video)
        self.worker.start()
    def play_video(self,path):
        self.player.stop(); url=QUrl.fromLocalFile(path)
        self.player.setSource(url); self.player.play()

if __name__=='__main__':
    app=QtWidgets.QApplication(sys.argv)
    w=MainWindow(); w.show(); sys.exit(app.exec())
