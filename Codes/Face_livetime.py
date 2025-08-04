import os
import sys
import random
import cv2
import numpy as np
import tensorflow as tf
import subprocess
from collections import deque, Counter
import tempfile

# ───────────────────────────────────────────────────────────
# 0) 환경 설정
# ───────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.realpath(__file__))
DATA_DIR    = os.path.join(BASE_DIR, 'data')                # data/<emotion> 폴더
MODEL_DIR   = os.path.join(BASE_DIR, 'model')
MODEL_PATH  = os.path.join(MODEL_DIR, 'best_face_model.keras')
MEAN_PATH   = os.path.join(MODEL_DIR, 'mean.npy')
STD_PATH    = os.path.join(MODEL_DIR, 'std.npy')

EMOTIONS        = ['neutral', 'sad', 'fear', 'happy']
CAL_EPOCHS      = 50
CAL_BATCH_SIZE  = 8
CAL_LR          = 1e-5
SEG_SEC         = 1  # 1초 단위

# ───────────────────────────────────────────────────────────
# 모듈 탐색 경로 추가
# ───────────────────────────────────────────────────────────
sys.path.append(BASE_DIR)
from Face_feature import extract_face_features_from_video
from Face_model   import build_face_model

# ───────────────────────────────────────────────────────────
# 프레임 리스트 → feature 추출용 래퍼
# ───────────────────────────────────────────────────────────
def extract_face_features_from_frames(frames, fps=30):
    """
    frames: list of BGR numpy images
    fps:    임시 비디오의 FPS
    returns: feats, meta (same as extract_face_features_from_video)
    """
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

# ───────────────────────────────────────────────────────────
# 1) 캘리브레이션 & 파인튜닝
# ───────────────────────────────────────────────────────────
def calibrate_and_finetune():
    print("Loading pre-trained model for calibration...")
    model = tf.keras.models.load_model(MODEL_PATH)

    Xc_list, yc_list = [], []

    for idx, emo in enumerate(EMOTIONS):
        emo_dir = os.path.join(DATA_DIR, emo)
        vids = [f for f in os.listdir(emo_dir) if f.lower().endswith(('.avi','.mp4'))]
        if not vids:
            print(f"[WARN] No videos in {emo_dir}, skipping.")
            continue

        vid_file = random.choice(vids)
        vid_path = os.path.join(emo_dir, vid_file)
        print(f"\n>> Calibrating for '{emo}' using {vid_file}")

        # 오디오 재생 (ffplay 필요)
        audio_proc = subprocess.Popen([
            "ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", vid_path
        ])

        cap_vid    = cv2.VideoCapture(vid_path)
        cap_cam    = cv2.VideoCapture(0)
        fps        = cap_vid.get(cv2.CAP_PROP_FPS) or 30
        buf_len    = int(fps * SEG_SEC)
        buffer_cam = deque(maxlen=buf_len)

        win = "Calibration Video"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        while True:
            ret_v, frame_v = cap_vid.read()
            ret_c, frame_c = cap_cam.read()
            if not ret_v:
                break
            if ret_c:
                buffer_cam.append(frame_c)

            cv2.imshow(win, frame_v)

            if len(buffer_cam) == buf_len:
                feats, _ = extract_face_features_from_frames(list(buffer_cam), fps=fps)
                if feats.size > 0:
                    Xc_list.append(feats[:, 0])
                    yc_list.append(idx)
                buffer_cam.clear()

            if cv2.waitKey(int(1000/fps)) & 0xFF == 27:
                break

        cap_vid.release()
        cap_cam.release()
        cv2.destroyWindow(win)
        audio_proc.terminate()

    # 데이터 준비
    Xc = np.stack(Xc_list, axis=0)
    yc = np.array(yc_list, dtype=np.int32)

    # 정규화 파라미터 계산 & 저장
    mean = Xc.mean(axis=0)
    std  = Xc.std(axis=0) + 1e-6
    os.makedirs(MODEL_DIR, exist_ok=True)
    np.save(MEAN_PATH, mean)
    np.save(STD_PATH, std)

    # 파인튜닝
    model.compile(
        optimizer=tf.keras.optimizers.Adam(CAL_LR),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    Xc_in = ((Xc - mean) / std)[..., None, :]
    print("\n>> Fine-tuning model on calibration data...")
    model.fit(
        Xc_in, yc,
        epochs=CAL_EPOCHS,
        batch_size=CAL_BATCH_SIZE,
        verbose=2
    )
    model.save(MODEL_PATH)
    print("Calibration & fine-tuning complete.\n")

# ───────────────────────────────────────────────────────────
# 2) 평가 (테스트 영상 재생 + 예측)
# ───────────────────────────────────────────────────────────
def evaluate_with_video():
    print("Loading fine-tuned model for evaluation...")
    model = tf.keras.models.load_model(MODEL_PATH)
    mean  = np.load(MEAN_PATH)
    std   = np.load(STD_PATH)

    emo = random.choice(EMOTIONS)
    test_dir = os.path.join(DATA_DIR, emo)
    vids = [f for f in os.listdir(test_dir) if f.lower().endswith(('.avi','.mp4'))]
    if not vids:
        raise RuntimeError(f"No test videos in {test_dir}")
    test_vid = random.choice(vids)
    test_path = os.path.join(test_dir, test_vid)
    print(f"\n>> Evaluating on '{emo}' using {test_vid}")

    # 오디오 재생
    audio_proc = subprocess.Popen([
        "ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", test_path
    ])

    cap_vid    = cv2.VideoCapture(test_path)
    cap_cam    = cv2.VideoCapture(0)
    fps        = cap_vid.get(cv2.CAP_PROP_FPS) or 30
    buf_len    = int(fps * SEG_SEC)
    buffer_cam = deque(maxlen=buf_len)
    votes = []

    win = "Test Video"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while True:
        ret_v, frame_v = cap_vid.read()
        ret_c, frame_c = cap_cam.read()
        if not ret_v:
            break
        if ret_c:
            buffer_cam.append(frame_c)

        cv2.imshow(win, frame_v)

        if len(buffer_cam) == buf_len:
            feats, _ = extract_face_features_from_frames(list(buffer_cam), fps=fps)
            if feats.size > 0:
                x = feats[:, 0]
                x = (x - mean) / std
                x = x[None, None, :]
                pred = model.predict(x).argmax(axis=1)[0]
                votes.append(pred)
            buffer_cam.clear()

        if cv2.waitKey(int(1000/fps)) & 0xFF == 27:
            break

    cap_vid.release()
    cap_cam.release()
    cv2.destroyWindow(win)
    audio_proc.terminate()

    if votes:
        pred_idx = Counter(votes).most_common(1)[0][0]
        pred_emo = EMOTIONS[pred_idx]
        acc      = float(pred_emo == emo) * 100
        print(f"\nTrue: {emo}, Predicted: {pred_emo}, Accuracy: {acc:.1f}%")
    else:
        print("No predictions were made.")

if __name__ == '__main__':
    calibrate_and_finetune()
    evaluate_with_video()
