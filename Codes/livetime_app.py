#!/usr/bin/env python3
# File: livetime_app.py

import os
import sys
import time
import random
import argparse
import subprocess
import cv2
import numpy as np
import tensorflow as tf

from collections import deque, Counter
from concurrent.futures import ThreadPoolExecutor

# 프로젝트 루트에 Face_feature.py, Face_model.py 가 있어야 합니다.
from Face_feature import extract_face_features_from_frames
from Face_model   import build_face_model

# ───────────────────────────────────────────────────────────
# 기본 설정
# ───────────────────────────────────────────────────────────
EMOTIONS       = ['neutral', 'sad', 'fear', 'happy']
SEG_SEC        = 1       # 1초 단위로 캡처
CAL_EPOCHS     = 50
CAL_BATCH_SIZE = 8
CAL_LR         = 1e-5

def play_audio(video_path):
    """FFplay 로 오디오만 재생하고 프로세스 리턴"""
    return subprocess.Popen(
        ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", video_path],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

def calibrate(data_dir, model_dir):
    """1) Calibration & Fine-tune"""
    model_path = os.path.join(model_dir, 'best_face_model.keras')
    mean_path  = os.path.join(model_dir, 'mean.npy')
    std_path   = os.path.join(model_dir, 'std.npy')
    os.makedirs(model_dir, exist_ok=True)

    print("[*] Loading model for calibration…")
    model = tf.keras.models.load_model(model_path)

    Xc_list, yc_list = [], []
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = []

        for label, emo in enumerate(EMOTIONS):
            emo_dir = os.path.join(data_dir, emo)
            vids = [f for f in os.listdir(emo_dir)
                    if f.lower().endswith(('.mp4','.avi'))]
            if not vids:
                print(f"[WARN] {emo} 디렉토리에 영상이 없습니다, 건너뜁니다.")
                continue

            vid_file = random.choice(vids)
            vid_path = os.path.join(emo_dir, vid_file)
            print(f"\n>> [{emo}] Calibration 영상: {vid_file}")

            # 1) 오디오 스레드
            audio_proc = play_audio(vid_path)

            # 2) 비디오 + 웹캠 동기 캡처
            cap_vid = cv2.VideoCapture(vid_path)
            cap_cam = cv2.VideoCapture(0)
            fps     = cap_vid.get(cv2.CAP_PROP_FPS) or 30
            buf_len = int(fps * SEG_SEC)
            buffer_cam = deque(maxlen=buf_len)

            win = f"Calibrate: {emo}"
            cv2.namedWindow(win, cv2.WINDOW_NORMAL)
            cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_FULLSCREEN)

            while True:
                ret_v, frame_v = cap_vid.read()
                ret_c, frame_c = cap_cam.read()
                if not ret_v:
                    break
                if ret_c:
                    buffer_cam.append(frame_c)

                cv2.imshow(win, frame_v)
                key = cv2.waitKey(int(1000/fps))
                if key == 27:
                    break

                # 준비된 buffer 만큼 모이면 백그라운드에서 피처 추출 예약
                if len(buffer_cam) == buf_len:
                    buf_copy = list(buffer_cam)  # snapshot
                    future = executor.submit(
                        extract_face_features_from_frames,
                        buf_copy, fps
                    )
                    futures.append((future, label))
                    buffer_cam.clear()

            cap_vid.release()
            cap_cam.release()
            cv2.destroyWindow(win)
            audio_proc.terminate()

        # 모든 피처 추출 완료될 때까지 대기
        for future, label in futures:
            feats, _ = future.result()
            if feats.size:
                Xc_list.append(feats[:,0])
                yc_list.append(label)

    # calibration 데이터 정리
    Xc = np.stack(Xc_list, axis=0)
    yc = np.array(yc_list, dtype=np.int32)

    # 정규화 파라미터 저장
    mean = Xc.mean(axis=0)
    std  = Xc.std(axis=0) + 1e-6
    np.save(mean_path, mean)
    np.save(std_path, std)

    # 파인튜닝
    print("\n[*] Fine-tuning 시작…")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(CAL_LR),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    Xc_in = ((Xc - mean) / std)[..., None, :]
    model.fit(
        Xc_in, yc,
        epochs=CAL_EPOCHS,
        batch_size=CAL_BATCH_SIZE,
        verbose=2
    )
    model.save(model_path)
    print("[*] Calibration & Fine-tuning 완료.\n")


def evaluate(data_dir, model_dir):
    """2) 평가(랜덤 영상 재생 + 예측)"""
    model_path = os.path.join(model_dir, 'best_face_model.keras')
    mean = np.load(os.path.join(model_dir, 'mean.npy'))
    std  = np.load(os.path.join(model_dir, 'std.npy'))

    print("[*] Loading model for evaluation…")
    model = tf.keras.models.load_model(model_path)

    emo = random.choice(EMOTIONS)
    test_dir = os.path.join(data_dir, emo)
    vids = [f for f in os.listdir(test_dir)
            if f.lower().endswith(('.mp4','.avi'))]
    if not vids:
        raise RuntimeError(f"[ERR] {emo} 디렉토리에 테스트 영상이 없습니다!")
    test_vid = random.choice(vids)
    test_path = os.path.join(test_dir, test_vid)
    print(f"\n>> [Eval] True: {emo}  영상: {test_vid}")

    audio_proc = play_audio(test_path)
    cap_vid = cv2.VideoCapture(test_path)
    cap_cam = cv2.VideoCapture(0)
    fps     = cap_vid.get(cv2.CAP_PROP_FPS) or 30
    buf_len = int(fps * SEG_SEC)
    buffer_cam = deque(maxlen=buf_len)
    votes = []

    win = "Evaluation"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN,
                          cv2.WINDOW_FULLSCREEN)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = []
        while True:
            ret_v, frame_v = cap_vid.read()
            ret_c, frame_c = cap_cam.read()
            if not ret_v:
                break
            if ret_c:
                buffer_cam.append(frame_c)

            cv2.imshow(win, frame_v)
            key = cv2.waitKey(int(1000/fps))
            if key == 27:
                break

            if len(buffer_cam) == buf_len:
                buf_copy = list(buffer_cam)
                futures.append(executor.submit(
                    extract_face_features_from_frames,
                    buf_copy, fps
                ))
                buffer_cam.clear()

        # 끝난 뒤에 예측 처리
        for fut in futures:
            feats, _ = fut.result()
            if feats.size:
                x = feats[:,0]
                x = (x - mean) / std
                x = x[None, None, :]
                pred = model.predict(x, verbose=0).argmax(axis=1)[0]
                votes.append(pred)

    cap_vid.release()
    cap_cam.release()
    cv2.destroyWindow(win)
    audio_proc.terminate()

    if votes:
        pred_idx = Counter(votes).most_common(1)[0][0]
        pred_emo = EMOTIONS[pred_idx]
        acc = float(pred_emo == emo) * 100
        print(f"\n>> Predicted: {pred_emo}  (Accuracy: {acc:.1f}%)\n")
    else:
        print("[WARN] 예측할 프레임이 부족합니다.\n")


def main():
    p = argparse.ArgumentParser(
        description="Livetime: calibration & evaluation for face-emotion"
    )
    p.add_argument('mode',
                   choices=['calibrate','evaluate'],
                   help="실행 모드를 선택하세요.")
    p.add_argument('--data-dir', default='data',
                   help="감정별 영상이 든 최상위 폴더")
    p.add_argument('--model-dir', default='model',
                   help="모델(.keras) 과 mean/std 가 저장된 폴더")
    args = p.parse_args()

    if args.mode == 'calibrate':
        calibrate(args.data_dir, args.model_dir)
    else:
        evaluate(args.data_dir, args.model_dir)


if __name__ == '__main__':
    main()
