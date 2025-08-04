import os, sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input, Dense, Concatenate, Lambda, Multiply, Add
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import register_keras_serializable
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# ───────────────────────────────────────────────────────────────
# 0) 경로 및 설정
# ───────────────────────────────────────────────────────────────
VIDEO_ROOT    = "/home/bcml1/2025_EMOTION/face_video"
LABEL_ROOT    = "/home/bcml1/2025_EMOTION/DEAP_5labels"
EYE_FEAT_DIR  = "features_imp"        # eye features
FACE_FEAT_DIR = "face_features"       # face features
RESULT_DIR    = "multi_results2"

PRETRAIN_EYE_MODEL  = os.path.join("eye_model",  "best_eye.weights.h5")
PRETRAIN_FACE_MODEL = os.path.join("face_model", "best_face.weights.h5")

NUM_SUBJECTS  = 22
NUM_SAMPLES   = 40
START_SUBJECT = 1

SEGMENT_LEN   = 1  # segment 길이
BATCH_SIZE    = 16
EPOCHS        = 100
TEST_SIZE     = 0.2

os.makedirs(RESULT_DIR, exist_ok=True)

@register_keras_serializable(package='custom_layers')
class SplitModality(tf.keras.layers.Layer):
    def __init__(self, idx, **kwargs):
        super().__init__(**kwargs)
        self.idx = idx

    def call(self, inputs):
        return tf.expand_dims(inputs[:, self.idx], -1)

    def get_config(self):
        config = super().get_config()
        config.update({'idx': self.idx})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
# ───────────────────────────────────────────────────────────────
# 데이터 로딩 함수
# ───────────────────────────────────────────────────────────────
def prepare_segments(feature_dir, label_root, num_subj, num_samp, segment_len):
    X_list, y_list, meta = [], [], []
    for subj in range(1, num_subj+1):
        subj_id    = f"s{subj:02d}"
        label_path = os.path.join(label_root, f"subject{subj:02d}.npy")
        labels     = np.load(label_path)
        for trial in range(1, num_samp+1):
            feat_file = os.path.join(feature_dir, subj_id, f"{subj_id}_trial{trial:02d}.npy")
            if not os.path.isfile(feat_file):
                continue
            feats = np.load(feat_file)  # (feat_dim, n_seg)
            label = int(labels[trial-1])
            n_seg = feats.shape[1]
            for start in range(0, n_seg, segment_len):
                end = start + segment_len
                if end > n_seg: break
                window = feats[:, start:end]
                X_list.append(window.T)  # (segment_len, feat_dim)
                y_list.append(label)
                meta.append((subj_id, trial, start//segment_len))
    X = np.stack(X_list, axis=0)
    y = np.array(y_list, dtype=np.int32)
    return X, y, meta

# ───────────────────────────────────────────────────────────────
# main
# ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    # 1) 데이터 로드
    X_eye,  y_eye,  meta_eye  = prepare_segments(EYE_FEAT_DIR,  LABEL_ROOT, NUM_SUBJECTS, NUM_SAMPLES, SEGMENT_LEN)
    X_face, y_face, meta_face = prepare_segments(FACE_FEAT_DIR, LABEL_ROOT, NUM_SUBJECTS, NUM_SAMPLES, SEGMENT_LEN)
    print(f"Eye: {X_eye.shape}, Face: {X_face.shape}")

    # 2) 메타 일치 확인
    assert meta_eye == meta_face, "Eye/Face meta mismatch"

    # 3) 라벨 필터링 (라벨 4 제외)
    mask = (y_eye != 4)
    X_eye  = X_eye[mask]; X_face = X_face[mask]; y = y_eye[mask]

    # 4) 라벨 맵핑
    unique_labels = np.unique(y)
    label_map     = {lab: idx for idx, lab in enumerate(unique_labels)}
    y = np.array([label_map[v] for v in y], dtype=np.int32)
    NUM_CLASSES = len(unique_labels)
    print(f"Classes: {label_map}")

    # 5) NaN/Inf 제거 및 정규화 함수
    def clean_norm(X):
        good = ~np.isnan(X).any(axis=(1,2)) & ~np.isinf(X).any(axis=(1,2))
        X = X[good]
        N, T, D = X.shape
        XF = X.reshape(N*T, D)
        m, s = XF.mean(0), XF.std(0) + 1e-6
        X = ((XF - m)/s).reshape(N, T, D)
        return X, good

    X_eye,  mask_eye  = clean_norm(X_eye)
    X_face, mask_face = clean_norm(X_face)
    common = mask_eye & mask_face
    X_eye  = X_eye[common]; X_face = X_face[common]; y = y[common]
    print(f"After clean: Eye {X_eye.shape}, Face {X_face.shape}, y {y.shape}")

    # 6) train/test split
    X_eye_tr, X_eye_te, X_face_tr, X_face_te, y_tr, y_te = train_test_split(
        X_eye, X_face, y, test_size=TEST_SIZE, random_state=42, stratify=y)

        # 7) 사전 학습 모델 아키텍처 재생성 및 가중치 로드
    from Eye_model import build_eye_model  # 기존 코드에서 사용하던 함수
    from Face_model import build_face_model

    # Eye 모델 재생성 및 가중치 로드
    eye_base = build_eye_model(
        num_classes=NUM_CLASSES,
        seq_length=X_eye_tr.shape[1],
        feature_dim=X_eye_tr.shape[2]
    )
    eye_base.load_weights(PRETRAIN_EYE_MODEL)
    # Face 모델 재생성 및 가중치 로드 (weights-only .h5)
    face_base = build_face_model(
        num_classes=NUM_CLASSES,
        seq_length=X_face_tr.shape[1],
        feature_dim=X_face_tr.shape[2]
    )
    face_base.load_weights(PRETRAIN_FACE_MODEL)

    # Feature extractor로 분기
    eye_extractor = Model(inputs=eye_base.input, outputs=eye_base.layers[-2].output)
    face_extractor = Model(inputs=face_base.input, outputs=face_base.layers[-2].output)
    # 두 extractor는 고정
    eye_extractor.trainable = False
    face_extractor.trainable = False

    # 9) 게이팅 기반 멀티모달 모델 정의
    in_eye  = Input(shape=X_eye_tr.shape[1:],  name='eye_input')
    in_face = Input(shape=X_face_tr.shape[1:], name='face_input')

    feat_eye  = eye_extractor(in_eye)    # (batch, D1)
    feat_face = face_extractor(in_face)  # (batch, D2)

    # 1) 두 출력을 concat
    concat_feats = Concatenate(name='concat_feats')([feat_eye, feat_face])

    # 2) modality gate: [w_eye, w_face], softmax
    #    bias_initializer을 통해 초기에 face gate를 크게 시작
    gates = Dense(
        2,
        activation='softmax',
        name='modality_gate',
        bias_initializer=tf.keras.initializers.Constant([0.3, 0.7])
    )(concat_feats)  # (batch, 2)

    # 3) gate 별로 분리
    w_eye  = SplitModality(0, name='w_eye')(gates)
    w_face = SplitModality(1, name='w_face')(gates)
    # 4) 각 feature에 gate 곱해주기
    feat_eye_weighted  = Multiply(name='eye_weighted')([feat_eye,  w_eye])   # (batch, D1)
    feat_face_weighted = Multiply(name='face_weighted')([feat_face, w_face]) # (batch, D2)

    # 5) 차원이 맞으면 더해주거나 concat 가능
    #    (D1, D2가 다르면 concat 하셔도 됩니다. 여기선 add 대신 concat 예시)
    fused = Add(name='fused_feature')([feat_eye_weighted, feat_face_weighted])

    # 출력층
    x   = Dense(128, activation='relu', name='fusion_dense')(fused)
    out = Dense(NUM_CLASSES, activation='softmax', name='output')(x)

    multi_model = Model([in_eye, in_face], out, name='gated_fusion_model')
    multi_model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    multi_model.summary()

    # 10) 학습 및 콜백
    cp = ModelCheckpoint(os.path.join(RESULT_DIR,'best_multi_model.keras'), monitor='val_accuracy', save_best_only=True, verbose=1)
    es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
    history = multi_model.fit([X_eye_tr,X_face_tr], y_tr, validation_split=0.2, epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[cp,es], verbose=2)

    # 11) 평가 및 결과 저장
    loss,acc = multi_model.evaluate([X_eye_te,X_face_te], y_te, verbose=2)
    print(f"Test Loss: {loss:.4f}, Acc: {acc:.4f}")
    y_pred = multi_model.predict([X_eye_te,X_face_te]).argmax(axis=1)
    report = classification_report(y_te,y_pred, target_names=[str(l) for l in unique_labels])
    cm = confusion_matrix(y_te,y_pred)
    print(report); print("Confusion matrix:\n", cm)

    with open(os.path.join(RESULT_DIR,'classification_report.txt'),'w') as f: f.write(report)
    np.savetxt(os.path.join(RESULT_DIR,'confusion_matrix.txt'), cm, fmt='%d')
    plt.figure(); plt.plot(history.history['loss'], label='Train Loss'); plt.plot(history.history['val_loss'], label='Val Loss'); plt.legend(); plt.savefig(os.path.join(RESULT_DIR,'loss_curve.png')); plt.close()
    plt.figure(); plt.plot(history.history['accuracy'], label='Train Acc'); plt.plot(history.history['val_accuracy'], label='Val Acc'); plt.legend(); plt.savefig(os.path.join(RESULT_DIR,'accuracy_curve.png')); plt.close()
