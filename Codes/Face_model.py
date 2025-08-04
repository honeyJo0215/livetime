# Face_model.py

import tensorflow as tf
from tensorflow.keras import layers, Model

def build_face_model(num_classes: int,
                     seq_length: int = None,
                     feature_dim: int = 30) -> tf.keras.Model:
    """
    Builds a model for 얼굴 표정 feature 시퀀스를 분류합니다.
      - 시퀀스 길이 >= 2: Conv1D + MaxPooling 블록
      - 시퀀스 길이 == 1: MLP(Dense) 블록

    Args:
        num_classes: 출력 클래스 수 (예: 4)
        seq_length: 입력 시퀀스 길이. None 이면 Conv1D 블록 항상 사용.
        feature_dim: 입력 feature 차원 (여기서는 30)

    Returns:
        Keras Model
    """
    inputs = layers.Input(shape=(seq_length, feature_dim), name="face_input")
    x = layers.LayerNormalization(name="layer_norm")(inputs)

    # 시퀀스 길이가 2 이상일 때만 Conv1D+Pooling 사용
    if seq_length is None or seq_length >= 2:
        # Conv-Pool 블록 1
        x = layers.Conv1D(64, kernel_size=3, padding="same", activation="relu", name="conv1")(x)
        x = layers.MaxPooling1D(pool_size=2, name="pool1")(x)

        # Conv-Pool 블록 2
        x = layers.Conv1D(128, kernel_size=3, padding="same", activation="relu", name="conv2")(x)
        x = layers.MaxPooling1D(pool_size=2, name="pool2")(x)

        # Conv-Pool 블록 3
        x = layers.Conv1D(256, kernel_size=3, padding="same", activation="relu", name="conv3")(x)
        x = layers.MaxPooling1D(pool_size=2, name="pool3")(x)

        # 시퀀스 전체 요약
        x = layers.GlobalAveragePooling1D(name="global_avg_pool")(x)
        x = layers.Dropout(0.5, name="dropout_conv")(x)

    else:
        # 시퀀스 길이가 1일 때: Dense MLP
        x = layers.Flatten(name="flatten")(x)                    # (batch, feature_dim)
        x = layers.Dense(128, activation="relu", name="dense1")(x)
        x = layers.Dropout(0.5, name="dropout1")(x)
        x = layers.Dense(64, activation="relu", name="dense2")(x)
        x = layers.Dropout(0.5, name="dropout2")(x)

    outputs = layers.Dense(num_classes, activation="softmax", name="output")(x)
    return Model(inputs, outputs, name="FaceFeatureModel")


if __name__ == "__main__":
    # 예시: 4가지 감정 분류, 시퀀스 길이 8, feature_dim=30
    model = build_face_model(num_classes=4, seq_length=8, feature_dim=30)
    model.summary()
