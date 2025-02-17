"""
Motion Recognition Model with CNN Architecture
Author: Cong Liu
Date: 2025-02-16
Description:
    This script implements a motion recognition model using a CNN architecture.
    Features include data preprocessing, augmentation, dual-branch model architecture,
    and ONNX model export for cross-platform compatibility.
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
import tf2onnx
import onnx
from scipy import stats
from scipy.interpolate import interp1d
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models, callbacks, utils, regularizers
from tensorflow.keras.preprocessing.sequence import pad_sequences


# === 1. Global Configuration Settings ===
class Config:
    """Global configuration parameters for model training and data processing"""
    SAVE_PATH = './Dataset/'  # Path to training data
    MODEL_NAME = 'model.keras'  # Keras model save filename
    MODEL_ONNX_NAME = 'model.onnx'  # ONNX model save filename
    NUM_ROWS = 1500  # Number of timesteps per sample
    BATCH_SIZE = 300  # Training batch size
    EPOCHS = 600  # Maximum training epochs
    LEARNING_RATE = 0.001  # Initial learning rate
    MOTION_NAMES = [  # Supported motion classes
        'FLIP_OVER', 'LONG_VIBRATION', 'ROTATE_CLOCKWISE', 'ROTATE_COUNTERCLOCKWISE',
        'SHAKE_BACKWARD', 'SHAKE_FORWARD', 'SHORT_VIBRATION', 'TILT_LEFT', 'TILT_RIGHT', 'STILL'
    ]
    MOTION_LABELS = {name: idx for idx, name in enumerate(MOTION_NAMES)}  # Label encoding


# === 2. Data Preprocessing Utilities ===
def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values in DataFrame using forward/backward fill"""
    if df.isnull().sum().sum() > 0:
        df = df.fillna(method='ffill').fillna(method='bfill')
    return df


def remove_outliers_zscore(data: np.ndarray, threshold: float = 6.5) -> np.ndarray:
    """Remove outliers using Z-score method"""
    z_scores = np.abs(stats.zscore(data, axis=0))
    return data[(z_scores < threshold).all(axis=1)]


def add_gaussian_noise(data: np.ndarray, mean: float = 0, std: float = 0.01) -> np.ndarray:
    """Add Gaussian noise to input data for augmentation"""
    noise = np.random.normal(mean, std, data.shape)
    return data + noise


def time_warp(data: np.ndarray, sigma: float = 0.8) -> np.ndarray:
    """
    Apply time warping augmentation to 2D time-series data (timesteps × features)

    Args:
        data: Input array of shape (timesteps, features)
        sigma: Warping intensity parameter
    """
    if data.ndim != 2:
        raise ValueError("Input must be 2D array (timesteps × features)")

    orig_steps = np.arange(data.shape[0])
    random_offsets = np.random.normal(loc=1.0, scale=sigma, size=(data.shape[0],))
    new_steps = np.cumsum(random_offsets)
    new_steps = (new_steps - new_steps.min()) / (new_steps.max() - new_steps.min()) * (data.shape[0] - 1)

    warped_data = np.zeros_like(data)
    for i in range(data.shape[1]):
        interpolator = interp1d(orig_steps, data[:, i], kind='linear', fill_value='extrapolate')
        warped_data[:, i] = interpolator(new_steps)
    return warped_data


def moving_average(data: np.ndarray, window_size: int = 3) -> np.ndarray:
    """
    Apply moving average smoothing to 2D time-series data

    Args:
        data: Input array of shape (timesteps, features)
        window_size: Size of smoothing window
    """
    if data.ndim != 2:
        raise ValueError("Input must be 2D array (timesteps × features)")

    smoothed = np.zeros_like(data)
    for i in range(data.shape[1]):
        smoothed[:, i] = np.convolve(
            data[:, i],
            np.ones(window_size) / window_size,
            mode='same'
        )
    return smoothed


# === 3. Data Loading and Processing Pipeline ===
def load_and_preprocess_data(directory: str, max_rows: int, balance_method: str = 'smote'):
    """
    Load and preprocess motion sensor data from CSV files

    Args:
        directory: Path to data directory
        max_rows: Number of timesteps per sample
        balance_method: Data balancing method ('smote' or 'undersampling')

    Returns:
        Processed data splits (x_train, x_test, y_train, y_test)
    """
    data_list, labels = [], []
    expected_columns = ['Pitch', 'Roll', 'Gyro_X', 'Gyro_Y', 'Gyro_Z', 'Accel_X', 'Accel_Y', 'Accel_Z']

    # Load and process each CSV file
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            motion_name = filename.split('_record')[0]
            if motion_name not in Config.MOTION_LABELS:
                continue

            file_path = os.path.join(directory, filename)
            df = pd.read_csv(file_path)
            df = handle_missing_values(df)
            df = remove_outliers_zscore(df)

            if not set(expected_columns).issubset(df.columns):
                continue

            data = df[expected_columns].values[:max_rows]
            data_list.append(data)
            labels.append(Config.MOTION_LABELS[motion_name])

    # === Original Class Distribution ===
    from collections import Counter
    print("\n=== Original Class Distribution ===")
    label_counter = Counter(labels)
    for motion in Config.MOTION_NAMES:
        count = label_counter.get(Config.MOTION_LABELS[motion], 0)
        print(f"{motion}: {count} samples")

    # Sequence padding and label encoding
    x_padded = pad_sequences(
        data_list,
        maxlen=max_rows,
        dtype='float32',
        padding='post',
        truncating='post',
        value=0.0
    )
    y_encoded = utils.to_categorical(labels, num_classes=len(Config.MOTION_NAMES))

    # Train-test split with stratification
    x_train, x_test, y_train, y_test = train_test_split(
        x_padded, y_encoded,
        test_size=0.2,
        stratify=np.argmax(y_encoded, axis=1),
        random_state=42
    )

    # === Training Set Distribution Before Augmentation ===
    y_train_labels = np.argmax(y_train, axis=1)
    y_test_labels = np.argmax(y_test, axis=1)

    print("\n=== Training Set Distribution Before Augmentation ===")
    train_counter = Counter(y_train_labels)
    for motion in Config.MOTION_NAMES:
        count = train_counter.get(Config.MOTION_LABELS[motion], 0)
        print(f"{motion}: {count} samples")

    print("\n=== Test Set Distribution ===")
    test_counter = Counter(y_test_labels)
    for motion in Config.MOTION_NAMES:
        count = test_counter.get(Config.MOTION_LABELS[motion], 0)
        print(f"{motion}: {count} samples")

    # Apply data augmentation to training set
    x_train = np.array([add_gaussian_noise(sample) for sample in x_train])
    x_train = np.array([time_warp(sample) for sample in x_train])
    x_train = np.array([moving_average(sample) for sample in x_train])

    # Handle class imbalance
    if balance_method in ['smote', 'undersampling']:
        x_train_flat = x_train.reshape(len(x_train), -1)
        sampler = SMOTE(random_state=42) if balance_method == 'smote' else RandomUnderSampler(random_state=42)
        x_train_flat, y_train = sampler.fit_resample(x_train_flat, np.argmax(y_train, axis=1))
        x_train = x_train_flat.reshape(-1, x_train.shape[1], x_train.shape[2])
        y_train = utils.to_categorical(y_train, num_classes=len(Config.MOTION_NAMES))

        # === Training Set After Balancing ===
        print("\n=== Training Set After Balancing ===")
        balanced_counter = Counter(np.argmax(y_train, axis=1))
        for motion in Config.MOTION_NAMES:
            count = balanced_counter.get(Config.MOTION_LABELS[motion], 0)
            print(f"{motion}: {count} samples")

    # Standardize data using training statistics
    mean = np.mean(x_train, axis=(0, 1))
    std = np.std(x_train, axis=(0, 1)) + 1e-8
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std

    return x_train, x_test, y_train, y_test


# === 4. Dual-Branch Model Architecture ===
class AngleFeatureEncoder(layers.Layer):
    """Custom layer for encoding angle features with periodic activation"""

    def call(self, inputs):
        """
        Apply sinusoidal transformation to angle features

        Args:
            inputs: Tensor containing angle features (Pitch and Roll)

        Returns:
            Tensor concatenating sin and cos transformations of inputs
        """
        return tf.concat([tf.sin(inputs), tf.cos(inputs)], axis=-1)


def build_model(input_shape: tuple, num_classes: int) -> tf.keras.Model:
    """
    Build dual-branch CNN model architecture

    Args:
        input_shape: Input data shape (timesteps, features)
        num_classes: Number of output classes

    Returns:
        Compiled Keras model
    """
    inputs = layers.Input(shape=input_shape)
    masked = layers.Masking(mask_value=0.0)(inputs)  # Handle padded sequences

    # Feature separation
    angle_inputs = masked[:, :, :2]  # Pitch and Roll
    sensor_inputs = masked[:, :, 2:]  # Gyro and Accel

    # Enhanced angle processing branch
    angle_x = layers.Conv1D(64, 5, padding='same')(angle_inputs)
    angle_x = AngleFeatureEncoder()(angle_x)  # Now outputs 128 features (sin+cos)
    angle_x = layers.BatchNormalization()(angle_x)
    angle_x = layers.Conv1D(128, 5, padding='same', dilation_rate=2)(angle_x)
    angle_x = AngleFeatureEncoder()(angle_x)  # Apply periodic activation again
    angle_x = layers.GlobalAveragePooling1D()(angle_x)

    # Sensor processing branch
    sensor_x = layers.Conv1D(64, 5, padding='same', activation='relu')(sensor_inputs)
    sensor_x = layers.BatchNormalization()(sensor_x)
    sensor_x = layers.Conv1D(128, 5, padding='same', dilation_rate=2, activation='relu')(sensor_x)
    sensor_x = layers.GlobalAveragePooling1D()(sensor_x)

    # Feature fusion and classification
    merged = layers.concatenate([angle_x, sensor_x])
    x = layers.Dense(128, kernel_regularizer=regularizers.l2(0.01))(merged)
    x = layers.ReLU()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=Config.LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


# === 5. Model Training and Export ===
def train_model(x_train: np.ndarray, y_train: np.ndarray,
                x_val: np.ndarray, y_val: np.ndarray) -> tf.keras.Model:
    """
    Train model with early stopping and checkpointing

    Args:
        x_train: Training data
        y_train: Training labels
        x_val: Validation data
        y_val: Validation labels

    Returns:
        Trained Keras model
    """
    model = build_model((Config.NUM_ROWS, 8), len(Config.MOTION_NAMES))

    callbacks_list = [
        callbacks.EarlyStopping(patience=100, restore_best_weights=True),
        callbacks.ModelCheckpoint(Config.MODEL_NAME, save_best_only=True),
        callbacks.ReduceLROnPlateau(factor=0.2, patience=50)
    ]

    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=Config.EPOCHS,
        batch_size=Config.BATCH_SIZE,
        callbacks=callbacks_list,
        verbose=1
    )
    return model


if __name__ == "__main__":
    # Data preparation
    x_train, x_test, y_train, y_test = load_and_preprocess_data(
        Config.SAVE_PATH,
        max_rows=Config.NUM_ROWS,
        balance_method='smote'
    )

    # Model training
    print("🚀 Starting model training...")
    trained_model = train_model(x_train, y_train, x_test, y_test)
    print("✅ Training completed successfully")

    # Model export
    input_signature = [tf.TensorSpec(trained_model.inputs[0].shape, trained_model.inputs[0].dtype, name='input')]
    onnx_model, _ = tf2onnx.convert.from_keras(trained_model, input_signature, opset=13)
    onnx.save(onnx_model, Config.MODEL_ONNX_NAME)
    print(f"✅ ONNX model saved to {Config.MODEL_ONNX_NAME}")
