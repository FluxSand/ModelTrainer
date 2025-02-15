import os
import numpy as np
import pandas as pd
import tensorflow as tf
import tf2onnx
import onnx
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models, callbacks, utils, regularizers
from tensorflow.keras.preprocessing.sequence import pad_sequences


# === 1. Configuration Parameters ===
class Config:
    SAVE_PATH = './Dataset/'  # Data directory
    MODEL_NAME = 'model_cnn_lstm.keras'
    MODEL_ONNX_NAME = 'model_cnn_lstm.onnx'
    NUM_ROWS = 1500  # Maximum time steps per sample
    BATCH_SIZE = 80  # Training batch size
    EPOCHS = 2000  # Maximum training epochs
    LEARNING_RATE = 0.0005  # Learning rate
    MOTION_NAMES = [  # Motion categories
        'FLIP_OVER', 'LONG_VIBRATION', 'ROTATE_CLOCKWISE', 'ROTATE_COUNTERCLOCKWISE',
        'SHAKE_BACKWARD', 'SHAKE_FORWARD', 'SHORT_VIBRATION', 'TILT_LEFT', 'TILT_RIGHT', 'STILL'
    ]
    MOTION_LABELS = {name: idx for idx, name in enumerate(MOTION_NAMES)}  # Label encoding mapping


# === 2. Data Loading ===
def load_dataset(directory, max_rows=None):
    """Load IMU sensor CSV dataset
    Args:
        directory: Path to data directory
        max_rows: Maximum rows to read per file
    Returns:
        data_list: Sensor data list [n_samples, time_steps, features]
        labels: Corresponding label list
    """
    data_list, labels = [], []
    expected_columns = ['Pitch', 'Roll', 'Gyro_X', 'Gyro_Y', 'Gyro_Z', 'Accel_X', 'Accel_Y', 'Accel_Z']

    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            try:
                # Parse motion category
                motion_name = filename.split('_record')[0]
                if motion_name not in Config.MOTION_LABELS:
                    continue

                # Read CSV file
                file_path = os.path.join(directory, filename)
                df = pd.read_csv(file_path)

                # Validate data columns
                if not set(expected_columns).issubset(df.columns):
                    print(f"⚠️ File {filename} missing required columns, skipped")
                    continue

                # Extract and truncate data
                data = df[expected_columns].values[:max_rows]
                data_list.append(data)
                labels.append(Config.MOTION_LABELS[motion_name])

            except Exception as e:
                print(f"⛔ Error loading {filename}: {e}")

    return data_list, labels


# === 3. Data Preprocessing ===
def preprocess_data(data_list, labels, max_len):
    """Data padding and preprocessing
    Args:
        data_list: Raw data list
        labels: Raw label list
        max_len: Padding length
    Returns:
        x_padded: Padded data array [n_samples, max_len, features]
        y_encoded: One-hot encoded labels
    """
    # Sequence padding
    x_padded = pad_sequences(
        data_list,
        maxlen=max_len,
        dtype='float32',
        padding='post',
        truncating='post',
        value=0.0
    )

    # Label encoding
    y_encoded = utils.to_categorical(labels, num_classes=len(Config.MOTION_NAMES))
    return x_padded, y_encoded


# === 4. Data Normalization ===
def normalize_data(x_train, x_test):
    """Normalize data using training set statistics"""
    mean = np.mean(x_train, axis=(0, 1))  # Feature-wise mean
    std = np.std(x_train, axis=(0, 1)) + 1e-8  # Prevent division by zero

    x_train_norm = (x_train - mean) / std
    x_test_norm = (x_test - mean) / std
    return x_train_norm, x_test_norm


# === 5. Model Architecture ===
def build_model(input_shape, num_classes):
    """Build CNN+BiLSTM model architecture"""
    inputs = layers.Input(shape=input_shape)

    # Masking layer for padding values
    x = layers.Masking(mask_value=0.0)(inputs)

    # CNN feature extraction
    x = layers.Conv1D(64, kernel_size=5, activation='relu', padding='same')(x)
    x = layers.Conv1D(128, kernel_size=5, activation='relu', padding='same', dilation_rate=2)(x)
    x = layers.AveragePooling1D(pool_size=2)(x)

    # BiLSTM temporal modeling
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(64))(x)

    # Classification layers
    x = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    # Compile model
    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=Config.LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


# === 6. Model Training ===
def train_model(x_train, y_train, x_val, y_val):
    """Model training and validation"""
    model = build_model((Config.NUM_ROWS, 8), len(Config.MOTION_NAMES))

    # Callbacks
    callbacks_list = [
        callbacks.EarlyStopping(monitor='val_loss', patience=250, restore_best_weights=True),
        callbacks.ModelCheckpoint(
            Config.MODEL_NAME,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max'
        ),
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=100, min_lr=1e-6)
    ]

    # Training process
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=Config.EPOCHS,
        batch_size=Config.BATCH_SIZE,
        callbacks=callbacks_list,
        shuffle=True,
        verbose=1
    )
    return model


# === Main Program ===
if __name__ == "__main__":
    # Data loading and preprocessing
    data_list, labels = load_dataset(Config.SAVE_PATH, max_rows=Config.NUM_ROWS)

    if len(data_list) < 5:
        print("⚠️ Dataset too small! Please add more samples.")
        exit()

    x_padded, y_encoded = preprocess_data(data_list, labels, Config.NUM_ROWS)

    # Stratified data splitting
    x_train, x_test, y_train, y_test = train_test_split(
        x_padded, y_encoded,
        test_size=0.2,
        stratify=np.argmax(y_encoded, axis=1),
        random_state=42
    )

    # Data normalization
    x_train, x_test = normalize_data(x_train, x_test)

    # Model training
    model = train_model(x_train, y_train, x_test, y_test)
    model.summary()

    # Export ONNX model
    input_signature = [tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype, name='input')]
    onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature, opset=13)
    onnx.save(onnx_model, Config.MODEL_ONNX_NAME)
    print(f"\n✅ ONNX model saved to {Config.MODEL_ONNX_NAME}")