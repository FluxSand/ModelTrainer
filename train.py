import os
import numpy as np
import pandas as pd
import tensorflow as tf
import tf2onnx
import onnx
from tensorflow.keras import layers, models, callbacks, utils, regularizers
from tensorflow.keras.preprocessing.sequence import pad_sequences

# === 1. Configuration Parameters ===
SAVE_PATH = './Dataset/'  # Data directory
MODEL_NAME = 'model_cnn_lstm.keras'
MODEL_ONNX_NAME = 'model_cnn_lstm.onnx'
NUM_ROWS = 1500  # Maximum time steps per sample
BATCH_SIZE = 80
EPOCHS = 2000
LEARNING_RATE = 0.0005  # Learning rate

# Define motion categories
MOTION_NAMES = [
    'FLIP_OVER', 'LONG_VIBRATION', 'ROTATE_CLOCKWISE', 'ROTATE_COUNTERCLOCKWISE',
    'SHAKE_BACKWARD', 'SHAKE_FORWARD', 'SHORT_VIBRATION', 'TILT_LEFT', 'TILT_RIGHT', 'STILL'
]
MOTION_LABELS = {name: idx for idx, name in enumerate(MOTION_NAMES)}


# === 2. Data Loading Function ===
def load_dataset(directory, max_rows=None):
    data_file_list, file_labels = [], []

    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            motion_name = filename.split('_record')[0]
            if motion_name in MOTION_LABELS:
                file_path = os.path.join(directory, filename)
                df = pd.read_csv(file_path)

                # Select IMU sensor data
                expected_columns = ['Pitch', 'Roll', 'Gyro_X', 'Gyro_Y', 'Gyro_Z', 'Accel_X', 'Accel_Y', 'Accel_Z']
                if set(expected_columns).issubset(df.columns):
                    data = df[expected_columns].to_numpy()
                    if max_rows:
                        data = data[:max_rows]  # Limit time steps
                    data_file_list.append(data)
                    file_labels.append(MOTION_LABELS[motion_name])

    return data_file_list, file_labels


# === 3. Data Preprocessing ===
def preprocess_data(data_file_list, file_labels, max_len):
    file_list_padded = pad_sequences(data_file_list, maxlen=max_len, dtype='float32', padding='post', value=0)
    labels_one_hot = utils.to_categorical(file_labels, num_classes=len(MOTION_NAMES))
    return file_list_padded, labels_one_hot


# === 4. CNN + LSTM Model ===
def build_model(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)

    # CNN for feature extraction (larger receptive field)
    x = layers.Conv1D(64, kernel_size=5, activation='relu', padding='same')(inputs)
    x = layers.Conv1D(128, kernel_size=5, activation='relu', padding='same', dilation_rate=2)(x)
    x = layers.AveragePooling1D(pool_size=2)(x)  # Reduce pooling effect

    # Bi-Directional LSTM for capturing sequential dependencies
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(64))(x)

    # Fully connected layers
    x = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


# === 5. Model Training ===
def train_model(x_train, y_train, x_test, y_test):
    model = build_model((NUM_ROWS, 8), len(MOTION_NAMES))

    early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=250, restore_best_weights=True)
    checkpoint = callbacks.ModelCheckpoint(MODEL_NAME, monitor='val_accuracy', save_best_only=True, mode='max')

    history = model.fit(
        x_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(x_test, y_test),
        shuffle=True,
        callbacks=[early_stopping, checkpoint]
    )
    return model


# === 6. Load Data and Train ===
file_list, labels = load_dataset(SAVE_PATH, max_rows=NUM_ROWS)

if len(file_list) < 5:
    print("⚠️ Dataset too small for training! Please add more data.")
    exit()

x_data, y_data = preprocess_data(file_list, labels, NUM_ROWS)

# Split dataset: 80% training, 20% testing
num_samples = len(x_data)
train_size = int(num_samples * 0.8)
indices = np.arange(num_samples)
np.random.shuffle(indices)

x_train, x_test = x_data[indices[:train_size]], x_data[indices[train_size:]]
y_train, y_test = y_data[indices[:train_size]], y_data[indices[train_size:]]

# Train the model
model = train_model(x_train, y_train, x_test, y_test)
model.summary()

# === 7. Export to ONNX ===
input_signature = [tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype, name='input_data')]
onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature, opset=13)

with open(MODEL_ONNX_NAME, "wb") as f:
    f.write(onnx_model.SerializeToString())

print(f"\n✅ ONNX model saved to {MODEL_ONNX_NAME}")
