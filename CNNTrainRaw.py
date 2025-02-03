import os
import numpy as np
import pandas as pd
import tensorflow as tf
import tf2onnx
import onnx
from tensorflow.keras import models, layers, callbacks, utils, regularizers
from tensorflow.keras.preprocessing.sequence import pad_sequences

# === 1. Configuration Parameters ===
SAVE_PATH = './Dataset/'  # Data directory
MODEL_NAME = 'model.keras'
MODEL_ONNX_NAME = 'model.onnx'
NUM_ROWS = 1500  # Maximum time steps per sample
BATCH_SIZE = 80
EPOCHS = 2000
LEARNING_RATE = 0.0005  # Lower learning rate to prevent loss=0

# Define new motion categories (including STILL)
MOTION_NAMES = [
    'FLIP_OVER', 'LONG_VIBRATION', 'ROTATE_CLOCKWISE', 'ROTATE_COUNTERCLOCKWISE',
    'SHAKE_BACKWARD', 'SHAKE_FORWARD', 'SHORT_VIBRATION', 'TILT_LEFT', 'TILT_RIGHT', 'STILL'
]
MOTION_LABELS = {name: idx for idx, name in enumerate(MOTION_NAMES)}


# === 2. Data Loading ===
def load_dataset(directory, max_rows=None):
    data_file_list, file_labels = [], []

    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            # 🔍 Parse filename to extract the complete motion_name
            motion_name = filename.split('_record')[0]  # Ensure correct category extraction

            # 🛑 Debug: Print motion_name to verify parsing
            print(f"📂 Found file: {filename} | Parsed motion category: {motion_name}")

            if motion_name in MOTION_LABELS:
                file_path = os.path.join(directory, filename)
                df = pd.read_csv(file_path)

                # Select 8 IMU-related channels
                expected_columns = ['Pitch', 'Roll', 'Gyro_X', 'Gyro_Y', 'Gyro_Z', 'Accel_X', 'Accel_Y', 'Accel_Z']
                if set(expected_columns).issubset(df.columns):
                    data = df[expected_columns].to_numpy()

                    if max_rows:
                        data = data[:max_rows]  # Limit maximum number of rows

                    data_file_list.append(data)
                    file_labels.append(MOTION_LABELS[motion_name])
                else:
                    print(f"⚠️ Warning: File {filename} missing required columns, skipped!")
            else:
                print(f"🚨 Unrecognized category: {motion_name}, check if `MOTION_NAMES` matches correctly!")

    # ✅ Check data distribution
    print("\n✅ Data category distribution:")
    print(pd.Series(file_labels).value_counts())

    return data_file_list, file_labels


# === 3. Data Preprocessing ===
def preprocess_data(data_file_list, file_labels, max_len):
    file_list_padded = pad_sequences(data_file_list, maxlen=max_len, dtype='float32', padding='post', value=0)
    labels_one_hot = utils.to_categorical(file_labels, num_classes=len(MOTION_NAMES))
    return file_list_padded, labels_one_hot


# === 4. Improved CNN Model ===
def build_model(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv1D(64, 3, activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.01))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(128, 3, activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Conv1D(256, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(512, 3, activation='relu', padding='same')(x)
    x = layers.GlobalAveragePooling1D()(x)

    x = layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.Dropout(0.6)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


# === 5. Train Model ===
def train_model(x_train, y_train, x_test, y_test):
    model = build_model((NUM_ROWS, 8), len(MOTION_NAMES))

    early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True)
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

# Terminate early if dataset is too small
if len(file_list) < 5:
    print("⚠️ Dataset too small for training! Please add more data.")
    exit()

x_data, y_data = preprocess_data(file_list, labels, NUM_ROWS)

# 80% training set, 20% test set
num_samples = len(x_data)
train_size = int(num_samples * 0.8)

indices = np.arange(num_samples)
np.random.shuffle(indices)

x_train, x_test = x_data[indices[:train_size]], x_data[indices[train_size:]]
y_train, y_test = y_data[indices[:train_size]], y_data[indices[train_size:]]

# Check dataset distribution
train_classes, train_counts = np.unique(y_train.argmax(axis=1), return_counts=True)
val_classes, val_counts = np.unique(y_test.argmax(axis=1), return_counts=True)
print(f"✅ Training set category distribution: {dict(zip(train_classes, train_counts))}")
print(f"✅ Validation set category distribution: {dict(zip(val_classes, val_counts))}")

# Train the model
model = train_model(x_train, y_train, x_test, y_test)
model.summary()

# === 7. Convert and Save ONNX Model ===
input_signature = [tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype, name='input_data')]
onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature, opset=13)

with open(MODEL_ONNX_NAME, "wb") as f:
    f.write(onnx_model.SerializeToString())

print(f"\n✅ ONNX model saved to {MODEL_ONNX_NAME}")
