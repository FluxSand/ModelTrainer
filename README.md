# IMU Motion Analysis Toolkit

Tools for IMU sensor data processing and visualization

## Usage

```bash
git clone --depth=1 https://github.com/FluxSand/Dataset.git Dataset
python train.py
# cp model_cnn_lstm.onnx XXX
```

## 📦 Files

### 1. CNN-LSTM Motion Classifier (`model_cnn_lstm.py`)

**Features**:

- Hybrid CNN-BiLSTM architecture for temporal pattern recognition
- Advanced data preprocessing pipeline
  - Sequence padding & normalizationiles
  - Stratified dataset splitting
- ONNX model export capability
- Embedded early stopping & learning rate scheduling

### 2. IMU Data Visualizer (`imu_visualizer.py`)

**Features**:

- Interactive CSV file management
  - File browsing & deletion
  - Auto-refresh directory
- Multi-view sensor visualization
  - 3D pose animation with adjustable parameters
  - Synchronized 2D plots (Pitch/Roll/Gyro/Accel)
- Customizable playback controls
  - Frame skip ratio (1-50)
  - Speed adjustment (10-200ms)

**Requirements**:

```text
tensorflow==2.10.0
pandas==1.5.0
matplotlib==3.6.0
scipy==1.9.0
ttkbootstrap==1.10.1
tf2onnx==1.13.0
```

## 📂 Dataset Structure

Each CSV should contain:

```csv
Pitch,Roll,Gyro_X,Gyro_Y,Gyro_Z,Accel_X,Accel_Y,Accel_Z
```

## 📸 Sample Output

**Classifier Training**:

```text
Epoch 150/2000
187/187 [==============================] - 15s 78ms/step - loss: 0.0327 - accuracy: 0.7215
Val accuracy: 0.6982 → Best model saved
```

**Visualizer Interface**:
![IMU Visualizer Screenshot](./imgs/Visualizer1.png)
![IMU Visualizer Screenshot](./imgs/Visualizer2.png)
