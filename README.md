# GaitGuard
A machine learning approach to fall-risk prediction using gait analysis

## Current Pipeline
- Input: 10 meter walk test, with 10 meter walk forwards, 180 degree turn, and 10 meter walk back\n
- AI Model: CNN-CBAM fused with SVM classifier for unlabeled IMU time-series data\n
- Physical data collection: Adafruit ESP32-S3 microprocessor with ism330dhcx IMU, uses BLE with bleak for real-time streaming\n
- Additional Data analysis: Largest Lyapunov Exponent (LLE) derivations using Rosenstein's Method, split between walking and turning segments of input data\n
