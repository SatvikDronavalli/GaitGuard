# GaitGuard
A machine learning approach to fall-risk prediction using gait analysis. Satvik and Shaurya's 2025 and 2026 RSEF project.

## Current Pipeline
- Input: 10 meter walk test, with 10 meter walk forwards, 180 degree turn, and 10 meter walk back
- AI Model: CNN-CBAM fused with SVM classifier for unlabeled IMU time-series data
- Physical data collection: Adafruit ESP32-S3 microprocessor with ism330dhcx IMU, uses BLE with bleak for real-time streaming
- Additional Data analysis: Largest Lyapunov Exponent (LLE) derivations using Rosenstein's Method, split between walking and turning segments of input data
