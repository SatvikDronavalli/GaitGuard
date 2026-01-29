import asyncio
from bleak import BleakScanner, BleakClient
import matplotlib.pyplot as plt
import struct
from collections import deque
import keyboard
import numpy as np
import joblib
import tensorflow as tf
from tensorflow import keras
from IMU_Pods.TUG_data_visualization import butter_lowpass
from IMU_Pods.AI_Pipeline import ChannelAttention1D


CHAR_UUID_IMU = "beb5483e-36e1-4688-b7f5-ea07361b26a8"
PACKET_FMT = "<ffffffI"
PACKET_SIZE = struct.calcsize(PACKET_FMT)

MAX_POINTS = 500
gx_buf = deque(maxlen=MAX_POINTS)
ax_arr = []
ay_arr = []
az_arr = []
gx_arr = []
gy_arr = []
gz_arr = []
total_time = 0
cnt = 0
init_time = 0

def normalize(val):
    new_val = val.copy()
    print(new_val.shape )
    if new_val.ndim > 1:
        for i in range(new_val.ndim):
            mean = np.mean(new_val[:, i])
            std = np.std(new_val[:, i])
            new_val[:, i] = (new_val[:, i] - mean) / (std + 1e-8)
    else:
        mean = np.mean(new_val[:])
        std = np.std(new_val[:])
        new_val[:] = (new_val[:] - mean) / (std + 1e-8)

    return new_val

def resample(norm_val, desired_len):
    if norm_val.ndim > 1:
        n, c = norm_val.shape
        t_old = np.linspace(0.0, 1.0, n, dtype=np.float64)
        t_new = np.linspace(0.0, 1.0, desired_len, dtype=np.float64)
        x = norm_val.astype(np.float64, copy=False)
        y = np.empty((desired_len, c), dtype=np.float64)
        for ch in range(c):
            y[:, ch] = np.interp(t_new, t_old, x[:, ch])
    else:
        n = len(norm_val)
        t_old = np.linspace(0.0, 1.0, n, dtype=np.float64)
        t_new = np.linspace(0.0, 1.0, desired_len, dtype=np.float64)
        x = norm_val.astype(np.float64, copy=False)
        y = np.empty((desired_len), dtype=np.float64)
        y[:] = np.interp(t_new, t_old, x[:])
    return y


async def main():
    devices = await BleakScanner.discover()
    target = None

    for d in devices:
        if d.name == "GaitGuard":
            target = d
            print("GaitGuard found")
            break

    if not target:
        print("GaitGuard not found")
        return

    async with BleakClient(target.address) as client:
        print("Connected")

        # Intended frequency is 200 Hz, received frequency is 192-196 Hz due to overhead
        def handler(sender, data):
            if len(data) == PACKET_SIZE:
                global cnt, total_time, init_time
                ax, ay, az, gx, gy, gz, t = struct.unpack(PACKET_FMT, data)
                ax_arr.append(ax)
                ay_arr.append(ay)
                az_arr.append(az)
                gx_arr.append(gx)
                gy_arr.append(gy)
                gz_arr.append(gz)
                cnt += 1
                if init_time == 0:
                    init_time = t
                total_time = t


        await client.start_notify(CHAR_UUID_IMU, handler)

        while not keyboard.is_pressed('space'):
            await asyncio.sleep(0.1)

        '''
        # ----- Matplotlib setup -----
        plt.ion()
        fig, gx_plot = plt.subplots()
        line, = gx_plot.plot([], [])
        gx_plot.set_xlabel("Sample")
        gx_plot.set_ylabel("gx")
        gx_plot.set_title("Live gx from GaitGuard")
        plt.show(block=False)

        duration = 60.0
        dt = 0.05
        steps = int(duration / dt) 

        for _ in range(steps):
            if gx_buf:
                x_data = range(len(gx_buf))
                y_data = list(gx_buf) 

                line.set_data(x_data, y_data) 
                gx_plot.relim()
                gx_plot.autoscale_view()
 
                fig.canvas.draw()
                plt.pause(0.001)

            await asyncio.sleep(dt) '''

        print("Stopping notify")
        await client.stop_notify(CHAR_UUID_IMU)




asyncio.run(main())
print(f"{round(cnt / ((total_time - init_time)/1000))} Hz")

# Post processing

FREQUENCY = 200

gx_arr = normalize(butter_lowpass(gx_arr, FREQUENCY, upper=12, order=8))
gy_arr = normalize(butter_lowpass(gy_arr, FREQUENCY, upper=12, order=8))
gz_arr = normalize(butter_lowpass(gz_arr, FREQUENCY, upper=12, order=8))


walk1_time = round(float(input("Enter the duration of the first walking segment (s): ")), 2)
turn_time = round(float(input("Enter the duration of the turning segment (s): ")), 2)
walk2_time = round(float(input("Enter the duration of the second walking segment (s): ")), 2)

checkpoint1 = round(FREQUENCY*walk1_time)
checkpoint2 = round(checkpoint1 + FREQUENCY*turn_time)
checkpoint3 = round(checkpoint2 + FREQUENCY*walk2_time)

gx_walk1 = resample(gx_arr[:checkpoint1], 2000)
gx_turn = resample(gx_arr[checkpoint1: checkpoint2], 2000)
gx_walk2 = resample(gx_arr[checkpoint2: checkpoint3], 2000)
gy_walk1 = resample(gy_arr[:checkpoint1], 2000)
gy_turn = resample(gy_arr[checkpoint1: checkpoint2], 2000)
gy_walk2 = resample(gy_arr[checkpoint2: checkpoint3], 2000)
gz_walk1 = resample(gz_arr[:checkpoint1], 2000)
gz_turn = resample(gz_arr[checkpoint1: checkpoint2], 2000)
gz_walk2 = resample(gz_arr[checkpoint2: checkpoint3], 2000)

model_input = np.stack((gx_walk1,gx_turn,gx_walk2,gy_walk1,gy_turn,gy_walk2,gz_walk1,gz_turn,gz_walk2), axis=1)
model_input = np.expand_dims(model_input, axis=0)

conv = keras.models.load_model('cnn_model.keras')

feature_extractor = keras.Model(
        inputs=conv.input,
        outputs=conv.layers[-2].output
    )

features = feature_extractor.predict(model_input, verbose=0)
svm_predictor = joblib.load('svm_classifier.joblib')
predictions = svm_predictor.predict(features)
if predictions[0]:
    print("Be careful! You are currently at risk of falls!")
else:
    print("Don't worry, your walking is healthy!")