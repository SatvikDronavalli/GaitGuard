import asyncio
from bleak import BleakScanner, BleakClient
import matplotlib.pyplot as plt
import struct
from collections import deque

CHAR_UUID_IMU = "beb5483e-36e1-4688-b7f5-ea07361b26a8"
PACKET_FMT = "<ffffffI"
PACKET_SIZE = struct.calcsize(PACKET_FMT)

MAX_POINTS = 500
gx_buf = deque(maxlen=MAX_POINTS)

async def main():
    devices = await BleakScanner.discover()
    target = None

    for d in devices:
        if d.name == "GaitGuard":
            target = d
            break

    if not target:
        print("GaitGuard not found")
        return

    async with BleakClient(target.address) as client:
        print("Connected")

        def handler(sender, data):
            if len(data) == PACKET_SIZE:
                accel_x, ay, az, gx, gy, gz, t = struct.unpack(PACKET_FMT, data)
                gx_buf.append(gx)

        await client.start_notify(CHAR_UUID_IMU, handler)

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

            await asyncio.sleep(dt)

        print("Stopping notify")
        await client.stop_notify(CHAR_UUID_IMU)

asyncio.run(main())
