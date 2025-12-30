import asyncio
from bleak import BleakScanner, BleakClient
import matplotlib.pyplot as plt
import struct
from collections import deque

CHAR_UUID_IMU = "beb5483e-36e1-4688-b7f5-ea07361b26a8"
PACKET_FMT = "<ffffffI"
PACKET_SIZE = struct.calcsize(PACKET_FMT)

MAX_POINTS = 500
ax_buf = deque(maxlen=MAX_POINTS)

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
                ax_buf.append(accel_x)

        await client.start_notify(CHAR_UUID_IMU, handler)

        # ----- Matplotlib setup -----
        plt.ion()
        fig, ax_plot = plt.subplots()
        line, = ax_plot.plot([], [])
        ax_plot.set_xlabel("Sample")
        ax_plot.set_ylabel("ax")
        ax_plot.set_title("Live ax from GaitGuard")
        plt.show(block=False)

        duration = 60.0
        dt = 0.05
        steps = int(duration / dt)

        for _ in range(steps):
            if ax_buf:
                x_data = range(len(ax_buf))
                y_data = list(ax_buf)

                line.set_data(x_data, y_data)
                ax_plot.relim()
                ax_plot.autoscale_view()

                fig.canvas.draw()
                plt.pause(0.001)

            await asyncio.sleep(dt)

        print("Stopping notify")
        await client.stop_notify(CHAR_UUID_IMU)

asyncio.run(main())
