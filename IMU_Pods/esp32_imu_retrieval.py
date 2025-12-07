# SPDX-FileCopyrightText: 2020 ladyada for Adafruit Industries
# SPDX-License-Identifier: MIT

"""
This example scans for any BLE advertisements and prints one advertisement and one scan response
from every device found.
"""

# BLERadio: https://github.com/adafruit/Adafruit_CircuitPython_BLE/blob/main/adafruit_ble/__init__.py

import asyncio
from bleak import BleakScanner, BleakClient
import struct


CHAR_UUID_IMU = "beb5483e-36e1-4688-b7f5-ea07361b26a8"
CHAR_UUID_LIPO = None
PACKET_FMT = "<ffffffI" # 6 floats for 6DoF IMU + uint32 for timestamp
PACKET_SIZE = struct.calcsize(PACKET_FMT)

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
                ax,ay,az,gx,gy,gz,t = struct.unpack(PACKET_FMT, data)

                # Rotation unit: rad/s, Acceleration unit: gy
                print(f"A=({ax:.2f},{ay:.2f},{az:.2f})  G=({gx:.2f},{gy:.2f},{gz:.2f})  t={t}")

        await client.start_notify(CHAR_UUID_IMU, handler)
        await asyncio.sleep(60)

asyncio.run(main())