# SPDX-FileCopyrightText: 2020 ladyada for Adafruit Industries
# SPDX-License-Identifier: MIT

"""
This example scans for any BLE advertisements and prints one advertisement and one scan response
from every device found.
"""

# Docs: https://docs.circuitpython.org/projects/esp32spi/en/latest/index.html

from adafruit_ble import BLERadio

ble = BLERadio()
print("scanning")
found = set()
scan_responses = set()
connection = None
for advertisement in ble.start_scan():
    if advertisement.complete_name == "GaitGuard":
        addr = advertisement.address
        connection = ble.connect(advertisement)
        print("Connected!")
        break
    '''
    addr = advertisement.address
    if advertisement.scan_response and addr not in scan_responses:
        scan_responses.add(addr)
    elif not advertisement.scan_response and addr not in found:
        found.add(addr)
    else:
        continue
    print(addr, advertisement)
    print("\t" + repr(advertisement))
    print() '''
#print(dir(connection))



connection.disconnect()
print("scan done")