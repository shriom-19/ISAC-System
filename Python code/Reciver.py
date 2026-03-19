import serial
import re
import math
import subprocess
import threading

# 🔥 Change COM port if needed
PORT = "COM5"
BAUD = 115200

ser = serial.Serial(PORT, BAUD, timeout=1)


# ===================== 🔥 PING FUNCTION =====================
def start_ping():
    # Windows continuous ping (-t)
    subprocess.Popen(
        ["ping", "10.82.53.253", "-t"],   # 🔥 use gateway IP (IMPORTANT)
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )


# ===================== 🔥 CSI PARSER =====================
def parse_csi(line):
    try:
        if "DATA:[" not in line:
            return None

        # Timestamp
        ts_match = re.search(r'TS:(\d+:\d+:\d+)', line)
        timestamp = ts_match.group(1) if ts_match else None

        # RSSI
        rssi_match = re.search(r'RSSI:(-?\d+)', line)
        rssi = int(rssi_match.group(1)) if rssi_match else None

        # DATA
        data_match = re.search(r'DATA:\[(.*)\]', line)
        if not data_match:
            return None

        raw_data = list(map(int, data_match.group(1).split(',')))

        # Convert to amplitude (128 values)
        csi_array = []
        for i in range(0, len(raw_data), 2):
            real = raw_data[i]
            imag = raw_data[i + 1]

            amp = int(math.sqrt(real**2 + imag**2))
            csi_array.append(amp)

        return timestamp, rssi, csi_array

    except:
        return None


# ===================== 🚀 START PING THREAD =====================
ping_thread = threading.Thread(target=start_ping, daemon=True)
ping_thread.start()


# ===================== 🚀 MAIN LOOP =====================
while True:
    try:
        line = ser.readline().decode(errors="ignore").strip()

        result = parse_csi(line)

        if result:
            ts, rssi, csi = result

            print("TIME:", ts)
            print("RSSI:", rssi)
            print("CSI (128):", csi)
            print("-" * 60)

    except KeyboardInterrupt:
        print("Stopped")
        break