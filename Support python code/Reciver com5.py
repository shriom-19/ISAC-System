import serial
import re
import threading
import time

PORTS = ["COM5", "COM6"]
BAUD = 115200

print_lock = threading.Lock()


# ===================== CSI PARSER =====================
def parse_csi(line):
    try:
        if "DATA:[" not in line:
            return None

        line = line.replace(" ", "")

        ts = re.search(r'TS:(\d+:\d+:\d+|\d+)', line)
        rssi = re.search(r'RSSI:(-?\d+)', line)
        data = re.search(r'DATA:\[([-\d,]+)\]', line)

        if not data:
            return None

        raw = list(map(int, data.group(1).split(',')))

        csi = []
        for i in range(0, len(raw) - 1, 2):
            real = raw[i]
            imag = raw[i + 1]
            amp = abs(real) + abs(imag)
            csi.append(amp)

        return ts.group(1), int(rssi.group(1)), csi

    except:
        return None


# ===================== SERIAL READER =====================
def read_serial(port):
    try:
        ser = serial.Serial(port, BAUD, timeout=1)
        with print_lock:
            print(f"✅ Connected to {port}")
    except:
        with print_lock:
            print(f"❌ Failed to connect {port}")
        return

    while True:
        try:
            line = ser.readline().decode(errors="ignore").strip()

            if not line:
                continue

            result = parse_csi(line)

            with print_lock:
                if result:
                    ts, rssi, csi = result

                    print(f"\n📡 [{port}]")
                    print(f"TIME : {ts}")
                    print(f"RSSI : {rssi}")
                    print(f"CSI  : {csi}")   # 🔥 FULL CSI
                    print("-" * 60)

                else:
                    print(f"[{port}] RAW: {line}")

        except Exception as e:
            with print_lock:
                print(f"⚠️ Error on {port}: {e}")
            time.sleep(1)


# ===================== START THREADS =====================
for port in PORTS:
    threading.Thread(target=read_serial, args=(port,), daemon=True).start()


# ===================== KEEP ALIVE =====================
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("Stopped")