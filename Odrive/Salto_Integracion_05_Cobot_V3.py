import odrive
import time
import math
import matplotlib.pyplot as plt
import threading
import keyboard
import pyrealsense2 as rs
import numpy as np
import cv2
import pandas as pd

# ==== CONFIGURACIÓN ====
DT = 0.016
DURACION_TOTAL = 3.0
MUESTRAS = int(DURACION_TOTAL / DT)

REF1_INICIAL = 0.2404
REF2_INICIAL = 0.1161
REF1_FINAL = 0.0
REF2_FINAL = 0.25
UMBRAL_POS = 0.01

# ==== CONEXIÓN A ODrives ====
print("🔍 Buscando ODrives...")
odrv1 = odrive.find_any(serial_number="384B34733539")
print("✅ ODrive 1 conectado")
time.sleep(1.0)
odrv2 = odrive.find_any(serial_number="384434593539")
print("✅ ODrive 2 conectado")

for odrv in [odrv1, odrv2]:
    odrv.axis0.controller.config.control_mode = 3
    odrv.axis0.controller.config.input_mode = 1
    odrv.axis0.requested_state = 8
    time.sleep(0.1)

odrv1.axis0.controller.input_pos = REF1_INICIAL
odrv2.axis0.controller.input_pos = REF2_INICIAL
print("⚙️ Motores en posición inicial...")
time.sleep(1.5)

print("⏳ Presiona 'E' para iniciar trayectoria...")
while not keyboard.is_pressed('e'):
    time.sleep(0.01)
print("🎯 Ejecutando trayectoria por 3 segundos...")

# ==== VARIABLES COMPARTIDAS ====
current_refs = {'ref1': REF1_INICIAL, 'ref2': REF2_INICIAL, 'running': True}
ref_lock = threading.Lock()
shared_data = {'z': float('nan')}
data_lock = threading.Lock()
sample_request = threading.Event()
sample_ready = threading.Event()

# ==== BUFFERS ====
enc1_vals, enc2_vals = [], []
u1_vals, u2_vals = [], []
iq1_vals, iq2_vals = [], []
error1_vals, error2_vals = [], []
z_vals = []
iteration_times = []

# ==== HILO DE LECTURA ====
def read_thread_func():
    while current_refs['running']:
        sample_request.wait()
        if not current_refs['running']:
            break

        with ref_lock:
            ref1 = current_refs['ref1']
            ref2 = current_refs['ref2']

        enc1 = odrv1.axis0.pos_estimate * 2 * math.pi
        enc2 = odrv2.axis0.pos_estimate * 2 * math.pi
        ref1_rad = ref1 * 2 * math.pi
        ref2_rad = ref2 * 2 * math.pi
        u1 = odrv1.axis0.controller.config.pos_gain * (ref1 - odrv1.axis0.pos_estimate)
        u2 = odrv2.axis0.controller.config.pos_gain * (ref2 - odrv2.axis0.pos_estimate)
        iq1 = odrv1.axis0.motor.foc.Iq_measured
        iq2 = odrv2.axis0.motor.foc.Iq_measured

        with data_lock:
            z = shared_data['z']

        enc1_vals.append(enc1)
        enc2_vals.append(enc2)
        u1_vals.append(u1)
        u2_vals.append(u2)
        error1_vals.append(ref1_rad - enc1)
        error2_vals.append(ref2_rad - enc2)
        iq1_vals.append(iq1)
        iq2_vals.append(iq2)
        z_vals.append(1.42 - z)

        sample_ready.set()
        sample_request.clear()

# ==== HILO DE CÁMARA ====
def camera_thread_func():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 60)
    config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 60)
    align = rs.align(rs.stream.color)
    pipeline.start(config)

    kernel = np.ones((5, 5), np.uint8)
    x_min, y_min = 400, 200
    x_max, y_max = 550, 350

    while current_refs['running']:
        frameset = pipeline.wait_for_frames()
        aligned = align.process(frameset)
        depth_frame = aligned.get_depth_frame()
        color_frame = aligned.get_color_frame()

        if not depth_frame or not color_frame:
            with data_lock:
                shared_data['z'] = float('nan')
            continue

        color_image = np.asanyarray(color_frame.get_data())
        roi_color = color_image[y_min:y_max, x_min:x_max]

        hsv = cv2.cvtColor(roi_color, cv2.COLOR_BGR2HSV)
        lower1 = np.array([0, 70, 50])
        upper1 = np.array([10, 255, 255])
        lower2 = np.array([170, 70, 50])
        upper2 = np.array([180, 255, 255])
        mask1 = cv2.inRange(hsv, lower1, upper1)
        mask2 = cv2.inRange(hsv, lower2, upper2)
        mask = cv2.bitwise_or(mask1, mask2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        result = cv2.bitwise_and(roi_color, roi_color, mask=mask)
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)

        circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=30,
                                   param1=100, param2=15, minRadius=7, maxRadius=50)
        z_val = float('nan')
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for c in circles[0, :]:
                x, y, r = c
                mask_roi = mask[max(y - r, 0):min(y + r, mask.shape[0]), max(x - r, 0):min(x + r, mask.shape[1])]
                if mask_roi.size == 0:
                    continue
                if cv2.mean(mask_roi)[0] < 50:
                    continue
                x_full = x + x_min
                y_full = y + y_min
                z_val = depth_frame.get_distance(x_full, y_full)
                break

        with data_lock:
            shared_data['z'] = z_val

    pipeline.stop()

# ==== INICIAR HILOS ====
read_thread = threading.Thread(target=read_thread_func)
cam_thread = threading.Thread(target=camera_thread_func)
read_thread.start()
cam_thread.start()
time.sleep(2)

# ==== EJECUTAR TRAYECTORIA ====
next_time = time.monotonic()

for i in range(MUESTRAS):
    t_start = time.monotonic()
    t = i * DT

    if i == int(MUESTRAS // 2):
        with ref_lock:
            current_refs['ref1'] = REF1_FINAL
            current_refs['ref2'] = REF2_FINAL
        odrv1.axis0.controller.input_pos = REF1_FINAL
        odrv2.axis0.controller.input_pos = REF2_FINAL

    sample_request.set()
    sample_ready.wait()
    sample_ready.clear()

    next_time += DT
    remaining = next_time - time.monotonic()
    if remaining > 0:
        if remaining > 0.002:
            time.sleep(remaining - 0.001)
        while time.monotonic() < next_time:
            pass

    iteration_times.append(time.monotonic() - t_start)

# ==== FINALIZAR ====
current_refs['running'] = False
sample_request.set()
read_thread.join()
cam_thread.join()
for odrv in [odrv1, odrv2]:
    odrv.axis0.requested_state = 1
print("✅ Motores en estado IDLE.")

# ==== GRAFICAR ====
time_vector = np.arange(MUESTRAS)
evento_cambio = [0 if i < MUESTRAS // 2 else 1 for i in range(MUESTRAS)]

plt.figure(figsize=(14, 18))

plt.subplot(6, 1, 1)
plt.plot(time_vector, enc1_vals, label='Encoder Motor 1')
plt.plot(time_vector, enc2_vals, label='Encoder Motor 2')
plt.axhline(REF1_FINAL * 2 * math.pi, color='red', linestyle='--', label='Ref Final M1')
plt.axhline(REF2_FINAL * 2 * math.pi, color='blue', linestyle='--', label='Ref Final M2')
plt.axvline(MUESTRAS // 2, color='gray', linestyle=':', label='Cambio Ref')
plt.title('Encoders vs Referencias')
plt.xlabel('Muestra')
plt.ylabel('Rad')
plt.legend()

plt.subplot(6, 1, 2)
plt.plot(u1_vals, label='u1')
plt.plot(u2_vals, label='u2')
plt.title("Salida del PID")
plt.ylabel("Control (u)")
plt.legend()

plt.subplot(6, 1, 3)
plt.plot(error1_vals, label='Error M1')
plt.plot(error2_vals, label='Error M2')
plt.title('Error de Seguimiento')
plt.ylabel('Error (rad)')
plt.legend()

plt.subplot(6, 1, 4)
plt.plot(iq1_vals, label='Iq M1')
plt.plot(iq2_vals, label='Iq M2')
plt.title('Corriente Iq')
plt.ylabel('Corriente (A)')
plt.legend()

plt.subplot(6, 1, 5)
plt.plot(z_vals, label='Z RealSense')
plt.title('Desplazamiento Z')
plt.ylabel('m')
plt.legend()

plt.subplot(6, 1, 6)
plt.plot(iteration_times, label='Δt iteración')
plt.title('Duración por iteración')
plt.ylabel('s')
plt.legend()

plt.tight_layout()
plt.show()

# ==== CSV ====
def pad_or_truncate(lst, length):
    return (lst + [float('nan')] * (length - len(lst)))[:length]

max_len = max(
    len(enc1_vals), len(enc2_vals), len(z_vals),
    len(iteration_times), len(evento_cambio), len(time_vector)
)

data = {
    'tiempo_s': pad_or_truncate(time_vector.tolist(), max_len),
    'enc1_rad': pad_or_truncate(enc1_vals, max_len),
    'enc2_rad': pad_or_truncate(enc2_vals, max_len),
    'profundidad_m': pad_or_truncate(z_vals, max_len),
    'duracion_muestra_s': pad_or_truncate(iteration_times, max_len),
    'evento_cambio': pad_or_truncate(evento_cambio, max_len)
}

df = pd.DataFrame(data)
df.to_csv("datos_trayectoria.csv", index=False)
print("💾 Datos guardados en 'datos_trayectoria.csv'")
