import odrive
import time
import pandas as pd
import keyboard
import matplotlib.pyplot as plt
import math
import threading
import pyrealsense2 as rs
import numpy as np
import cv2

contador1 = time.monotonic()

# Leer archivo CSV
df = pd.read_csv('datos1s.csv', header=None, names=['pos1', 'pos2'])
df = df.iloc[::16].reset_index(drop=True)

# Convertir de revoluciones a radianes
df['pos1_rad'] = df['pos1'] * 2 * math.pi
df['pos2_rad'] = df['pos2'] * 2 * math.pi

# Conectar con los ODrives
print("🔍 Buscando ODrives...")
odrv1 = odrive.find_any(serial_number="384B34733539")
print("✅ ODrive 1 conectado")
time.sleep(1.0)
odrv2 = odrive.find_any(serial_number="384434593539")
print("✅ ODrive 2 conectado")

# Activar motores
for odrv in [odrv1, odrv2]:
    odrv.axis0.controller.config.control_mode = 3  # POSITION_CONTROL
    odrv.axis0.controller.config.input_mode = 1    # PASSTHROUGH
    odrv.axis0.requested_state = 8                 # CLOSED_LOOP_CONTROL
    time.sleep(0.1)

# Buffers para datos
enc1_vals, enc2_vals = [], []
u1_vals, u2_vals = [], []
error1_vals, error2_vals = [], []
iq1_vals, iq2_vals = [], []
z_vals = []
iteration_times = []

# Posición inicial
odrv1.axis0.controller.input_pos = df.iloc[0]['pos2']
odrv2.axis0.controller.input_pos = df.iloc[0]['pos1']
print("⏳ Motores en posición inicial. Presiona 'E' para ejecutar trayectoria...")

# Esperar inicio
while not keyboard.is_pressed('e'):
    time.sleep(0.01)

print("🎯 Ejecutando trayectoria (presiona 'Q' para detener)...")

# Variables compartidas para referencia y sincronización
current_refs = {'ref1': 0.0, 'ref2': 0.0, 'running': True}
ref_lock = threading.Lock()
shared_data = {'z': float('nan')}
data_lock = threading.Lock()

# Eventos para sincronización de muestras
sample_request = threading.Event()
sample_ready = threading.Event()

def read_thread_func():
    while current_refs['running']:
        # Esperar señal para tomar muestra
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

        # Avisar que la muestra ya fue tomada
        sample_ready.set()
        # Limpiar la señal para próxima muestra
        sample_request.clear()

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
        aligned_frames = align.process(frameset)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            with data_lock:
                shared_data['z'] = float('nan')
            continue

        color_image = np.asanyarray(color_frame.get_data())
        roi_color = color_image[y_min:y_max, x_min:x_max]

        hsv = cv2.cvtColor(roi_color, cv2.COLOR_BGR2HSV)
        lower_red1 = np.array([0, 70, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 70, 50])
        upper_red2 = np.array([180, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        result = cv2.bitwise_and(roi_color, roi_color, mask=mask)
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)

        circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT,
                                   dp=1, minDist=30,
                                   param1=100, param2=15,
                                   minRadius=7, maxRadius=50)

        z_val = float('nan')
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for c in circles[0, :]:
                x, y, r = c
                mask_roi = mask[max(y - r, 0):min(y + r, mask.shape[0]),
                                max(x - r, 0):min(x + r, mask.shape[1])]
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

read_thread = threading.Thread(target=read_thread_func)
cam_thread = threading.Thread(target=camera_thread_func)

read_thread.start()
cam_thread.start()

time.sleep(2)  # Esperar que los hilos se inicien

dt = 0.016
next_time = time.monotonic()
contador2 = time.monotonic()

try:
    for i, row in df.iterrows():
        start_iter = time.monotonic()

        with ref_lock:
            current_refs['ref1'] = row['pos2']
            current_refs['ref2'] = row['pos1']

        odrv1.axis0.controller.input_pos = row['pos2']
        odrv2.axis0.controller.input_pos = row['pos1']

        # Pedir muestra al hilo de lectura
        sample_request.set()

        # Esperar que el hilo de lectura termine la muestra
        sample_ready.wait()
        sample_ready.clear()

        next_time += dt
        now = time.monotonic()
        sleep_time = next_time - now
        if sleep_time > 0:
            time.sleep(sleep_time)

        end_iter = time.monotonic()
        iteration_times.append(end_iter - start_iter)

except KeyboardInterrupt:
    print("⛔ Detenido por el usuario (Ctrl+C).")

contador3 = time.monotonic()

# Finalizar
current_refs['running'] = False
# Despertar hilos bloqueados para salir
sample_request.set()
read_thread.join()

cam_thread.join()

for odrv in [odrv1, odrv2]:
    odrv.axis0.requested_state = 1
print("✅ Motores en estado IDLE.")

# Graficar resultados
plt.figure(figsize=(12, 16))

plt.subplot(6, 1, 1)
plt.plot(df['pos2_rad'], label='Referencia Motor 1', linestyle='--')
plt.plot(df['pos1_rad'], label='Referencia Motor 2', linestyle='--')
plt.plot(enc1_vals, label='Encoder Motor 1')
plt.plot(enc2_vals, label='Encoder Motor 2')
plt.title('Lectura de Encoders vs Referencia')
plt.ylabel('Posición (rad)')
plt.legend()

plt.subplot(6, 1, 2)
plt.plot(u1_vals, label='Salida PID Motor 1')
plt.plot(u2_vals, label='Salida PID Motor 2')
plt.title('Salida del PID')
plt.ylabel('Control (u)')
plt.legend()

plt.subplot(6, 1, 3)
plt.plot(error1_vals, label='Error Motor 1')
plt.plot(error2_vals, label='Error Motor 2')
plt.title('Error de Seguimiento')
plt.xlabel('Muestra')
plt.ylabel('Error (rad)')
plt.legend()

plt.subplot(6, 1, 4)
plt.plot(iq1_vals, label='Corriente Motor 1 (Iq)')
plt.plot(iq2_vals, label='Corriente Motor 2 (Iq)')
plt.title('Corriente de los Motores')
plt.xlabel('Muestra')
plt.ylabel('Corriente (A)')
plt.legend()

plt.subplot(6, 1, 5)
plt.plot(z_vals, label='Desplazamiento vertical estimado')
plt.title('Desplazamiento del círculo rojo (Z)')
plt.xlabel('Muestra')
#plt.xlim(0, 299)
plt.ylabel('Profundidad (m)')
plt.legend()

plt.subplot(6, 1, 6)
plt.plot(iteration_times, label='Tiempo de muestreo por iteración (s)')
plt.title('Tiempo de muestreo por iteración')
plt.xlabel('Iteración')
plt.ylabel('Duración (s)')
plt.legend()

plt.tight_layout()
plt.show()

contador4 = time.monotonic()

print(f"Tiempo desde inicio hasta antes de trayectoria: {contador2 - contador1:.3f} s")
print(f"Tiempo durante la trayectoria: {contador3 - contador2:.3f} s")
print(f"Tiempo desde finalización hasta impresión: {contador4 - contador3:.3f} s")
print(f"Tiempo total: {contador4 - contador1:.3f} s")

# print(f"Longitud final de z_vals: {len(z_vals)}")
# for i, val in enumerate(z_vals):
#     print(f"{i}: {val:.5f}")

