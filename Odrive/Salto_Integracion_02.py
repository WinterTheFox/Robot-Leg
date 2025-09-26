import odrive
import time
import pandas as pd
import keyboard
import matplotlib.pyplot as plt
import math
import threading

contador1 = time.monotonic()

# Leer archivo CSV
df = pd.read_csv('datos1s.csv', header=None, names=['pos1', 'pos2'])
df = df.iloc[::1].reset_index(drop=True)

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

# Buffers
enc1_vals, enc2_vals = [], []
u1_vals, u2_vals = [], []
error1_vals, error2_vals = [], []
iq1_vals, iq2_vals = [], []

# Posición inicial
odrv1.axis0.controller.input_pos = df.iloc[0]['pos2']
odrv2.axis0.controller.input_pos = df.iloc[0]['pos1']
print("⏳ Motores en posición inicial. Presiona 'E' para ejecutar trayectoria...")

# Esperar inicio
while not keyboard.is_pressed('e'):
    time.sleep(0.01)

print("🎯 Ejecutando trayectoria (presiona 'Q' para detener)...")

contador2 = time.monotonic()

# Variables compartidas para referencia
current_refs = {'ref1': 0.0, 'ref2': 0.0, 'running': True}
ref_lock = threading.Lock()

def read_thread_func():
    while current_refs['running']:
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

        enc1_vals.append(enc1)
        enc2_vals.append(enc2)
        u1_vals.append(u1)
        u2_vals.append(u2)
        error1_vals.append(ref1_rad - enc1)
        error2_vals.append(ref2_rad - enc2)
        iq1_vals.append(iq1)
        iq2_vals.append(iq2)

        time.sleep(0.001)

read_thread = threading.Thread(target=read_thread_func)
read_thread.start()

# Hilo de control
dt = 0.001
next_time = time.monotonic()

try:
    for i, row in df.iterrows():
        if keyboard.is_pressed('q'):
            print("⛔ Tecla 'Q' presionada. Deteniendo programa.")
            break

        with ref_lock:
            current_refs['ref1'] = row['pos2']
            current_refs['ref2'] = row['pos1']

        odrv1.axis0.controller.input_pos = row['pos2']
        odrv2.axis0.controller.input_pos = row['pos1']

        next_time += dt
        sleep_time = next_time - time.monotonic()
        if sleep_time > 0:
            time.sleep(sleep_time)
        else:
            print(f"⚠️ Iteración {i} retrasada por {-sleep_time:.4f} s")

except KeyboardInterrupt:
    print("⛔ Detenido por el usuario (Ctrl+C).")

contador3 = time.monotonic()

# Finalizar
current_refs['running'] = False
read_thread.join()

for odrv in [odrv1, odrv2]:
    odrv.axis0.requested_state = 1
print("✅ Motores en estado IDLE.")

# Gráficas
plt.figure(figsize=(12, 12))

plt.subplot(4, 1, 1)
plt.plot(df['pos2_rad'], label='Referencia Motor 1', linestyle='--')
plt.plot(df['pos1_rad'], label='Referencia Motor 2', linestyle='--')
plt.plot(enc1_vals, label='Encoder Motor 1')
plt.plot(enc2_vals, label='Encoder Motor 2')
plt.title('Lectura de Encoders vs Referencia')
plt.ylabel('Posición (rad)')
plt.legend()

plt.subplot(4, 1, 2)
plt.plot(u1_vals, label='Salida PID Motor 1')
plt.plot(u2_vals, label='Salida PID Motor 2')
plt.title('Salida del PID')
plt.ylabel('Control (u)')
plt.legend()

plt.subplot(4, 1, 3)
plt.plot(error1_vals, label='Error Motor 1')
plt.plot(error2_vals, label='Error Motor 2')
plt.title('Error de Seguimiento')
plt.xlabel('Muestra')
plt.ylabel('Error (rad)')
plt.legend()

plt.subplot(4, 1, 4)
plt.plot(iq1_vals, label='Corriente Motor 1 (Iq)')
plt.plot(iq2_vals, label='Corriente Motor 2 (Iq)')
plt.title('Corriente de los Motores')
plt.xlabel('Muestra')
plt.ylabel('Corriente (A)')
plt.legend()

plt.tight_layout()
plt.show()

contador4 = time.monotonic()

print(f"Tiempo desde inicio hasta antes de trayectoria: {contador2 - contador1:.3f} s")
print(f"Tiempo durante la trayectoria: {contador3 - contador2:.3f} s")
print(f"Tiempo desde finalización hasta impresión: {contador4 - contador3:.3f} s")
print(f"Tiempo total: {contador4 - contador1:.3f} s")
