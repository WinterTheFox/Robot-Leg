import odrive
import time
import pandas as pd
import keyboard
import matplotlib.pyplot as plt
import math

# Leer archivo CSV
df = pd.read_csv('datos_positivos.csv', header=None, names=['pos1', 'pos2'])

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

# Activar motores en modo CLOSED_LOOP_CONTROL
for odrv in [odrv1, odrv2]:
    odrv.axis0.controller.config.control_mode = 3  # POSITION_CONTROL
    odrv.axis0.controller.config.input_mode = 1    # PASSTHROUGH
    odrv.axis0.requested_state = 8                 # CLOSED_LOOP_CONTROL
    time.sleep(0.1)

# Posicionarse en la posición inicial
odrv1.axis0.controller.input_pos = df.iloc[0]['pos2']
odrv2.axis0.controller.input_pos = df.iloc[0]['pos1']
print("⏳ Motores en posición inicial. Presiona 'E' para ejecutar trayectoria...")

# Esperar tecla de inicio
while not keyboard.is_pressed('e'):
    time.sleep(0.01)

print("🎯 Ejecutando trayectoria desde archivo (presiona 'Q' para detener)...")

# Inicializar buffers para datos
enc1_vals, enc2_vals = [], []
u1_vals, u2_vals = [], []
error1_vals, error2_vals = [], []

# Tiempo inicial
inicio = time.monotonic()

try:
    for i, row in df.iterrows():
        if keyboard.is_pressed('q'):
            print("⛔ Tecla 'Q' presionada. Deteniendo programa.")
            break

        ref1 = row['pos2']
        ref2 = row['pos1']

        odrv1.axis0.controller.input_pos = ref1
        odrv2.axis0.controller.input_pos = ref2

        enc1 = odrv1.axis0.pos_estimate * 2 * math.pi
        enc2 = odrv2.axis0.pos_estimate * 2 * math.pi
        ref1_rad = ref1 * 2 * math.pi
        ref2_rad = ref2 * 2 * math.pi
        u1 = odrv1.axis0.controller.config.pos_gain * (ref1 - odrv1.axis0.pos_estimate)
        u2 = odrv2.axis0.controller.config.pos_gain * (ref2 - odrv2.axis0.pos_estimate)

        enc1_vals.append(enc1)
        enc2_vals.append(enc2)
        u1_vals.append(u1)
        u2_vals.append(u2)
        error1_vals.append(ref1_rad - enc1)
        error2_vals.append(ref2_rad - enc2)

        while (time.monotonic() - inicio) < (i + 1) / 1000.0:
            time.sleep(0.0001)

except KeyboardInterrupt:
    print("⛔ Detenido por el usuario (Ctrl+C).")

# Liberar motores
for odrv in [odrv1, odrv2]:
    odrv.axis0.requested_state = 1  # IDLE
print("✅ Motores en estado IDLE.")

# Graficar resultados
plt.figure(figsize=(12, 8))

# Gráfica de posiciones (real vs referencia)
plt.subplot(3, 1, 1)
plt.plot(df['pos2_rad'], label='Referencia Motor 1', linestyle='--')
plt.plot(df['pos1_rad'], label='Referencia Motor 2', linestyle='--')
plt.plot(enc1_vals, label='Encoder Motor 1')
plt.plot(enc2_vals, label='Encoder Motor 2')
plt.title('Lectura de Encoders vs Referencia')
plt.ylabel('Posición (rad)')
plt.legend()

# Salida del PID
plt.subplot(3, 1, 2)
plt.plot(u1_vals, label='Salida PID Motor 1')
plt.plot(u2_vals, label='Salida PID Motor 2')
plt.title('Salida del PID')
plt.ylabel('Control (u)')
plt.legend()

# Error
plt.subplot(3, 1, 3)
plt.plot(error1_vals, label='Error Motor 1')
plt.plot(error2_vals, label='Error Motor 2')
plt.title('Error de Seguimiento')
plt.xlabel('Muestra')
plt.ylabel('Error (rad)')
plt.legend()

plt.tight_layout()
plt.show()
