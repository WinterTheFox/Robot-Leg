import odrive
import time
import pandas as pd
import math
import matplotlib.pyplot as plt

# Leer archivo CSV
print("📂 Cargando trayectoria...")
df = pd.read_csv('datos1s.csv', header=None, names=['pos1', 'pos2'])
df = df.iloc[::16].reset_index(drop=True)
df['pos1_rad'] = df['pos1'] * 2 * math.pi
df['pos2_rad'] = df['pos2'] * 2 * math.pi

# Conectar con los ODrives
print("🔍 Buscando ODrives...")
odrv1 = odrive.find_any(serial_number="384B34733539")
print("✅ ODrive 1 conectado")
odrv2 = odrive.find_any(serial_number="384434593539")
print("✅ ODrive 2 conectado")

# Activar motores
for odrv in [odrv1, odrv2]:
    odrv.axis0.controller.config.control_mode = 3
    odrv.axis0.controller.config.input_mode = 1
    odrv.axis0.requested_state = 8

# Posición inicial
odrv1.axis0.controller.input_pos = df.iloc[0]['pos2']
odrv2.axis0.controller.input_pos = df.iloc[0]['pos1']
print("🏁 Motores listos. Ejecutando trayectoria...")

# Buffers
iteration_times = []

dt = 0.016  # 16 ms (60 Hz)
start_time = time.perf_counter()

ban_3 = time.monotonic()
for i, row in df.iterrows():
    target_time = start_time + (i + 1) * dt

    # Enviar comando
    odrv1.axis0.controller.input_pos = row['pos2']
    odrv2.axis0.controller.input_pos = row['pos1']

    # Espera activa con cálculo de tiempo restante
    ban_1 = time.monotonic()
    while True:
        remaining = target_time - time.perf_counter()
        if remaining <= 0:
            break 
        elif remaining > 0.002:
            time.sleep(remaining - 0.001)
        else:
            pass  # espera activa corta
    ban_2 = time.monotonic()

    iteration_times.append(ban_2 - ban_1)

ban_4 = time.monotonic()

print(f"Tiempo total de ejecución: {ban_4 - ban_3:.4f} segundos")

# Graficar
plt.plot(iteration_times, label='Tiempo de muestreo por iteración (s)')
plt.title('Tiempo de muestreo por iteración')
plt.xlabel('Iteración')
plt.ylabel('Duración (s)')
plt.legend()
plt.grid()
plt.show()
