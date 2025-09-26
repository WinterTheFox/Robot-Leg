import odrive
import time
import pandas as pd
import keyboard  # pip install keyboard

# 📂 Leer archivo CSV
df = pd.read_csv('datos_positivos.csv', header=None, names=['pos1', 'pos2'])

# 🔌 Conexión con los ODrives usando números de serie
print("🔍 Buscando ODrives...")
odrv1 = odrive.find_any(serial_number="384B34733539")
print("✅ ODrive 1 conectado")
time.sleep(1.0)
odrv2 = odrive.find_any(serial_number="384434593539")
print("✅ ODrive 2 conectado")

# 🛠 Activar motores en modo CLOSED_LOOP_CONTROL
for odrv in [odrv1, odrv2]:
    odrv.axis0.controller.config.control_mode = 3  # POSITION_CONTROL
    odrv.axis0.controller.config.input_mode = 1    # PASSTHROUGH
    odrv.axis0.requested_state = 8                 # CLOSED_LOOP_CONTROL
    time.sleep(0.1)

# 🎯 Ejecutar trayectoria
print("🎯 Ejecutando trayectoria desde archivo (presiona 'Q' para detener)...")

# Tiempo inicial
inicio = time.monotonic()

try:
    for i, row in df.iterrows():

        # Verificar si se presionó 'q'
        if keyboard.is_pressed('q'):
            print("🛑 Tecla 'Q' presionada. Deteniendo programa.")
            break

        # Asignar posición
        odrv1.axis0.controller.input_pos = row['pos2']
        odrv2.axis0.controller.input_pos = row['pos1']

        # Esperar a que se cumpla el tiempo deseado (1 ms por muestra)
        while (time.monotonic() - inicio) < (i + 1) / 1000.0:
            time.sleep(0.0001)

except KeyboardInterrupt:
    print("🛑 Detenido por el usuario (Ctrl+C).")

# 💤 Liberar motores
for odrv in [odrv1, odrv2]:
    odrv.axis0.requested_state = 1  # IDLE
print("✅ Motores en estado IDLE.")
