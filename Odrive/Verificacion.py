import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# === Cargar CSV ===
df = pd.read_csv("datos_trayectoria.csv")

# === Extraer datos necesarios ===
theta1 = df['enc2_rad']  # ← enc2_rad representa el primer ángulo
theta2 = df['enc1_rad']  # ← enc1_rad representa el segundo ángulo
z_real = df['profundidad_m']
tiempo = df['tiempo_s']

# === Parámetros del mecanismo ===
l1 = 0.2    # Longitud del primer eslabón (m)
l2 = 0.18   # Longitud del segundo eslabón (m)
#offset = 0.2765-0.028  # Offset vertical (m)
offset = 0.0150  # Offset vertical (m)

# === Calcular Z0 teórico ===
z_ref = l1 * np.sin(theta1) + l2 * np.sin(theta1 + theta2) + offset

# === Graficar comparación ===
plt.figure(figsize=(10, 6))
plt.plot(tiempo, z_real, label="Z medida (RealSense)", color="blue")
plt.plot(tiempo, z_ref, label="Z teórica (modelo)", color="orange", linestyle='--')
plt.xlabel("Tiempo (s)")
plt.ylabel("Desplazamiento Z (m)")
plt.title("Comparación entre Z medida y Z teórica")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
