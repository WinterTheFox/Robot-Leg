import pandas as pd
import matplotlib.pyplot as plt

# Cargar archivo CSV
df = pd.read_csv("datos_trayectoria.csv")

# Crear figura con subplots
fig, axs = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
fig.suptitle("Análisis de trayectoria", fontsize=16)

# Subplot 1: Ángulos de encoders
axs[0].plot(df['tiempo_s'], df['enc1_rad'], label='Encoder 1 [rad]')
axs[0].plot(df['tiempo_s'], df['enc2_rad'], label='Encoder 2 [rad]')
axs[0].set_ylabel("Ángulo [rad]")
axs[0].set_title("Ángulos de Encoders")
axs[0].legend()
axs[0].grid(True)

# Subplot 2: Profundidad
axs[1].plot(df['tiempo_s'], df['profundidad_m'], color='purple', label='Profundidad [m]')
axs[1].set_ylabel("Profundidad [m]")
axs[1].set_title("Desplazamiento vertical")
axs[1].legend()
axs[1].grid(True)

# Subplot 3: Duración de muestra
axs[2].plot(df['tiempo_s'], df['duracion_muestra_s'], color='orange', label='Duración de muestra [s]')
axs[2].set_ylabel("Duración [s]")
axs[2].set_title("Duración entre muestras")
axs[2].legend()
axs[2].grid(True)

# Subplot 4: Evento de cambio (si existe)
if 'evento_cambio' in df.columns:
    axs[3].plot(df['tiempo_s'], df['evento_cambio'], label='Evento de cambio', linestyle='--', marker='o')
    axs[3].set_ylabel("Evento")
    axs[3].set_title("Cambio de trayectoria")
    axs[3].legend()
    axs[3].grid(True)
else:
    axs[3].axis('off')  # Oculta si no hay datos

# Etiqueta del eje X general
plt.xlabel("Tiempo [s]")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
