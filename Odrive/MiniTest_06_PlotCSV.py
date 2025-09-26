import pandas as pd
import matplotlib.pyplot as plt
import math

# Leer archivo CSV (ajusta la ruta si es necesario)
df = pd.read_csv('datos1s.csv', header=None, names=['pos1', 'pos2'])

# Convertir a radianes
df['pos1_rad'] = df['pos1'] * 2 * math.pi
df['pos2_rad'] = df['pos2'] * 2 * math.pi

# Graficar posiciones en revoluciones
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(df['pos1'], label='Pos1 (rev)')
plt.plot(df['pos2'], label='Pos2 (rev)')
plt.title('Posiciones en Revoluciones')
plt.xlabel('Muestra')
plt.ylabel('Revoluciones')
plt.legend()

# Graficar posiciones en radianes
plt.subplot(1, 2, 2)
plt.plot(df['pos1_rad'], label='Pos1 (rad)')
plt.plot(df['pos2_rad'], label='Pos2 (rad)')
plt.title('Posiciones en Radianes')
plt.xlabel('Muestra')
plt.ylabel('Radianes')
plt.legend()

plt.tight_layout()
plt.show()
