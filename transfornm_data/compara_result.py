import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# === 1. Cargar resultados ===
df_ahrs = pd.read_excel('salida_ahrs.xlsx')
df_skin = pd.read_excel('salida_skinematics.xlsx')

# === 2. Asegurar que tienen misma longitud ===
min_len = min(len(df_ahrs), len(df_skin))
df_ahrs = df_ahrs.iloc[:min_len]
df_skin = df_skin.iloc[:min_len]

# === 3. Calcular error absoluto medio por eje ===
error_pos = np.abs(df_ahrs[['Pos_X', 'Pos_Y', 'Pos_Z']].values -
                   df_skin[['Pos_X', 'Pos_Y', 'Pos_Z']].values)

mae_x = np.mean(error_pos[:, 0])
mae_y = np.mean(error_pos[:, 1])
mae_z = np.mean(error_pos[:, 2])
error_total = np.linalg.norm(error_pos, axis=1).mean()

# === 4. Mostrar resultados ===
print("=== Comparación de librerías ===")
print(f"Error medio absoluto en X: {mae_x:.4f} m")
print(f"Error medio absoluto en Y: {mae_y:.4f} m")
print(f"Error medio absoluto en Z: {mae_z:.4f} m")
print(f"Error total promedio: {error_total:.4f} m")

mejor = 'Skinematics' if error_total < 0.5 else 'AHRS (Madgwick)'
print(f"\n>>> Se recomienda usar: {mejor}")

# === 5. Gráfico comparativo ===
plt.figure(figsize=(8, 6))
plt.plot(df_ahrs['Pos_X'], df_ahrs['Pos_Y'], label='AHRS (Madgwick)')
plt.plot(df_skin['Pos_X'], df_skin['Pos_Y'], label='Skinematics')
plt.title('Comparación de Trayectorias Estimadas')
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig('grafico_comparacion.png')
plt.show()
