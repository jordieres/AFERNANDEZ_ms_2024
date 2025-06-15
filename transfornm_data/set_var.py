import pandas as pd

archivo = r"C:\Users\Gliglo\OneDrive - Universidad Politécnica de Madrid\Documentos\UPM\TFG\Proyecto_TFG\AFERNANDEZ_ms_2024\test_InfluxDB\out\dat_2024_tabuenca_left.xlsx"
df = pd.read_excel(archivo)


startTime = 0

# === CALCULA samplePeriod Y stopTime DESDE _time ===
if '_time' in df.columns:
    try:
        df['_time'] = pd.to_datetime(df['_time'])
        df = df.sort_values('_time').reset_index(drop=True)
        dt = df['_time'].diff().dt.total_seconds().dropna()
        samplePeriod = dt.mean()
        frecuencia = 1 / samplePeriod
        stopTime = (len(df) - 1) * samplePeriod

        print("✅ Datos calculados a partir de los timestamps:")
        print(f"startTime = {startTime}")
        print(f"stopTime = {stopTime:.6f}  # Duración total")
        print(f"samplePeriod = {samplePeriod:.8f}")
        print(f"frecuencia = {frecuencia:.2f} Hz")
    except Exception as e:
        print("⚠️ Error al procesar la columna '_time':", e)
else:
    print("❌ La columna '_time' no existe. No se puede calcular automáticamente.")
    print("✏️ Establece manualmente el samplePeriod y la frecuencia.")
