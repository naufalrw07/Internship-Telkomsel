import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from fastapi import HTTPException
from api.db import get_connection

# Mapping sensor per room
ROOM_MAP = {
    "gayungan": {
        "ROOM1": ["DHT1", "DHT2", "DHT3", "DHT4"],
        "ROOM2": ["DHT5"],
        "ROOM3": ["DHT6"],
        "ROOM4": ["DHT7", "DHT8", "DHT9"],
        "ROOM5": ["DHT10", "DHT11", "DHT12"]
    },
    "kebalen": {
        "ROOM1": ["DHT1", "DHT2", "DHT3", "DHT4"],
        "ROOM2": ["DHT5", "DHT6"]
    }
}


def fetch_data(location: str, room: str, hours: int):
    """
    Ambil data dari MySQL sesuai lokasi & room → rata-rata per 5 menit → ambil N terakhir.
    """
    sensors_in_room = ROOM_MAP[location][room]
    records_needed = hours * 12  # 12 titik per jam (5 menit sekali)

    conn = get_connection()
    if conn is None:
        raise HTTPException(status_code=500, detail="Gagal koneksi DB")

    cursor = conn.cursor(dictionary=True)

    query = f"""
    SELECT time_id, sensor_id, temperature, humidity
    FROM {location}
    WHERE sensor_id IN ({','.join(['%s'] * len(sensors_in_room))})
    ORDER BY time_id DESC
    """
    cursor.execute(query, tuple(sensors_in_room))
    rows = cursor.fetchall()

    cursor.close()
    conn.close()

    if not rows:
        raise HTTPException(status_code=404, detail="Data tidak ditemukan")

    # Proses rata-rata per 5 menit
    df = pd.DataFrame(rows)
    df["time_id"] = pd.to_datetime(df["time_id"])
    df["timestamp_5min"] = df["time_id"].dt.floor("5min")

    avg_df = (
        df.groupby("timestamp_5min")[["temperature", "humidity"]]
        .mean()
        .reset_index()
        .sort_values("timestamp_5min")
    )

    # Ambil data terakhir sesuai durasi request
    result = avg_df.tail(records_needed).to_dict(orient="records")

    if len(result) < records_needed:
        raise HTTPException(
            status_code=400,
            detail=f"Data tidak cukup: butuh {records_needed}, hanya ada {len(result)}"
        )

    return result


def make_prediction(models_dict, lokasi: str, room: str, duration: int):
    """
    Ambil data asli dari DB → rata-rata → prediksi sesuai durasi request.
    """
    # Ambil sequence awal sepanjang durasi request
    data = fetch_data(lokasi, room, hours=duration)

    # Input sequence awal
    X_temp = [d["temperature"] for d in data]
    X_hum = [d["humidity"] for d in data]

    # Ambil model & scaler
    temp_model, temp_scaler = models_dict["temperature"][room]
    hum_model, hum_scaler = models_dict["humidity"][room]

    # Tentukan waktu awal prediksi (dibulatkan ke 5 menit berikutnya)
    now = datetime.now()
    minute_offset = (5 - (now.minute % 5)) % 5
    if minute_offset == 0:
        minute_offset = 5
    start_time = now.replace(second=0, microsecond=0) + timedelta(minutes=minute_offset)

    predictions = []

    for i in range(duration * 12):  # prediksi tiap 5 menit
        # ---- Temperature ----
        input_temp = np.array(X_temp[-12:]).reshape(-1, 1)
        input_temp_scaled = temp_scaler.transform(input_temp)
        input_temp_scaled = input_temp_scaled.reshape(1, input_temp_scaled.shape[0], 1)

        next_temp_scaled = temp_model.predict(input_temp_scaled, verbose=0)
        next_temp = temp_scaler.inverse_transform(next_temp_scaled)[0][0]

        # ---- Humidity ----
        input_hum = np.array(X_hum[-12:]).reshape(-1, 1)
        input_hum_scaled = hum_scaler.transform(input_hum)
        input_hum_scaled = input_hum_scaled.reshape(1, input_hum_scaled.shape[0], 1)

        next_hum_scaled = hum_model.predict(input_hum_scaled, verbose=0)
        next_hum = hum_scaler.inverse_transform(next_hum_scaled)[0][0]

        # Timestamp prediksi
        pred_time = start_time + timedelta(minutes=5 * i)

        predictions.append({
            "timestamp": pred_time.isoformat(),
            "temperature": round(float(next_temp), 2),
            "humidity": round(float(next_hum), 2)
        })

        # Update sequence untuk step berikutnya
        X_temp.append(next_temp)
        X_hum.append(next_hum)

    return {
        "room": room,
        "predictions": predictions
    }


def save_predictions(lokasi, room, predictions):
    """
    Simpan hasil prediksi ke DB.
    """
    table_name = f"prediksi_{lokasi}"
    conn = get_connection()
    if conn is None:
        print("Database connection failed")
        return

    cursor = conn.cursor()

    insert_query = f"""
    INSERT INTO {table_name} (room_id, timestamp, temperature, humidity)
    VALUES (%s, %s, %s, %s)
    """

    data = [(room, p["timestamp"], p["temperature"], p["humidity"]) for p in predictions]

    try:
        cursor.executemany(insert_query, data)
        conn.commit()
        print(f"Inserted {len(data)} rows into {table_name}")
    except Exception as e:
        print("Insert error:", e)
        conn.rollback()
    finally:
        cursor.close()
        conn.close()
