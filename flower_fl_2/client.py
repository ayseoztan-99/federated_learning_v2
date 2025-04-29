#keras_client.py

import flwr as fl
import tensorflow as tf
import pandas as pd
import numpy as np
import os
import sys
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from config import SERVER_ADDRESS, CLIENT_DATA_DIR, TH, TD, TW, TP, LOCAL_EPOCHS
from model import build_multi_lstm_model

# GPU Ayarları
print("Available GPUs:", tf.config.list_physical_devices('GPU'))
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

logical_gpus = tf.config.experimental.list_logical_devices('GPU')
print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")

client_id = int(sys.argv[1]) if len(sys.argv) > 1 else 1
print(f"[Client {client_id}] Initializing...")

# Veri Yükleme ve Sıralama
file_path = os.path.join(CLIENT_DATA_DIR, f"client_data_{client_id}.csv")
df = pd.read_csv(file_path)
df = df.sort_values(by=["location", "timestep"]).reset_index(drop=True)

timesteps_per_day = 288
train_days = 50
test_days = 12
train_size = train_days * timesteps_per_day
test_size = test_days * timesteps_per_day

locations = df["location"].unique()

# Train/Test Ham Ayrımı (günlük pencere ile)
train_data = df.iloc[:train_size]
test_data = df.iloc[train_size:train_size + test_size]

# Özellik çıkarımı fonksiyonu
def extract_features(data, th, td, tw, tp, timesteps_per_day):
    X_recent, X_daily, X_weekly, Y = [], [], [], []
    for loc in data["location"].unique():
        df_loc = data[data["location"] == loc].reset_index(drop=True)
        for i in range(max(th, td, tw), len(df_loc) - tp):
            recent = df_loc["flow"].iloc[i - th:i].values
            daily = df_loc["flow"].iloc[i - td - timesteps_per_day:i - timesteps_per_day].values
            weekly = df_loc["flow"].iloc[i - tw - 7 * timesteps_per_day:i - 7 * timesteps_per_day].values
            target = df_loc["flow"].iloc[i:i + tp].values
            if len(recent) == th and len(daily) == td and len(weekly) == tw and len(target) == tp:
                X_recent.append(recent)
                X_daily.append(daily)
                X_weekly.append(weekly)
                Y.append(target)
    return (np.array(X_recent), np.array(X_daily), np.array(X_weekly), np.array(Y))

# Train pencerelerini oluştur
X_recent_full, X_daily_full, X_weekly_full, Y_full = extract_features(
    train_data, TH, TD, TW, TP, timesteps_per_day
)

# Train örneklerini %80 train, %20 validation olarak böl
num_total = X_recent_full.shape[0]
num_train = int(num_total * 0.8)
num_val = num_total - num_train

X_recent_train = X_recent_full[:num_train]
X_daily_train  = X_daily_full[:num_train]
X_weekly_train = X_weekly_full[:num_train]
Y_train        = Y_full[:num_train]

X_recent_val = X_recent_full[num_train:]
X_daily_val  = X_daily_full[num_train:]
X_weekly_val = X_weekly_full[num_train:]
Y_val        = Y_full[num_train:]

# Test pencerelerini oluştur
X_recent_test, X_daily_test, X_weekly_test, Y_test = extract_features(
    test_data, TH, TD, TW, TP, timesteps_per_day
)

# Reshape işlemleri
X_recent_train = X_recent_train.reshape(-1, TH, 1)
X_daily_train = X_daily_train.reshape(-1, TD, 1)
X_weekly_train = X_weekly_train.reshape(-1, TW, 1)
Y_train = Y_train.reshape(-1, TP)

X_recent_val = X_recent_val.reshape(-1, TH, 1)
X_daily_val = X_daily_val.reshape(-1, TD, 1)
X_weekly_val = X_weekly_val.reshape(-1, TW, 1)
Y_val = Y_val.reshape(-1, TP)

X_recent_test = X_recent_test.reshape(-1, TH, 1)
X_daily_test = X_daily_test.reshape(-1, TD, 1)
X_weekly_test = X_weekly_test.reshape(-1, TW, 1)
Y_test = Y_test.reshape(-1, TP)

# Sadece train ile scaler fit et, diğerlerinde transform kullan
scaler_y = MinMaxScaler()
scaler_y.fit(Y_train)
Y_train_scaled = scaler_y.transform(Y_train)
Y_val_scaled = scaler_y.transform(Y_val)
Y_test_scaled = scaler_y.transform(Y_test)

y_train_dict = {
    "output_recent": Y_train_scaled,
    "output_daily": Y_train_scaled,
    "output_weekly": Y_train_scaled,
    "final_output": Y_train_scaled
}
y_val_dict = {
    "output_recent": Y_val_scaled,
    "output_daily": Y_val_scaled,
    "output_weekly": Y_val_scaled,
    "final_output": Y_val_scaled
}
y_test_dict = {
    "output_recent": Y_test_scaled,
    "output_daily": Y_test_scaled,
    "output_weekly": Y_test_scaled,
    "final_output": Y_test_scaled
}

# Modeli oluştur
model = build_multi_lstm_model(TH, TD, TW, TP)
print(f"[Client {client_id}] Model has been built and compiled.")

# FL Client sınıfı
class FLClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        model.fit(
            [X_recent_train, X_daily_train, X_weekly_train],
            y_train_dict,
            epochs=LOCAL_EPOCHS,
            batch_size=32,
            validation_data=(
                [X_recent_val, X_daily_val, X_weekly_val],
                y_val_dict
            ),
            verbose=0
        )
        return model.get_weights(), len(X_recent_train), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        preds = model.predict([X_recent_test, X_daily_test, X_weekly_test], verbose=0)
        Y_pred_inv = scaler_y.inverse_transform(preds[3])
        Y_test_inv = scaler_y.inverse_transform(Y_test_scaled)

        loss = model.evaluate([X_recent_test, X_daily_test, X_weekly_test], y_test_dict, verbose=0)
        rmse = np.sqrt(mean_squared_error(Y_test_inv, Y_pred_inv))
        r2 = r2_score(Y_test_inv, Y_pred_inv)
        mae = np.mean(np.abs(Y_test_inv - Y_pred_inv))
        mape = np.mean(np.abs((Y_test_inv - Y_pred_inv) / (Y_test_inv + 1e-5))) * 100

        print(f"[Client {client_id}] Loss: {loss[0]:.4f}, RMSE: {rmse:.2f}, R²: {r2:.4f}, MAE: {mae:.2f}, MAPE: {mape:.2f}%")

        return loss[0], len(X_recent_test), {
            "rmse": rmse,
            "r2": r2,
            "mae": mae,
            "mape": mape,
            "client_id": client_id
        }

if __name__ == "__main__":
    fl.client.start_client(
        server_address=SERVER_ADDRESS,
        client=FLClient().to_client()
    )

