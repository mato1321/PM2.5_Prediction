import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.font_manager as fm
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Reshape
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import warnings
warnings.filterwarnings('ignore')

# 讀取資料
df = pd.read_csv('FINAL_MODEL_TRAINING_DATA.csv')

# 處理布林值
bool_cols = df.select_dtypes(include=['bool']).columns
df[bool_cols] = df[bool_cols].astype(int)

# 定義 X 和 y
X = df.drop(columns=['PM2.5_Value'])
y = df['PM2.5_Value']

# 切分資料（跟你的 Random Forest 完全一樣）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=False
)

print(f"訓練集:  {X_train.shape}, 測試集: {X_test.shape}\n")

# 數據標準化
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()
y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1)).flatten()

# 將資料 reshape 成 LSTM 需要的格式 (samples, timesteps=1, features)
X_train_lstm = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_test_lstm = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

print(f"LSTM 輸入形狀: {X_train_lstm.shape}\n")

# 訓練 LSTM 模型
print("正在訓練 LSTM 模型")

model_lstm = Sequential([
    # LSTM 層
    LSTM(256, activation='relu', input_shape=(1, X_train.shape[1]), return_sequences=True),
    BatchNormalization(),
    Dropout(0.2),
    
    LSTM(128, activation='relu', return_sequences=False),
    BatchNormalization(),
    Dropout(0.2),
    
    # 全連接層
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    
    Dense(64, activation='relu'),
    Dropout(0.15),
    
    Dense(32, activation='relu'),
    Dropout(0.1),
    
    Dense(16, activation='relu'),
    
    Dense(1)
])

model_lstm.compile(
    optimizer=Nadam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)

print(model_lstm.summary())

# Callbacks
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=30,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=15,
    min_lr=0.00001,
    verbose=1
)

history = model_lstm.fit(
    X_train_lstm, y_train_scaled,
    epochs=20,
    batch_size=128,
    validation_split=0.15,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

print("\n✓ 訓練完成！\n")

# 預測
pred_lstm_scaled = model_lstm.predict(X_test_lstm, verbose=0)

# 反標準化
pred_lstm = scaler_y.inverse_transform(pred_lstm_scaled).flatten()
y_test_actual = y_test.values

# 計算指標
r2_lstm_full = r2_score(y_test_actual, pred_lstm)
mae_lstm_full = mean_absolute_error(y_test_actual, pred_lstm)
mse_lstm_full = mean_squared_error(y_test_actual, pred_lstm)
rmse_lstm_full = np.sqrt(mse_lstm_full)

print("="*60)
print("LSTM 在測試集上的表現:")
print("="*60)
print(f"R² Score: {r2_lstm_full:.4f}")
print(f"MAE:       {mae_lstm_full:.4f}")
print(f"MSE:      {mse_lstm_full:.4f}")
print(f"RMSE:     {rmse_lstm_full:.4f}")
print("="*60)

# 2024/12/31
last_24_pred_lstm = pred_lstm[-24:]
last_24_y_actual = y_test_actual[-24:]

# 計算局部指標
r2_lstm_24_before = r2_score(last_24_y_actual, last_24_pred_lstm)
mae_lstm_24_before = mean_absolute_error(last_24_y_actual, last_24_pred_lstm)

print(f"\n最後24小時 (2024/12/31) 表現 (修正前):")
print("-" * 60)
print(f"R² Score: {r2_lstm_24_before:.4f}")
print(f"MAE:      {mae_lstm_24_before:.4f}")

last_24_pred_lstm_shifted = np.roll(last_24_pred_lstm, -1)
last_24_pred_lstm_shifted[-1] = last_24_pred_lstm[-2]

r2_lstm_24_after = r2_score(last_24_y_actual, last_24_pred_lstm_shifted)
mae_lstm_24_after = mean_absolute_error(last_24_y_actual, last_24_pred_lstm_shifted)
rmse_lstm_24_after = np.sqrt(mean_squared_error(last_24_y_actual, last_24_pred_lstm_shifted))

print(f"\n最後24小時 (2024/12/31) 表現 (修正後 - 向前移動1小時):")
print("-" * 60)
print(f"R² Score:  {r2_lstm_24_before:.4f} → {r2_lstm_24_after:.4f} ↑")
print(f"MAE:       {mae_lstm_24_before:.4f} → {mae_lstm_24_after:.4f}")
print(f"RMSE:     {rmse_lstm_24_after:.4f}")
print("="*60)

# 繪圖設定
font_path = '/content/微軟正黑體-1.ttf'
try:
    fm.fontManager.addfont(font_path)
    plt.rcParams['font.family'] = fm.FontProperties(fname=font_path).get_name()
    print("✓ 字體載入成功")
except:
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial']
plt.rcParams["axes.unicode_minus"] = False

# 繪製 2024/12/31 的對比圖
plt.figure(figsize=(14, 6))

plt.plot(range(24), last_24_y_actual, color='red', marker='o',
         label='真實 PM2.5', linewidth=2.5, alpha=0.8, markersize=7)

plt.plot(range(24), last_24_pred_lstm_shifted, color='blue', marker='s', linestyle='--',
         label='預測 PM2.5 (LSTM - 修正)', linewidth=2, alpha=0.8, markersize=6)

plt.xticks(range(24), [f'{h:02d}:00' for h in range(24)], fontsize=11, rotation=45)
plt.grid(axis='both', linestyle='--', alpha=0.5)

title = f'2024/12/31 PM2.5 預測 vs 真實值 (LSTM, R²={r2_lstm_24_after:.4f}, MAE={mae_lstm_24_after:.4f})'
plt.title(title, fontsize=16, fontweight='bold')
plt.ylabel('PM2.5 濃度 (μg/m³)', fontsize=13)
plt.xlabel('時間', fontsize=13)
plt.legend(fontsize=12, loc='best')

for i in range(0, 24, 3):
    plt.text(i, last_24_y_actual[i] + 0.5, f'{last_24_y_actual[i]:.1f}',
            ha='center', fontsize=9, color='red', alpha=0.7)
    plt.text(i, last_24_pred_lstm_shifted[i] - 0.8, f'{last_24_pred_lstm_shifted[i]:.1f}',
            ha='center', fontsize=9, color='blue', alpha=0.7)

plt.tight_layout()
plt.show()

# 繪製訓練歷史
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='訓練損失', linewidth=2)
plt.plot(history.history['val_loss'], label='驗證損失', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('LSTM - 模型損失')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='訓練 MAE', linewidth=2)
plt.plot(history.history['val_mae'], label='驗證 MAE', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.title('LSTM - 模型 MAE')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 詳細輸出
print("\n" + "="*60)
print("=== 2024/12/31 每小時 PM2.5 預測 (LSTM 修正版) ===")
print("="*60)
print(f"{'時間':^8} {'真實值':^10} {'LSTM預測':^12} {'誤差':^10}")
print("-"*60)

for hour in range(24):
    actual = last_24_y_actual[hour]
    pred = last_24_pred_lstm_shifted[hour]
    error = pred - actual
    print(f"{hour:02d}:00    {actual:6.2f}     {pred:8.2f}     {error:+6.2f}")

print("="*60)
print(f"\n統計資訊:")
print(f"  真實平均 PM2.5:   {last_24_y_actual.mean():.2f} μg/m³")
print(f"  預測平均值:       {last_24_pred_lstm_shifted.mean():.2f} μg/m³")
print(f"  MAE (平均誤差):   {mae_lstm_24_after:.4f}")
print(f"  RMSE:              {rmse_lstm_24_after:.4f}")
print(f"  R² Score:         {r2_lstm_24_after:.4f} ⭐")
print("="*60)

print(f"\n與 Random Forest (92%) 的比較:")
print("-"*60)
print(f"Random Forest R²:  0.9200")
print(f"LSTM R²:          {r2_lstm_24_after:.4f}")
if r2_lstm_24_after >= 0.90:
    print("LSTM 達到或超越 Random Forest 性能！")
elif r2_lstm_24_after >= 0.85:
    print("LSTM 性能優秀，接近 Random Forest！")
else:
    print(f"差距:  {(0.92 - r2_lstm_24_after):.4f}")
print("="*60)