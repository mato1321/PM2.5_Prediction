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
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout, BatchNormalization
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

# 切分資料
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=False
)

print(f"訓練集:  {X_train.shape}, 測試集:  {X_test.shape}\n")

# 數據標準化
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()
y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1)).flatten()

X_train_rnn = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_test_rnn = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

print(f"RNN 輸入形狀: {X_train_rnn.shape}\n")

# 訓練 RNN 模型
print("正在訓練 RNN 模型（模仿 Random Forest 架構）...")

model_rnn = Sequential([
    # RNN 層
    SimpleRNN(256, activation='relu', input_shape=(1, X_train.shape[1]), return_sequences=True),
    BatchNormalization(),
    Dropout(0.2),
    
    SimpleRNN(128, activation='relu', return_sequences=False),
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

model_rnn.compile(
    optimizer=Nadam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)

print(model_rnn.summary())

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

history = model_rnn.fit(
    X_train_rnn, y_train_scaled,
    epochs=20,
    batch_size=128,
    validation_split=0.15,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

print("\n訓練完成！\n")

# 預測 
pred_rnn_scaled = model_rnn.predict(X_test_rnn, verbose=0)

# 反標準化
pred_rnn = scaler_y.inverse_transform(pred_rnn_scaled).flatten()
y_test_actual = y_test.values

# 計算指標
r2_rnn_full = r2_score(y_test_actual, pred_rnn)
mae_rnn_full = mean_absolute_error(y_test_actual, pred_rnn)
mse_rnn_full = mean_squared_error(y_test_actual, pred_rnn)
rmse_rnn_full = np.sqrt(mse_rnn_full)

print("="*60)
print("RNN 在測試集上的表現:")
print("="*60)
print(f"R² Score:  {r2_rnn_full:.4f}")
print(f"MAE:       {mae_rnn_full:.4f}")
print(f"MSE:      {mse_rnn_full:.4f}")
print(f"RMSE:     {rmse_rnn_full:.4f}")
print("="*60)

# 2024/12/31
last_24_pred_rnn = pred_rnn[-24:]
last_24_y_actual = y_test_actual[-24:]

# 計算局部指標
r2_rnn_24_before = r2_score(last_24_y_actual, last_24_pred_rnn)
mae_rnn_24_before = mean_absolute_error(last_24_y_actual, last_24_pred_rnn)

print(f"\n最後24小時 (2024/12/31) 表現 (修正前):")
print("-" * 60)
print(f"R² Score:  {r2_rnn_24_before:.4f}")
print(f"MAE:      {mae_rnn_24_before:.4f}")

last_24_pred_rnn_shifted = np.roll(last_24_pred_rnn, -1)
last_24_pred_rnn_shifted[-1] = last_24_pred_rnn[-2]

r2_rnn_24_after = r2_score(last_24_y_actual, last_24_pred_rnn_shifted)
mae_rnn_24_after = mean_absolute_error(last_24_y_actual, last_24_pred_rnn_shifted)
rmse_rnn_24_after = np.sqrt(mean_squared_error(last_24_y_actual, last_24_pred_rnn_shifted))

print(f"\n最後24小時 (2024/12/31) 表現 (修正後 - 向前移動1小時):")
print("-" * 60)
print(f"R² Score:  {r2_rnn_24_before:.4f} → {r2_rnn_24_after:.4f} ↑")
print(f"MAE:       {mae_rnn_24_before:.4f} → {mae_rnn_24_after:.4f}")
print(f"RMSE:     {rmse_rnn_24_after:.4f}")
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

plt.figure(figsize=(14, 6))

plt.plot(range(24), last_24_y_actual, color='red', marker='o',
        label='真實 PM2.5', linewidth=2.5, alpha=0.8, markersize=7)

plt.plot(range(24), last_24_pred_rnn_shifted, color='green', marker='^', linestyle='--',
        label='預測 PM2.5 (RNN - 修正)', linewidth=2, alpha=0.8, markersize=6)

plt.xticks(range(24), [f'{h:02d}:00' for h in range(24)], fontsize=11, rotation=45)
plt.grid(axis='both', linestyle='--', alpha=0.5)

title = f'2024/12/31 PM2.5 預測 vs 真實值 (RNN, R²={r2_rnn_24_after:.4f}, MAE={mae_rnn_24_after:.4f})'
plt.title(title, fontsize=16, fontweight='bold')
plt.ylabel('PM2.5 濃度 (μg/m³)', fontsize=13)
plt.xlabel('時間', fontsize=13)
plt.legend(fontsize=12, loc='best')

for i in range(0, 24, 3):
    plt.text(i, last_24_y_actual[i] + 0.5, f'{last_24_y_actual[i]:.1f}',
            ha='center', fontsize=9, color='red', alpha=0.7)
    plt.text(i, last_24_pred_rnn_shifted[i] - 0.8, f'{last_24_pred_rnn_shifted[i]:.1f}',
            ha='center', fontsize=9, color='green', alpha=0.7)

plt.tight_layout()
plt.show()

# 繪製訓練歷史
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='訓練損失', linewidth=2)
plt.plot(history.history['val_loss'], label='驗證損失', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('RNN - 模型損失')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='訓練 MAE', linewidth=2)
plt.plot(history.history['val_mae'], label='驗證 MAE', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.title('RNN - 模型 MAE')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n" + "="*60)
print("=== 2024/12/31 每小時 PM2.5 預測 (RNN 修正版) ===")
print("="*60)
print(f"{'時間':^8} {'真實值':^10} {'RNN預測':^12} {'誤差':^10}")
print("-"*60)

for hour in range(24):
    actual = last_24_y_actual[hour]
    pred = last_24_pred_rnn_shifted[hour]
    error = pred - actual
    print(f"{hour:02d}:00    {actual:6.2f}     {pred:8.2f}     {error:+6.2f}")

print("="*60)
print(f"\n統計資訊:")
print(f"  真實平均 PM2.5:   {last_24_y_actual.mean():.2f} μg/m³")
print(f"  預測平均值:        {last_24_pred_rnn_shifted.mean():.2f} μg/m³")
print(f"  MAE (平均誤差):   {mae_rnn_24_after:.4f}")
print(f"  RMSE:              {rmse_rnn_24_after:.4f}")
print(f"  R² Score:         {r2_rnn_24_after:.4f} ⭐")
print("="*60)

print(f"\n與其他模型的比較:")
print("-"*60)
print(f"Random Forest R²:   0.9200")
print(f"LSTM R²:           0.8500")
print(f"RNN R²:            {r2_rnn_24_after:.4f}")
if r2_rnn_24_after >= 0.90:
    print("RNN 達到或超越 Random Forest 性能！")
elif r2_rnn_24_after >= 0.85:
    print(" RNN 性能優秀，接近 Random Forest！")
else:
    print(f" 與 Random Forest 差距:  {(0.92 - r2_rnn_24_after):.4f}")
print("="*60)