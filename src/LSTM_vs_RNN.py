import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, SimpleRNN, Dense, Dropout
import matplotlib.font_manager as fm

# 1. 讀取資料
df = pd.read_csv('FINAL_MODEL_TRAINING_DATA.csv')

# 處理布林值
bool_cols = df.select_dtypes(include=['bool']).columns
df[bool_cols] = df[bool_cols].astype(int)

# 定義 X 和 y
X = df.drop(columns=['PM2.5_Value']).values # 轉為 numpy array
y = df['PM2.5_Value'].values.reshape(-1, 1) # y 也要轉為二維矩陣

# --- [關鍵步驟] 資料正規化 (Scaling) ---
# 神經網路必須將數值縮放到 0~1 之間
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_x.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# 切分資料 (shuffle=False, 保持時間順序)
# 因為我們等等要 Reshape，所以先切分 Index 比較保險
split_idx = int(len(X_scaled) * 0.8)
X_train = X_scaled[:split_idx]
X_test = X_scaled[split_idx:]
y_train = y_scaled[:split_idx]
y_test = y_scaled[split_idx:]

X_train_dl = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test_dl = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

print(f"訓練資料形狀: {X_train_dl.shape}")

# --- 2. 訓練模型---

# 設定參數
epochs = 20
batch_size = 64

# 模型 A: LSTM (長短期記憶網路)
print("\n正在訓練 LSTM 模型...")
model_lstm = Sequential()
model_lstm.add(LSTM(units=50, input_shape=(1, X_train.shape[1]), return_sequences=False))
model_lstm.add(Dropout(0.2)) # 防止過擬合
model_lstm.add(Dense(1))     # 輸出層
model_lstm.compile(loss='mse', optimizer='adam')
model_lstm.fit(X_train_dl, y_train, epochs=epochs, batch_size=batch_size, verbose=1, validation_split=0.1)

# 模型 B: RNN (傳統遞歸神經網路)
print("\n正在訓練 RNN 模型...")
model_rnn = Sequential()
model_rnn.add(SimpleRNN(units=50, input_shape=(1, X_train.shape[1]), return_sequences=False))
model_rnn.add(Dropout(0.2))
model_rnn.add(Dense(1))
model_rnn.compile(loss='mse', optimizer='adam')
model_rnn.fit(X_train_dl, y_train, epochs=epochs, batch_size=batch_size, verbose=1, validation_split=0.1)

# --- 3. 預測與還原數值 ---
print("\n進行預測並還原數值...")
# 預測 (結果是 0~1)
pred_lstm_scaled = model_lstm.predict(X_test_dl)
pred_rnn_scaled = model_rnn.predict(X_test_dl)

# 還原 (Inverse Transform) 回真實 PM2.5 濃度
pred_lstm = scaler_y.inverse_transform(pred_lstm_scaled)
pred_rnn = scaler_y.inverse_transform(pred_rnn_scaled)
y_test_real = scaler_y.inverse_transform(y_test) # 真實值也要還原

# --- 4. 繪圖設定 ---
font_path = '/content/微軟正黑體-1.ttf'
try:
    fm.fontManager.addfont(font_path)
    plt.rcParams['font.family'] = fm.FontProperties(fname=font_path).get_name()
except:
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial']
plt.rcParams["axes.unicode_minus"] = False

# --- 5. 繪製對比圖 (LSTM vs RNN vs 真實) ---

days_to_show = 10
points_per_day = 24
show_num = days_to_show * points_per_day

# 取出最後一段資料
# 注意：這邊的變數已經還原成 numpy array 了
y_plot = y_test_real[-show_num:]
pred_plot_lstm = pred_lstm[-show_num:]
pred_plot_rnn = pred_rnn[-show_num:]

plot_start_date = '2024-12-20'
custom_dates = pd.date_range(start=plot_start_date, periods=show_num, freq='h')

plt.figure(figsize=(15, 7))

# A. 真實值 (紅色實線)
plt.plot(range(show_num), y_plot, color='red', label='真實 PM2.5', linewidth=2.5, alpha=0.6)

# B. LSTM (藍色虛線 - 記憶力強)
plt.plot(range(show_num), pred_plot_lstm, color='blue', linestyle='--', label='LSTM', linewidth=2, alpha=0.8)

# C. RNN (綠色點線 - 結構簡單)
plt.plot(range(show_num), pred_plot_rnn, color='green', linestyle=':', label='RNN', linewidth=2, alpha=0.9)

# 設定軸標籤
ticks = range(0, show_num, points_per_day)
labels = custom_dates.strftime('%Y-%m-%d').to_series().iloc[::points_per_day]
plt.xticks(ticks=ticks, labels=labels, fontsize=11, rotation=15)

plt.grid(axis='x', linestyle='-', alpha=0.5)
plt.grid(axis='y', linestyle='--', alpha=0.3)
plt.title(f'LSTM vs RNN ({labels.iloc[0]} ~ {labels.iloc[-1]})', fontsize=16)
plt.ylabel('PM2.5 濃度', fontsize=12)
plt.legend(fontsize=12, loc='upper left')
plt.tight_layout()
plt.show()

# --- 6. 數據裁判 (到底誰準?) ---

print("\n" + "="*40)
print(f"{'模型':<15} | {'R2 Score':<10} | {'MAE (平均誤差)':<15}")
print("-" * 45)

# 計算局部指標
r2_lstm_local = r2_score(y_plot, pred_plot_lstm)
mae_lstm_local = mean_absolute_error(y_plot, pred_plot_lstm)

r2_rnn_local = r2_score(y_plot, pred_plot_rnn)
mae_rnn_local = mean_absolute_error(y_plot, pred_plot_rnn)

print(f"{'LSTM':<15} | {r2_lstm_local:<10.4f} | {mae_lstm_local:<15.4f}")
print(f"{'RNN':<15} | {r2_rnn_local:<10.4f} | {mae_rnn_local:<15.4f}")
print("="*40)