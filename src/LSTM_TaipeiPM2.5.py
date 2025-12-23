import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

tf.random.set_seed(42)
np.random.seed(42)

# 格式：[Jan, May, Sep] 循環
raw_data_list = []
base_jan = 16.5
base_may = 13.5
base_sep = 12.0

# 模擬 7 年數據 (2019-2025)
for year in range(7):
    # 每年加入一點隨機波動與逐年改善 (-0.2/年)
    decay = year * 0.2
    noise = np.random.uniform(-0.3, 0.3)

    raw_data_list.append(base_jan - decay + noise) # Jan
    raw_data_list.append(base_may - decay + noise) # May
    raw_data_list.append(base_sep - decay + noise) # Sep

raw_data = np.array(raw_data_list)

# 數據正規化
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(raw_data.reshape(-1, 1))

# --- 3. 調整 LSTM 資料集 ---
# 因為現在一年有 3 個點，我們將 look_back 設為 3
# 意思是用「過去完整的一年 (Jan, May, Sep)」來預測下一個時間點
look_back = 3

X, y = [], []
for i in range(len(scaled_data) - look_back):
    X.append(scaled_data[i:(i + look_back), 0])
    y.append(scaled_data[i + look_back, 0])
X, y = np.array(X), np.array(y)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# --- 4. 訓練 LSTM 模型 ---
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(look_back, 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# 訓練 (Silent mode)
model.fit(X, y, epochs=250, verbose=0)

# --- 5. 執行預測 (2026-2028) ---
# 目標時間點：
# 2026: Jan, May, Sep
# 2027: Jan, May, Sep
# 2028: Jan, May, Sep
future_steps = 9
predictions = []

current_batch = scaled_data[-look_back:].reshape((1, look_back, 1))

for i in range(future_steps):
    pred = model.predict(current_batch, verbose=0)[0]
    predictions.append(pred)
    current_batch = np.append(current_batch[:, 1:, :], [[pred]], axis=1)

predicted_values = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

# --- 6. 建立繪圖資料 ---
date_strings = [
    '2026-01-01', '2026-05-01', '2026-09-01',
    '2027-01-01', '2027-05-01', '2027-09-01',
    '2028-01-01', '2028-05-01', '2028-09-01'
]
target_dates = pd.to_datetime(date_strings)

df_pred = pd.DataFrame({
    'Date': target_dates,
    'PM2.5': predicted_values.flatten()
})

# --- 7. 繪製精美折線圖 ---
plt.figure(figsize=(11, 6))
plt.plot(df_pred['Date'], df_pred['PM2.5'], marker='o', linestyle='-',
         color='#2ca02c', linewidth=2, markersize=8, label='LSTM Forecast (Trimester)')

# 設定標題與軸標籤
plt.title('Taipei PM2.5 Forecast', fontsize=15, fontweight='bold')
plt.ylabel('PM2.5 (µg/m³)', fontsize=12)
plt.xlabel('Date', fontsize=12)

# 設定 Y 軸範圍 (讓波動更明顯)
plt.ylim(10, 16)
plt.grid(True, linestyle='--', alpha=0.6)

# 設定 X 軸刻度 (確保顯示出 Jan, May, Sep)
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 5, 9]))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.xticks(rotation=45) # 日期傾斜以免重疊

# 標註數值
for i, val in enumerate(df_pred['PM2.5']):
    # 稍微錯開文字位置，避免遮擋
    offset = 10 if i % 2 == 0 else -15
    plt.annotate(f'{val:.2f}',
                 (df_pred['Date'][i], val),
                 xytext=(0, offset), textcoords='offset points',
                 ha='center', fontsize=9, fontweight='bold',
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

plt.tight_layout()
plt.show()

# 輸出表格
print(df_pred)