import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.font_manager as fm

font_path = '/content/微軟正黑體-1.ttf'

df = pd.read_csv('FINAL_MODEL_TRAINING_DATA.csv')
bool_cols = df.select_dtypes(include=['bool']).columns
df[bool_cols] = df[bool_cols].astype(int)

target_col = 'PM2.5_Value'
features = df.drop(columns=[target_col])
target = df[[target_col]]

# 正規化
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_scaled = scaler_X.fit_transform(features)
y_scaled = scaler_y.fit_transform(target)

# 切分資料
split = int(len(X_scaled) * 0.8)
X_train_raw = X_scaled[:split]
X_test_raw = X_scaled[split:]
y_train = y_scaled[:split]
y_test = y_scaled[split:]

# 轉換為 3D (Samples, 1, Features) 
X_train_dl = X_train_raw.reshape((X_train_raw.shape[0], 1, X_train_raw.shape[1]))
X_test_dl = X_test_raw.reshape((X_test_raw.shape[0], 1, X_test_raw.shape[1]))

# 建立模型 
print("正在訓練 LSTM 模型...")
model = Sequential([
    LSTM(50, return_sequences=False, input_shape=(1, X_train_raw.shape[1])),
    Dropout(0.2), # 防止過擬合
    Dense(1)      # 輸出層
])

model.compile(loss='mse', optimizer='adam')
# 增加 epochs 讓線貼得更近
model.fit(X_train_dl, y_train, epochs=50, batch_size=64, verbose=1, validation_split=0.1)

#預測與還原 
pred_scaled = model.predict(X_test_dl)
pred_inverse = scaler_y.inverse_transform(pred_scaled)
y_test_inverse = scaler_y.inverse_transform(y_test)

# 繪製藍色虛線圖
days_to_show = 10
points_per_day = 24
show_num = days_to_show * points_per_day

y_plot = y_test_inverse[-show_num:]
pred_plot = pred_inverse[-show_num:]

plot_start_date = '2024-12-20'
custom_dates = pd.date_range(start=plot_start_date, periods=show_num, freq='h')

plt.figure(figsize=(14, 6))

# A. 真實 PM2.5 (紅色實線)
plt.plot(range(show_num), y_plot,
         color='red',           
         linestyle='-',      
         label='真實 PM2.5',
         linewidth=2,
         alpha=0.7)

# B. LSTM 預測 (藍色虛線)
plt.plot(range(show_num), pred_plot,
         color='blue',         
         linestyle='--',      
         label='LSTM 預測值',
         linewidth=2,
         alpha=0.9)

# 設定 X 軸
ticks = range(0, show_num, points_per_day)
labels = custom_dates.strftime('%m/%d').to_series().iloc[::points_per_day]
plt.xticks(ticks=ticks, labels=labels, fontsize=11)

plt.grid(axis='x', linestyle='-', alpha=0.3)
plt.grid(axis='y', linestyle='--', alpha=0.3)
plt.title(f'PM2.5 LSTM 預測結果對比 (2024/{labels.iloc[0]} ~ 2024/{labels.iloc[-1]})', fontsize=16)
plt.ylabel('PM2.5 濃度', fontsize=12)
plt.xlabel('日期', fontsize=12)
plt.legend(fontsize=12, loc='upper left')

plt.tight_layout()
plt.show()