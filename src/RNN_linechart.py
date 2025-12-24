import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dropout, Dense
import matplotlib.font_manager as fm

plt.rcParams['axes.unicode_minus'] = False
font_path = '/content/微軟正黑體-1.ttf'

filename = 'FINAL_MODEL_TRAINING_DATA.csv'
df = pd.read_csv(filename)

bool_cols = df.select_dtypes(include=['bool']).columns
df[bool_cols] = df[bool_cols].astype(int)

target_col = 'PM2.5_Value'
features = df.drop(columns=[target_col])
target = df[[target_col]]
print(features)

# 鎖定最後 10 天 
days_to_predict = 10
hours_to_predict = days_to_predict * 24  # 240 筆

# 測試集：取檔案最後240 筆
X_test_raw = features.iloc[-hours_to_predict:]
y_test_raw = target.iloc[-hours_to_predict:]

# 訓練集：取剩下的前面所有資料
X_train_raw = features.iloc[:-hours_to_predict]
y_train_raw = target.iloc[:-hours_to_predict]

print(f"訓練資料筆數: {len(X_train_raw)}")
print(f"預測資料筆數: {len(X_test_raw)} (檔案尾端)")

# 資料正規化
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()

X_train_scaled = scaler_x.fit_transform(X_train_raw)
y_train_scaled = scaler_y.fit_transform(y_train_raw)

X_test_scaled = scaler_x.transform(X_test_raw)
y_test_scaled = scaler_y.transform(y_test_raw)

# 轉換為 RNN 格式 
X_train_rnn = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_test_rnn  = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

# 建立 SimpleRNN 模型
model = Sequential()
model.add(SimpleRNN(units=256, input_shape=(1, X_train_rnn.shape[2]), unroll=False))
model.add(Dropout(0.2))
model.add(Dense(units=1))
model.compile(loss="mse", optimizer="adam")

print(":rocket: 開始訓練 RNN...")
model.fit(X_train_rnn, y_train_scaled, batch_size=200, epochs=20, validation_split=0.1, verbose=1)

# 預測與還原 
predict_scaled = model.predict(X_test_rnn)
predict_y = scaler_y.inverse_transform(predict_scaled) 
real_y    = scaler_y.inverse_transform(y_test_scaled)

# 繪圖
plt.figure(figsize=(14, 6))


plot_dates = pd.date_range(start='2024-12-20', periods=len(real_y), freq='h')
date_labels = plot_dates.strftime('%m/%d')


plt.plot(plot_dates, real_y, color='red', label='真實 PM2.5', linewidth=2, alpha=0.7)
plt.plot(plot_dates, predict_y, color='blue', linestyle='--', label='預測結果(RNN)', linewidth=2, alpha=0.9)


ticks_step = 24
plt.xticks(plot_dates[::ticks_step], date_labels[::ticks_step], rotation=0, fontsize=11)

plt.grid(axis='x', linestyle='-', alpha=0.5)
plt.grid(axis='y', linestyle='--', alpha=0.3)
plt.title('PM2.5 RNN預測結果對比(2024/12/20 - 2024/12/29)', fontsize=16)
plt.ylabel('PM2.5 濃度', fontsize=12)
plt.xlabel('日期', fontsize=12)
plt.legend(fontsize=12, loc='upper left')
plt.tight_layout()
plt.show()

r2_subset = r2_score(y_plot, pred_plot)
print(f"這段期間 ({labels.iloc[0]} ~ {labels.iloc[-1]}) 的準確度 (R2): {r2_subset:.2f}")