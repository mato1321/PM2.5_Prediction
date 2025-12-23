import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dropout, Dense
import matplotlib.font_manager as fm

# --- 1. 設定中文字體 ---
plt.rcParams['axes.unicode_minus'] = False
font_path = '/content/微軟正黑體-1.ttf'

# --- 2. 讀取資料 ---
filename = 'FINAL_MODEL_TRAINING_DATA.csv'
df = pd.read_csv(filename)

# 處理布林值
bool_cols = df.select_dtypes(include=['bool']).columns
df[bool_cols] = df[bool_cols].astype(int)

# --- 3. 準備資料 ---
# 為了跟 XGBoost 完全同步，我們不看日期欄位，直接看「資料位置」
target_col = 'PM2.5_Value'
features = df.drop(columns=[target_col])
target = df[[target_col]]
print(features)

# --- 4. [關鍵修正] 強制鎖定最後 10 天 (240小時) ---
# 這樣做能保證跟 XGBoost 的測試集是完全同一批數據
days_to_predict = 10
hours_to_predict = days_to_predict * 24  # 240 筆

# 測試集：取檔案「最後」240 筆
X_test_raw = features.iloc[-hours_to_predict:]
y_test_raw = target.iloc[-hours_to_predict:]

# 訓練集：取剩下的前面所有資料
X_train_raw = features.iloc[:-hours_to_predict]
y_train_raw = target.iloc[:-hours_to_predict]

print(f"訓練資料筆數: {len(X_train_raw)}")
print(f"預測資料筆數: {len(X_test_raw)} (檔案尾端)")

# --- 5. 資料正規化 (RNN 必須做) ---
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()

# fit 只在訓練集上做
X_train_scaled = scaler_x.fit_transform(X_train_raw)
y_train_scaled = scaler_y.fit_transform(y_train_raw)

# 測試集用同樣的 scaler 轉換
X_test_scaled = scaler_x.transform(X_test_raw)
y_test_scaled = scaler_y.transform(y_test_raw)

# --- 6. 轉換為 RNN 格式 ---
# Reshape to (Samples, TimeSteps=1, Features)
X_train_rnn = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_test_rnn  = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

# --- 7. 建立 SimpleRNN 模型 ---
model = Sequential()
model.add(SimpleRNN(units=256, input_shape=(1, X_train_rnn.shape[2]), unroll=False))
model.add(Dropout(0.2))
model.add(Dense(units=1))
model.compile(loss="mse", optimizer="adam")

print(":rocket: 開始訓練 RNN...")
model.fit(X_train_rnn, y_train_scaled, batch_size=200, epochs=20, validation_split=0.1, verbose=1)

# --- 8. 預測與還原 ---
predict_scaled = model.predict(X_test_rnn)
predict_y = scaler_y.inverse_transform(predict_scaled) # RNN 預測結果
real_y    = scaler_y.inverse_transform(y_test_scaled)  # 真實結果

# --- 9. 繪圖 (座標與紅線與 XGBoost 完全一致) ---
plt.figure(figsize=(14, 6))

# 手動建立 X 軸日期：2024/12/20 ~ 12/29
# 這是為了讓圖表顯示正確的日期標籤
plot_dates = pd.date_range(start='2024-12-20', periods=len(real_y), freq='h')
date_labels = plot_dates.strftime('%m/%d')

# 畫線：顏色設定與之前一樣
plt.plot(plot_dates, real_y, color='red', label='真實 PM2.5', linewidth=2, alpha=0.7)
plt.plot(plot_dates, predict_y, color='blue', linestyle='--', label='預測結果(RNN)', linewidth=2, alpha=0.9)

# 設定 X 軸刻度 (一天一格)
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