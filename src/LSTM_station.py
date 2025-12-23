import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.font_manager as fm
import matplotlib as mpl
import os

# --- 1. 字體設定 (保證中文顯示) ---
font_path = "/content/微軟正黑體-1.ttf"

if not os.path.exists(font_path):
    raise FileNotFoundError("❌ 找不到字體檔：微軟正黑體-1.ttf")

# 載入字體
fm.fontManager.addfont(font_path)
font_prop = fm.FontProperties(fname=font_path)

# 設定全域字體
mpl.rcParams['font.family'] = font_prop.get_name()
mpl.rcParams['axes.unicode_minus'] = False


# --- 2. 資料讀取與特徵工程 ---
print("正在讀取並進行特徵工程 (Feature Engineering)...")
df_pm25 = pd.read_csv("ALL_YEARS_PM25_TARGET_AND_LAG_FEATURES.csv")
df_meteo = pd.read_csv("ALL_YEARS_METEO_STANDARDIZED (1).csv")

# 合併
df = pd.merge(df_pm25, df_meteo, on=['測站', '日期', '小時'], how='inner')

# 建立 Datetime
df['Datetime'] = pd.to_datetime(df['日期'] + ' ' + df['小時'].astype(str) + ':00:00')
df = df.sort_values(['測站', 'Datetime'])

# [改進 1] 加入時間週期特徵 (讓模型懂 上班時間 vs 半夜)
df['Hour_Sin'] = np.sin(2 * np.pi * df['小時'] / 24)
df['Hour_Cos'] = np.cos(2 * np.pi * df['小時'] / 24)

# 風向向量化
df['Wind_Sin'] = np.sin(df['WIND_DIREC'] * (np.pi / 180))
df['Wind_Cos'] = np.cos(df['WIND_DIREC'] * (np.pi / 180))

# 定義特徵 (加入 Hour_Sin/Cos)
feature_cols = ['PM25_Lag_1h', 'PM25_Lag_2h', 'PM25_Lag_24h', # 確保 Lag_24h 也在
                'RAINFALL', 'WIND_SPEED', 'RH', 'AMB_TEMP',
                'Wind_Sin', 'Wind_Cos', 'Hour_Sin', 'Hour_Cos']
target_col = 'PM2.5_Value'

# --- 3. 設定預測區間 ---
target_start_date = '2024-12-20'
target_end_date = '2024-12-29'
stations = df['測站'].unique()

print(f"預測目標: {target_start_date} ~ {target_end_date}")
print(f"使用特徵數: {len(feature_cols)} 個")

# 序列製作函式
def create_sequences(X, y, time_steps):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

# --- 4. 迴圈訓練 ---
for station in stations:
    print(f"\n⚡ 正在強化訓練測站：{station} ...")

    df_station = df[df['測站'] == station].copy()

    # 切分訓練與測試
    train_data = df_station[df_station['Datetime'] < pd.Timestamp(target_start_date)]

    # [改進 2] 延長 Lookback 到 48 小時 (兩天)
    lookback_hours = 48

    # 測試集準備
    lookback_start = pd.Timestamp(target_start_date) - pd.Timedelta(hours=lookback_hours)
    end_time = pd.Timestamp(target_end_date) + pd.Timedelta(hours=23)
    test_data = df_station[(df_station['Datetime'] >= lookback_start) &
                           (df_station['Datetime'] <= end_time)]

    if len(test_data) < lookback_hours:
        print(f"⚠️ {station} 資料不足，跳過。")
        continue

    # 正規化
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_train = scaler_x.fit_transform(train_data[feature_cols])
    y_train = scaler_y.fit_transform(train_data[[target_col]])
    X_test = scaler_x.transform(test_data[feature_cols])
    y_test = scaler_y.transform(test_data[[target_col]])

    # 製作序列
    X_train_seq, y_train_seq = create_sequences(X_train, y_train, lookback_hours)
    X_test_seq, y_test_seq = create_sequences(X_test, y_test, lookback_hours)

    if len(X_test_seq) == 0: continue

    # [改進 3] 建立雙層 LSTM 模型 (Stacked LSTM)
    model = Sequential([
        Input(shape=(lookback_hours, len(feature_cols))),

        # 第一層：128 神經元，return_sequences=True 代表要傳給下一層
        LSTM(128, return_sequences=True),
        Dropout(0.3), # 稍微提高 Dropout 防止過擬合

        # 第二層：64 神經元，不再回傳序列
        LSTM(64, return_sequences=False),
        Dropout(0.3),

        # [改進 4] 加入 Dense 層進行特徵整合
        Dense(32, activation='relu'),
        Dense(1)
    ])

    # 使用 Adam 優化器
    model.compile(optimizer='adam', loss='mse')

    # 設定 Callbacks (自動調整學習率 + 早停)
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=0.0001)

    # 訓練 (Epochs 設為 50，Batch Size 32)
    model.fit(X_train_seq, y_train_seq,
              epochs=50, batch_size=32,
              validation_split=0.1,
              callbacks=[early_stop, reduce_lr],
              verbose=0)

    # 預測
    pred_scaled = model.predict(X_test_seq, verbose=0)
    pred_val = scaler_y.inverse_transform(pred_scaled)
    actual_val = scaler_y.inverse_transform(y_test_seq)

    # 計算日平均
    predict_dates = test_data['Datetime'].iloc[lookback_hours:]
    result_df = pd.DataFrame({
        'Datetime': predict_dates,
        'Actual': actual_val.flatten(),
        'Predicted': pred_val.flatten()
    })
    result_df.set_index('Datetime', inplace=True)
    daily_avg = result_df.resample('D').mean()

    # --- 繪圖 (微調樣式) ---
    plt.figure(figsize=(10, 5))
    x_indices = range(len(daily_avg))
    date_labels = daily_avg.index.strftime('%m/%d')

    # 真實值 (紅色實線，加粗)
    plt.plot(x_indices, daily_avg['Actual'],
             color='#d62728', marker='o', markersize=8, linewidth=3, label='真實 PM2.5', alpha=0.8)

    # 預測值 (藍色虛線，加粗)
    plt.plot(x_indices, daily_avg['Predicted'],
             color='#1f77b4', marker='s', markersize=8, linewidth=3, linestyle='--', label='LSTM 預測', alpha=0.9)

    # 標示數值
    for i, (y_real, y_pred) in enumerate(zip(daily_avg['Actual'], daily_avg['Predicted'])):
        # 顯示差距
        plt.text(i, y_pred + 0.8, f'{y_pred:.1f}', ha='center', color='#1f77b4', fontsize=10, fontweight='bold')

    plt.title(f'{station} PM2.5 預測 ({target_start_date} ~ {target_end_date})', fontsize=15)
    plt.xlabel('日期', fontsize=12)
    plt.ylabel('PM2.5 濃度', fontsize=12)
    plt.xticks(x_indices, date_labels)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.show()

    print(f"✅ {station} 完成！")