import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import matplotlib.ticker as ticker
import matplotlib.font_manager as fm
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.inspection import permutation_importance

sns.set(style="whitegrid")
plt.rcParams['axes.unicode_minus'] = False
font_path = '/content/微軟正黑體-1.ttf'
try:
    my_font = fm.FontProperties(fname=font_path)
    print(f"成功載入字型: {font_path}")
except Exception as e:
    print(f"找不到字型檔案或載入失敗: {e}")
    print("將使用系統預設字型，中文可能會顯示為方框。")
    my_font = None

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
TIME_STEPS = 24

def load_and_prepare_data(filepath):
    print("正在讀取並預處理資料...")
    df = pd.read_csv(filepath)


    bool_cols = df.select_dtypes(include=['bool']).columns
    df[bool_cols] = df[bool_cols].astype(int)

    target = 'PM2.5_Value'
    X = df.drop(columns=[target])
    y = df[[target]]

    # 資料歸一化 (0~1)
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)

    return X_scaled, y_scaled, X.columns

# LSTM 專用：建立滑動視窗
def create_sequences(X, y, time_steps=TIME_STEPS):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

def split_data(X, y):
    print(f"正在轉換為 LSTM 序列格式 (Time Steps={TIME_STEPS})...")
    X_seq, y_seq = create_sequences(X, y, TIME_STEPS)

    X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)

    print(f"訓練集形狀: {X_train.shape}")
    print(f"測試集形狀: {X_test.shape}")
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    print("\n正在建構並訓練 LSTM 模型 (這需要一點時間)...")

    model = Sequential([
        # 第一層 LSTM
        LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.2),
        # 第二層 LSTM
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        # 輸出層
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')

    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        epochs=20,           
        batch_size=64,
        validation_split=0.1,
        callbacks=[early_stop],
        verbose=1
    )

    print("模型訓練完成")
    return model

def evaluate_model(model, X_test, y_test):
    print("\n正在評估模型...")

    # 預測 
    y_pred_scaled = model.predict(X_test)

    # 還原數值 
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    y_test_real = scaler_y.inverse_transform(y_test)

    mae = mean_absolute_error(y_test_real, y_pred)
    mse = mean_squared_error(y_test_real, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_real, y_pred)

    print("模型評估結果:")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R² Score: {r2:.4f}")

    # 回傳還原後的真實值與預測值
    return y_test_real, y_pred

# 測站誤差計算
def calculate_station_mae(X_test_seq, y_test_real, y_pred, feature_names):
    print("\n" + "="*30)
    print("各行政區 (測站) PM2.5 預測平均誤差 (MAE)")
    print("="*30)

    X_last_step = X_test_seq[:, -1, :]

    station_cols = [c for c in feature_names if c.startswith('Station_')]
    results = []

    for col_name in station_cols:
        col_idx = list(feature_names).index(col_name)

        mask = X_last_step[:, col_idx] > 0.5

        if mask.sum() > 0:
            local_y_test = y_test_real[mask]
            local_y_pred = y_pred[mask]
            mae = mean_absolute_error(local_y_test, local_y_pred)
            station_name = col_name.replace('Station_', '')
            results.append((station_name, mae))

    results.sort(key=lambda x: x[1])

    for name, mae in results:
        print(f"{name}: {mae:.4f}")
    print("="*30)

    names = [x[0] for x in results]
    values = [x[1] for x in results]

    plt.figure(figsize=(10, 6))
    plt.title("各行政區 PM2.5 預測誤差 (MAE)", fontproperties=my_font, fontsize=14)
    sns.barplot(x=values, y=names, color="steelblue")
    plt.xlabel("平均絕對誤差 (MAE)", fontproperties=my_font, fontsize=12)
    plt.ylabel("測站名稱", fontproperties=my_font, fontsize=12)
    plt.yticks(fontproperties=my_font)
    for i, v in enumerate(values):
        plt.text(v, i, f' {v:.4f}', va='center', fontweight='bold')
    plt.tight_layout()
    plt.show()

def plot_prediction_scatter(y_test_real, y_pred):
    plt.figure(figsize=(8, 8))
    plt.title("真實值 vs 預測值 (LSTM)", fontproperties=my_font, fontsize=14)
    plt.scatter(y_test_real, y_pred, alpha=0.3, color="steelblue", s=10)

    min_val = min(y_test_real.min(), y_pred.min())
    max_val = max(y_test_real.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)

    plt.xlabel("真實數值", fontproperties=my_font, fontsize=12)
    plt.ylabel("預測數值", fontproperties=my_font, fontsize=12)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    file_path = '/content/FINAL_MODEL_TRAINING_DATA.csv'

    try:
        # 1. 載入資料
        X_scaled, y_scaled, feature_names = load_and_prepare_data(file_path)

        # 2. 切分資料 (LSTM 專用格式)
        X_train, X_test, y_train, y_test = split_data(X_scaled, y_scaled)

        # 3. 訓練
        model = train_model(X_train, y_train)

        # 4. 評估與還原數值
        y_test_real, y_pred = evaluate_model(model, X_test, y_test)

        # 5. 畫圖 - 各測站誤差
        calculate_station_mae(X_test, y_test_real, y_pred, feature_names)

        # 6. 畫圖 - 真實 vs 預測
        plot_prediction_scatter(y_test_real, y_pred)

    except FileNotFoundError:
        print(f"找不到檔案: {file_path}。請確認路徑是否正確。")