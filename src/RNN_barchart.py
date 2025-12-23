import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
# [修改] 引入 Keras 與相關套件
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dropout, Dense
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import matplotlib.ticker as ticker
import matplotlib.font_manager as fm

sns.set(style="whitegrid")
plt.rcParams['axes.unicode_minus'] = False
font_path = '/content/微軟正黑體-1.ttf'
try:
    my_font = fm.FontProperties(fname=font_path)
    # 設定全域字體以確保標題正常顯示
    plt.rcParams['font.family'] = my_font.get_name()
    print(f"成功載入字型: {font_path}")
except Exception as e:
    print(f"找不到字型檔案或載入失敗: {e}")
    print("將使用系統預設字型，中文可能會顯示為方框。")
    my_font = None

def load_and_prepare_data(filepath):
    df = pd.read_csv(filepath)

    # [新增] RNN 需要數值輸入，將布林值 (True/False) 轉換為整數 (1/0)
    bool_cols = df.select_dtypes(include=['bool']).columns
    if len(bool_cols) > 0:
        df[bool_cols] = df[bool_cols].astype(int)

    target = 'PM2.5_Value'
    # 若有日期欄位建議移除，以免干擾訓練
    if 'Datetime' in df.columns:
        X = df.drop(columns=[target, 'Datetime'])
    else:
        X = df.drop(columns=[target])
    y = df[target]
    return X, y

def split_data(X, y):
    # RNN 處理時間序列通常建議 shuffle=False，但為了維持您原本的評估邏輯，這裡保持 random_state
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"訓練集大小: {X_train.shape}")
    print(f"測試集大小: {X_test.shape}")
    return X_train, X_test, y_train, y_test

# [修改] 訓練模型函式，改為建立 RNN
def train_model(X_train, y_train):
    print("\n正在處理資料並訓練 RNN 模型...")

    # 1. 數據標準化 (MinMaxScaler) - 神經網路必須步驟
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()

    # 轉換特徵
    X_train_scaled = scaler_x.fit_transform(X_train)
    # 轉換目標 (需要 reshape 為 2D array 才能 fit)
    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1))

    # 2. 轉換為 RNN 格式 (Samples, TimeSteps, Features)
    # 因為資料已經有 Lag Features，我們設 TimeSteps=1
    X_train_rnn = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))

    # 3. 建立 SimpleRNN 模型
    model = Sequential()
    # input_shape = (1, 特徵數量)
    model.add(SimpleRNN(units=256, input_shape=(1, X_train_scaled.shape[1]), unroll=True))
    model.add(Dropout(0.2)) # 防止過擬合
    model.add(Dense(units=1)) # 輸出層

    model.compile(loss="mse", optimizer="adam")

    # 4. 開始訓練
    # epochs=20, batch_size=200 為示範設定，可依需求調整
    print("開始迭代訓練 (Epochs)...")
    model.fit(X_train_rnn, y_train_scaled, batch_size=200, epochs=20, verbose=1, validation_split=0.1)

    print("模型訓練完成")
    # [重要] 為了後續預測，必須把 scaler 一起回傳
    return model, scaler_x, scaler_y

# [修改] 評估模型函式，加入反正規化步驟
def evaluate_model(model_pack, X_test, y_test):
    model, scaler_x, scaler_y = model_pack # 解包

    print("\n正在評估模型...")

    # 1. 對測試集做一樣的預處理
    X_test_scaled = scaler_x.transform(X_test)
    X_test_rnn = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

    # 2. 預測
    y_pred_scaled = model.predict(X_test_rnn)

    # 3. 將預測結果還原回真實數值 (Inverse Transform)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)

    # 轉為一維陣列以符合 metric 計算格式
    y_pred = y_pred.flatten()

    # 計算指標
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print("模型評估結果:")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R² Score: {r2:.4f}")
    return y_pred

def calculate_station_mae(X_test, y_test, y_pred):
    print("\n" + "="*30)
    print("各行政區 (測站) PM2.5 預測平均誤差 (MAE)")
    print("="*30)

    station_cols = [c for c in X_test.columns if c.startswith('Station_')]
    results = []

    for col in station_cols:
        mask = X_test[col] == 1
        if mask.sum() > 0:
            local_y_test = y_test[mask]
            # y_pred 是 numpy array，利用 boolean mask 篩選
            local_y_pred = y_pred[mask]

            mae = mean_absolute_error(local_y_test, local_y_pred)
            station_name = col.replace('Station_', '')
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

# [修改] 特徵重要性函式 - 增加對 RNN 的檢查
def plot_feature_importance(model_pack, feature_names, skip_first=False, log_scale=True):
    # RNN 模型被包在 tuple 裡，解包取出 model
    model = model_pack[0] if isinstance(model_pack, tuple) else model_pack

    # RNN 沒有 feature_importances_ 屬性
    if not hasattr(model, 'feature_importances_'):
        print("\n" + "="*30)
        print("提示：目前的模型 (RNN) 為神經網路黑盒模型，")
        print("無法直接像隨機森林一樣產出特徵重要性 (Feature Importance)。")
        print("若需分析特徵重要性，需使用 Permutation Importance 等進階方法。")
        print("本次執行將跳過特徵重要性繪圖。")
        print("="*30 + "\n")
        return

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    top_n = 10
    start_index = 1 if skip_first else 0
    current_indices = indices[start_index : start_index + top_n]

    plt.figure(figsize=(10, 6))

    title = "特徵重要性排名"
    if skip_first:
        title += " (排除第一名)"

    plt.title(title, fontproperties=my_font, fontsize=14)

    values = importances[current_indices]
    names = [feature_names[i] for i in current_indices]

    plt.barh(range(len(values)), values, align="center", color='steelblue')
    plt.yticks(range(len(values)), names, fontproperties=my_font)

    plt.xlabel("重要性分數", fontproperties=my_font, fontsize=12)
    plt.ylabel("特徵變數", fontproperties=my_font, fontsize=12)

    if log_scale:
        plt.xscale('log')
        plt.xlabel("重要性分數 (Log Scale)", fontproperties=my_font, fontsize=12)
        plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))

    plt.gca().invert_yaxis()

    for i, v in enumerate(values):
        padding = v * 0.05 if not log_scale else v * 0.1
        plt.text(v, i, f' {v:.4f}', va='center', fontweight='bold')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    file_path = '/content/FINAL_MODEL_TRAINING_DATA.csv'

    try:
        X, y = load_and_prepare_data(file_path)
        X_train, X_test, y_train, y_test = split_data(X, y)

        # 訓練 RNN 模型 (回傳包含 Scaler 的包)
        model_pack = train_model(X_train, y_train)

        # 評估模型 (傳入 model_pack)
        y_pred = evaluate_model(model_pack, X_test, y_test)

        # 繪製各區誤差圖 (格式不變)
        calculate_station_mae(X_test, y_test, y_pred)

        # 嘗試繪製特徵重要性 (會自動檢測 RNN 並跳過，避免報錯)
        plot_feature_importance(model_pack, X.columns, skip_first=False, log_scale=True)

    except FileNotFoundError:
        print(f"找不到檔案: {file_path}。請確認路徑是否正確。")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"發生錯誤: {e}")