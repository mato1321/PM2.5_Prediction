import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import matplotlib.ticker as ticker
import matplotlib.font_manager as fm

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

def load_and_prepare_data(filepath):
    df = pd.read_csv(filepath)
    target = 'PM2.5_Value'
    X = df.drop(columns=[target])
    y = df[target]
    return X, y

def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"訓練集大小: {X_train.shape}")
    print(f"測試集大小: {X_test.shape}")
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    print("\n正在訓練隨機森林模型...")
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    print("模型訓練完成")
    return rf_model

def evaluate_model(model, X_test, y_test):
    print("\n正在評估模型...")
    y_pred = model.predict(X_test)
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
            local_y_pred = y_pred[mask.values]
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

def plot_feature_importance(model, feature_names, skip_first=False, log_scale=True):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    top_n = 10
    start_index = 1 if skip_first else 0
    current_indices = indices[start_index : start_index + top_n]

    plt.figure(figsize=(10, 6))

    title = "特徵重要性排名"
    if skip_first:
        title += " (排除第一名)"
    if log_scale:
        title += ""

    plt.title(title, fontproperties=my_font, fontsize=14)

    values = importances[current_indices]
    names = [feature_names[i] for i in current_indices]

    plt.barh(range(len(values)), values, align="center", color='steelblue')

    plt.yticks(range(len(values)), names, fontproperties=my_font)

    plt.xlabel("重要性分數", fontproperties=my_font, fontsize=12)
    plt.ylabel("特徵變數", fontproperties=my_font, fontsize=12)

    if log_scale:
        plt.xscale('log')
        plt.xlabel("重要性分數 ", fontproperties=my_font, fontsize=12)
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
        model = train_model(X_train, y_train)
        y_pred = evaluate_model(model, X_test, y_test)

        calculate_station_mae(X_test, y_test, y_pred)
        plot_feature_importance(model, X.columns, skip_first=False, log_scale=True)

    except FileNotFoundError:
        print(f"找不到檔案: {file_path}。請確認路徑是否正確。")