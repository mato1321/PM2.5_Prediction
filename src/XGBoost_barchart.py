import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from matplotlib.ticker import ScalarFormatter

sns.set(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
plt.rcParams['axes.unicode_minus'] = False 

df = pd.read_csv('FINAL_MODEL_TRAINING_DATA.csv')

target_col = 'PM2.5_Value'
X = df.drop(columns=[target_col])
y = df[target_col]

# 切分訓練集與測試集 (80% 訓練, 20% 測試)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 訓練模型 
print("正在訓練模型 (Random Forest)...")
# 使用 RandomForestRegressor 
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# 計算各測站的 MAE 
station_cols = [col for col in df.columns if col.startswith('Station_')]
mae_results = []
y_pred = model.predict(X_test)

for station_col in station_cols:
    mask = X_test[station_col] == True
    if mask.sum() > 0:
        y_true_station = y_test[mask]
        y_pred_station = y_pred[mask]
        mae = mean_absolute_error(y_true_station, y_pred_station)
        station_name = station_col.replace('Station_', '') 
        mae_results.append({'Station': station_name, 'MAE': mae})

# 轉為 DataFrame 並排序
df_mae = pd.DataFrame(mae_results).sort_values('MAE')

#計算特徵重要性
importances = model.feature_importances_
feature_names = X.columns
df_importance = pd.DataFrame({'Feature': feature_names, 'Score': importances})
# 取前 10 名並排序
df_importance = df_importance.sort_values('Score', ascending=False).head(10)

# 繪製圖表 
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 各行政區 PM2.5 預測誤差 (MAE)
sns.barplot(x='MAE', y='Station', data=df_mae, ax=axes[0], color='steelblue')
axes[0].set_title('各行政區 PM2.5 預測誤差 (MAE)', fontsize=14)
axes[0].set_xlabel('平均絕對誤差 (MAE)', fontsize=12)
axes[0].set_ylabel('測站名稱', fontsize=12)

# 標註數值
for index, row in df_mae.reset_index(drop=True).iterrows():
    axes[0].text(row['MAE'] + 0.05, index, f"{row['MAE']:.4f}", 
                 va='center', fontsize=10, fontweight='bold', color='black')

# 特徵重要性排名
sns.barplot(x='Score', y='Feature', data=df_importance, ax=axes[1], color='steelblue')
axes[1].set_xscale('log') 
axes[1].set_xlim(0.01, 1) 
axes[1].set_title('特徵重要性排名', fontsize=14)
axes[1].set_xlabel('重要性分數', fontsize=12)
axes[1].set_ylabel('特徵變數', fontsize=12)

axes[1].set_xticks([0.01, 0.1, 1])
axes[1].get_xaxis().set_major_formatter(ScalarFormatter())


for index, row in df_importance.reset_index(drop=True).iterrows():
    axes[1].text(row['Score'] * 1.1, index, f"{row['Score']:.4f}", 
                 va='center', fontsize=10, fontweight='bold', color='black')

plt.tight_layout()
plt.show()

print("圖表繪製完成！")