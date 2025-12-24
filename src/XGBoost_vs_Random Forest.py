import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.font_manager as fm

df = pd.read_csv('FINAL_MODEL_TRAINING_DATA.csv')

bool_cols = df.select_dtypes(include=['bool']).columns
df[bool_cols] = df[bool_cols].astype(int)

X = df.drop(columns=['PM2.5_Value'])
y = df['PM2.5_Value']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

#訓練模型 

# 模型 A: XGBoost
print("正在訓練 XGBoost...")
model_xgb = xgb.XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6, n_jobs=-1, random_state=42)
model_xgb.fit(X_train, y_train)
pred_xgb = model_xgb.predict(X_test)

# 模型 B: Random Forest
print("正在訓練 Random Forest (這會比較久)...")
model_rf = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42) # 為了速度先設100顆樹
model_rf.fit(X_train, y_train)
pred_rf = model_rf.predict(X_test)

print("訓練完成！")

# 繪圖設定 
font_path = '/content/微軟正黑體-1.ttf'
try:
    fm.fontManager.addfont(font_path)
    plt.rcParams['font.family'] = fm.FontProperties(fname=font_path).get_name()
except:
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial']
plt.rcParams["axes.unicode_minus"] = False

# 繪製對比圖 (XGB vs RF vs 真實)

days_to_show = 10
points_per_day = 24
show_num = days_to_show * points_per_day

# 取出最後一段資料
y_plot = y_test.values[-show_num:]
pred_plot_xgb = pred_xgb[-show_num:]
pred_plot_rf = pred_rf[-show_num:]

plot_start_date = '2024-12-20'
custom_dates = pd.date_range(start=plot_start_date, periods=show_num, freq='h')

plt.figure(figsize=(15, 7))

# A. 真實值 (紅色實線)
plt.plot(range(show_num), y_plot, color='red', label='真實 PM2.5', linewidth=2.5, alpha=0.6)

# B. XGBoost (藍色虛線 )
plt.plot(range(show_num), pred_plot_xgb, color='blue', linestyle='--', label='XGBoost', linewidth=2, alpha=0.8)

# C. Random Forest (綠色點線 )
plt.plot(range(show_num), pred_plot_rf, color='green', linestyle=':', label='Random Forest', linewidth=2, alpha=0.9)


ticks = range(0, show_num, points_per_day)
labels = custom_dates.strftime('%Y-%m-%d').to_series().iloc[::points_per_day]
plt.xticks(ticks=ticks, labels=labels, fontsize=11, rotation=15)

plt.grid(axis='x', linestyle='-', alpha=0.5)
plt.grid(axis='y', linestyle='--', alpha=0.3)
plt.title(f'XGBoost vs Random Forest ({labels.iloc[0]} ~ {labels.iloc[-1]})', fontsize=16)
plt.ylabel('PM2.5 濃度', fontsize=12)
plt.legend(fontsize=12, loc='upper left')
plt.tight_layout()
plt.show()

# 數據裁判 

print("\n" + "="*40)
print(f"{'模型':<15} | {'R2 Score':<10} | {'MAE (平均誤差)':<15}")
print("-" * 45)

# 計算局部指標
r2_xgb_local = r2_score(y_plot, pred_plot_xgb)
mae_xgb_local = mean_absolute_error(y_plot, pred_plot_xgb)

r2_rf_local = r2_score(y_plot, pred_plot_rf)
mae_rf_local = mean_absolute_error(y_plot, pred_plot_rf)

print(f"{'XGBoost':<15} | {r2_xgb_local:<10.4f} | {mae_xgb_local:<15.4f}")
print(f"{'Random Forest':<15} | {r2_rf_local:<10.4f} | {mae_rf_local:<15.4f}")
print("="*40)
