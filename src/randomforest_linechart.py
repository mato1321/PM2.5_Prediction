import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.font_manager as fm

df = pd.read_csv('FINAL_MODEL_TRAINING_DATA.csv')
bool_cols = df.select_dtypes(include=['bool']).columns
df[bool_cols] = df[bool_cols].astype(int)
X = df.drop(columns=['PM2.5_Value'])
y = df['PM2.5_Value']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
print("訓練隨機森林模型")
model = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42)
model.fit(X_train, y_train)
print("訓練完成")

predict = model.predict(X_test)
font_path = '/content/微軟正黑體-1.ttf'
try:
    fm.fontManager.addfont(font_path)
    plt.rcParams['font.family'] = fm.FontProperties(fname=font_path).get_name()
except:
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams["axes.unicode_minus"] = False
days_to_show = 10           # 顯示 10 天
points_per_day = 24
show_num = days_to_show * points_per_day
y_plot = y_test.values[-show_num:]
pred_plot = predict[-show_num:]
plot_start_date = '2024-12-20'
custom_dates = pd.date_range(start=plot_start_date, periods=show_num, freq='h')
plt.figure(figsize=(14, 6))

# 畫線
plt.plot(range(show_num), y_plot, color='red', label='真實 PM2.5', linewidth=2, alpha=0.7)
plt.plot(range(show_num), pred_plot, color='blue', linestyle='--', label='預測結果 (Random Forest)', linewidth=2, alpha=0.8)
# 設定 X 軸刻度
ticks = range(0, show_num, points_per_day) # 每 24 小時一格
# 使用自訂日期來標示
labels = custom_dates.strftime('%Y-%m-%d').to_series().iloc[::points_per_day]
plt.xticks(ticks=ticks, labels=labels, fontsize=11, rotation=15)
plt.grid(axis='x', linestyle='-', alpha=0.5)
plt.grid(axis='y', linestyle='--', alpha=0.3)
plt.title(f'PM2.5 隨機森林預測結果對比 ({labels.iloc[0]} ~ {labels.iloc[-1]})', fontsize=16)
plt.ylabel('PM2.5 濃度', fontsize=12)
plt.xlabel('日期', fontsize=12)
plt.legend(fontsize=12, loc='upper left')
plt.tight_layout()
plt.show()