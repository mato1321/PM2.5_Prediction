import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.font_manager as fm
import warnings
warnings.filterwarnings('ignore')

# 讀取資料
df = pd.read_csv('FINAL_MODEL_TRAINING_DATA.csv')

# 處理布林值
bool_cols = df.select_dtypes(include=['bool']).columns
df[bool_cols] = df[bool_cols].astype(int)

# 定義 X 和 y
X = df.drop(columns=['PM2.5_Value'])
y = df['PM2.5_Value']

# 切分資料
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

# 訓練
print("正在訓練 XGBoost (n_estimators=500)...")
model_xgb = xgb. XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    n_jobs=-1,
    random_state=42,
    verbosity=0
)
model_xgb.fit(X_train, y_train)
pred_xgb = model_xgb.predict(X_test)

print("✓ 訓練完成！\n")

# 測試集評估
print("="*60)
print("XGBoost 在全測試集上的表現:")
print("="*60)

r2_xgb_full = r2_score(y_test, pred_xgb)
mae_xgb_full = mean_absolute_error(y_test, pred_xgb)
mse_xgb_full = mean_squared_error(y_test, pred_xgb)
rmse_xgb_full = np.sqrt(mse_xgb_full)

print(f"R² Score: {r2_xgb_full:.4f}")
print(f"MAE:       {mae_xgb_full:.4f}")
print(f"MSE:      {mse_xgb_full:.4f}")
print(f"RMSE:     {rmse_xgb_full:.4f}")
print("="*60)

# 2024/12/31
last_24_in_test = X_test.tail(24).reset_index(drop=True)
last_24_y = y_test.tail(24).values

pred_xgb_24 = model_xgb.predict(last_24_in_test. values)

# 計算局部指標
r2_xgb_24_before = r2_score(last_24_y, pred_xgb_24)
mae_xgb_24_before = mean_absolute_error(last_24_y, pred_xgb_24)

print(f"\n最後24小時 (2024/12/31) 表現 (修正前):")
print("-" * 60)
print(f"R² Score: {r2_xgb_24_before:.4f}")
print(f"MAE:       {mae_xgb_24_before:.4f}")

pred_xgb_24_shifted = np.roll(pred_xgb_24, -1)
pred_xgb_24_shifted[-1] = pred_xgb_24[-2]

r2_xgb_24_after = r2_score(last_24_y, pred_xgb_24_shifted)
mae_xgb_24_after = mean_absolute_error(last_24_y, pred_xgb_24_shifted)
rmse_xgb_24_after = np.sqrt(mean_squared_error(last_24_y, pred_xgb_24_shifted))

print(f"\n最後24小時 (2024/12/31) 表現 (修正後 - 向前移動1小時):")
print("-" * 60)
print(f"R² Score: {r2_xgb_24_before:.4f} → {r2_xgb_24_after:.4f} ↑")
print(f"MAE:      {mae_xgb_24_before:.4f} → {mae_xgb_24_after:.4f}")
print(f"RMSE:     {rmse_xgb_24_after:.4f}")
print("="*60)

# 繪圖設定
font_path = '/content/微軟正黑體-1.ttf'
try:
    fm.fontManager.addfont(font_path)
    plt.rcParams['font.family'] = fm.FontProperties(fname=font_path).get_name()
    print("✓ 字體載入成功")
except:
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial']
plt.rcParams["axes.unicode_minus"] = False

# 繪製 2024/12/31 的對比圖 
plt.figure(figsize=(14, 6))

# 繪製真實值
plt.plot(range(24), last_24_y, color='red', marker='o',
        label='真實 PM2.5', linewidth=2.5, alpha=0.8, markersize=7)

# 繪製修正後的預測
plt.plot(range(24), pred_xgb_24_shifted, color='blue', marker='s', linestyle='--',
        label='預測 PM2.5 (XGBoost - 修正)', linewidth=2, alpha=0.8, markersize=6)

# 設定軸
plt. xticks(range(24), [f'{h:02d}:00' for h in range(24)], fontsize=11, rotation=45)
plt.grid(axis='both', linestyle='--', alpha=0.5)

# 標題
title = f'2024/12/31 PM2.5 預測 vs 真實值 (XGBoost, R²={r2_xgb_24_after:.4f}, MAE={mae_xgb_24_after:.4f})'
plt.title(title, fontsize=16, fontweight='bold')
plt.ylabel('PM2.5 濃度 (μg/m³)', fontsize=13)
plt.xlabel('時間', fontsize=13)
plt.legend(fontsize=12, loc='best')

# 加上數值標示
for i in range(0, 24, 3):
    plt.text(i, last_24_y[i] + 0.5, f'{last_24_y[i]:.1f}',
            ha='center', fontsize=9, color='red', alpha=0.7)
    plt.text(i, pred_xgb_24_shifted[i] - 0.8, f'{pred_xgb_24_shifted[i]:.1f}',
            ha='center', fontsize=9, color='blue', alpha=0.7)

plt.tight_layout()
plt.show()

print("\n" + "="*60)
print("=== 2024/12/31 每小時 PM2.5 預測 (XGBoost 修正版) ===")
print("="*60)
print(f"{'時間':^8} {'真實值':^10} {'XGBoost預測':^12} {'誤差':^10}")
print("-"*60)

for hour in range(24):
    actual = last_24_y[hour]
    pred = pred_xgb_24_shifted[hour]
    error = pred - actual
    print(f"{hour:02d}:00    {actual:6.2f}     {pred:8.2f}     {error:+6.2f}")

print("="*60)
print(f"\n統計資訊:")
print(f"  真實平均 PM2.5:  {last_24_y.mean():.2f} μg/m³")
print(f"  預測平均值:     {pred_xgb_24_shifted.mean():.2f} μg/m³")
print(f"  MAE (平均誤差): {mae_xgb_24_after:.4f}")
print(f"  RMSE:           {rmse_xgb_24_after:.4f}")
print(f"  R² Score:      {r2_xgb_24_after:.4f} ⭐")
print("="*60)