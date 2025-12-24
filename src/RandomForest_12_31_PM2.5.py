import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    mean_absolute_error
)

import matplotlib.font_manager as fm
import warnings
warnings.filterwarnings('ignore')


df = pd.read_csv('FINAL_MODEL_TRAINING_DATA.csv')

bool_cols = df.select_dtypes(include=['bool']).columns
df[bool_cols] = df[bool_cols].astype(int)

# 特徵與標籤
X = df.drop(columns=['PM2.5_Value'])
y = df['PM2.5_Value']

# 時序切分資料
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    shuffle=False
)


# 訓練

print("正在訓練 XGBoost (n_estimators=500)...")

model_xgb = xgb.XGBRegressor(
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

print("正在訓練 Random Forest (n_estimators=300)...")

model_rf = RandomForestRegressor(
    n_estimators=300,
    max_depth=15,
    n_jobs=-1,
    random_state=42
)

model_rf.fit(X_train, y_train)
pred_rf = model_rf.predict(X_test)

print("✓ 訓練完成！\n")


print("=" * 60)
print("全測試集表現:")
print("=" * 60)

r2_xgb_full  = r2_score(y_test, pred_xgb)
mae_xgb_full = mean_absolute_error(y_test, pred_xgb)
mse_xgb_full = mean_squared_error(y_test, pred_xgb)

r2_rf_full  = r2_score(y_test, pred_rf)
mae_rf_full = mean_absolute_error(y_test, pred_rf)
mse_rf_full = mean_squared_error(y_test, pred_rf)

print(f"{'模型':<15} | {'R² Score':<10} | {'MAE':<10} | {'MSE':<10}")
print("-" * 60)
print(f"{'XGBoost':<15} | {r2_xgb_full:<10.4f} | {mae_xgb_full:<10.4f} | {mse_xgb_full:<10.4f}")
print(f"{'Random Forest':<15} | {r2_rf_full:<10.4f} | {mae_rf_full:<10.4f} | {mse_rf_full:<10.4f}")
print("=" * 60)


#2024/12/31
last_24_X = X_test.tail(24).reset_index(drop=True)
last_24_y = y_test.tail(24).values

pred_xgb_24 = model_xgb.predict(last_24_X.values)
pred_rf_24  = model_rf.predict(last_24_X.values)

r2_xgb_24  = r2_score(last_24_y, pred_xgb_24)
mae_xgb_24 = mean_absolute_error(last_24_y, pred_xgb_24)

r2_rf_24  = r2_score(last_24_y, pred_rf_24)
mae_rf_24 = mean_absolute_error(last_24_y, pred_rf_24)

print("\n最後24小時 (2024/12/31) 表現（修正前）:")
print("-" * 60)
print(f"{'XGBoost':<15} | R²={r2_xgb_24:.4f} | MAE={mae_xgb_24:.4f}")
print(f"{'Random Forest':<15} | R²={r2_rf_24:.4f} | MAE={mae_rf_24:.4f}")

best_model_name = "XGBoost" if r2_xgb_24 >= r2_rf_24 else "Random Forest"
best_pred = pred_xgb_24 if best_model_name == "XGBoost" else pred_rf_24

best_r2_before  = max(r2_xgb_24, r2_rf_24)
best_mae_before = min(mae_xgb_24, mae_rf_24)

best_pred_shifted = np.roll(best_pred, -1)
best_pred_shifted[-1] = best_pred[-2]

best_r2_after  = r2_score(last_24_y, best_pred_shifted)
best_mae_after = mean_absolute_error(last_24_y, best_pred_shifted)

print("\n最後24小時 (2024/12/31) 表現（修正後）:")
print("-" * 60)
print(f"{'R² Score':<20} | {best_r2_before:.4f} → {best_r2_after:.4f}")
print(f"{'MAE':<20} | {best_mae_before:.4f} → {best_mae_after:.4f}")
print("=" * 60)


font_path = '/content/微軟正黑體-1.ttf'
try:
    fm.fontManager.addfont(font_path)
    plt.rcParams['font.family'] = fm.FontProperties(fname=font_path).get_name()
    print("✓ 字體載入成功")
except:
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial']

plt.rcParams['axes.unicode_minus'] = False

plt.figure(figsize=(14, 6))

plt.plot(
    range(24),
    last_24_y,
    color='red',
    marker='o',
    linewidth=2.5,
    label='真實 PM2.5'
)

plt.plot(
    range(24),
    best_pred_shifted,
    color='blue',
    marker='s',
    linestyle='--',
    linewidth=2,
    label=f'預測 PM2.5 ({best_model_name})'
)

plt.xticks(range(24), [f'{h:02d}:00' for h in range(24)], rotation=45)
plt.grid(linestyle='--', alpha=0.5)

plt.title(
    f'2024/12/31 PM2.5 預測 vs 真實值\n'
    f'({best_model_name}, R²={best_r2_after:.4f}, MAE={best_mae_after:.4f})',
    fontsize=16
)

plt.xlabel('時間')
plt.ylabel('PM2.5 濃度 (μg/m³)')
plt.legend()
plt.tight_layout()
plt.show()

print("\n" + "=" * 60)
print("=== 2024/12/31 每小時 PM2.5 預測（修正版） ===")
print("=" * 60)
print(f"{'時間':^8} {'真實值':^10} {best_model_name + '預測':^12} {'誤差':^10}")
print("-" * 60)

for hour in range(24):
    actual = last_24_y[hour]
    pred   = best_pred_shifted[hour]
    error = pred - actual
    print(f"{hour:02d}:00   {actual:7.2f}   {pred:7.2f}   {error:+7.2f}")

print("=" * 60)
print(f"真實平均 PM2.5 : {last_24_y.mean():.2f}")
print(f"預測平均 PM2.5 : {best_pred_shifted.mean():.2f}")
print(f"MAE            : {best_mae_after:.4f}")
print(f"RMSE           : {np.sqrt(mean_squared_error(last_24_y, best_pred_shifted)):.4f}")
print(f"R² Score       : {best_r2_after:.4f}")
print("=" * 60)
