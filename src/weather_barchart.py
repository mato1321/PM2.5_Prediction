import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# =========================
# 1. 設定檔案路徑並讀取資料
# =========================
file_path = '/content/FINAL_MODEL_TRAINING_DATA.csv'
df = pd.read_csv(file_path)
print("✅ 資料讀取成功")
print(df.head())
# =========================
# 2. 指定目標欄位與氣象欄位
# =========================
target_col = 'PM2.5_Value'
weather_cols = ['RAINFALL', 'WIND_SPEED', 'RH']

# 檢查欄位是否存在
missing_cols = [col for col in weather_cols + [target_col] if col not in df.columns]
if missing_cols:
    raise ValueError(f"❌ 缺少欄位：{missing_cols}")

# =========================
# 3. 計算相關係數
# =========================
correlations = (
    df[weather_cols]
    .corrwith(df[target_col])
    .sort_values(ascending=False)
)

print(f"\n--- 氣象因子對 {target_col} 的影響分數 ---")
print(correlations)
print("------------------------------------------")

# =========================
# 4. 繪製相關性長條圖
# =========================
plt.figure(figsize=(8, 5))

# 正相關紅色、負相關藍色
colors = ['#d62728' if x > 0 else '#1f77b4' for x in correlations.values]

ax = sns.barplot(
    x=correlations.index,
    y=correlations.values,
    palette=colors
)

plt.title('Weather Impact on PM2.5 (Correlation Score)', fontsize=14)
plt.ylabel('Correlation Coefficient', fontsize=12)
plt.xlabel('Weather Variables', fontsize=12)
plt.axhline(0, color='black', linewidth=0.8)
plt.grid(axis='y', linestyle='--', alpha=0.5)

# 在柱狀圖上標示數值
for i, v in enumerate(correlations.values):
    ax.text(
        i,
        v + (0.01 if v > 0 else -0.05),
        f"{v:.2f}",
        ha='center',
        va='bottom' if v > 0 else 'top',
        fontsize=10,
        fontweight='bold'
    )

plt.tight_layout()
plt.show()
