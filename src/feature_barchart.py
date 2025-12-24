import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

file_path = '/content/FINAL_MODEL_TRAINING_DATA.csv'
df = pd.read_csv(file_path)
print("✅ 資料讀取成功！")

selected_features = [
    'PM2.5_Value',
    'PM25_Lag_1h',
    'Wind_Cos',
    'AMB_TEMP',
    'Hour',
    'Wind_Sin',
    'WIND_SPEED',
    'RH'
]


selected_df = df[selected_features]
selected_df = selected_df.dropna()
corr_matrix = selected_df.corr()

# 與 PM2.5_Value 的相關係數
target_col = 'PM2.5_Value'
target_corr = corr_matrix[target_col].drop(target_col)
target_corr = target_corr.sort_values(ascending=False)

# 繪圖
plt.figure(figsize=(10, 6))
sns.barplot(
    x=target_corr.index,
    y=target_corr.values,
    palette='coolwarm'
)

plt.title('Correlation with PM2.5_Value (Selected Features)', fontsize=15)
plt.ylabel('Correlation Coefficient', fontsize=12)
plt.xlabel('Features', fontsize=12)
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
