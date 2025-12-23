import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# --- 1. 設定中文字體 (Colab 專用，避免亂碼) ---
font_path = "/content/微軟正黑體-1.ttf"
fm.fontManager.addfont(font_path)
plt.rcParams['font.family'] = fm.FontProperties(fname=font_path).get_name()
plt.rcParams['axes.unicode_minus'] = False # 讓負號正常顯示

# --- 2. 讀取資料 ---
# 請確認已將 FINAL_MODEL_TRAINING_DATA.csv 上傳到 Colab 左側
df = pd.read_csv("FINAL_MODEL_TRAINING_DATA.csv")

# --- 3. 挑選您指定的變數 + PM2.5 ---
# 我們把 PM2.5 放在第一個，這樣看第一行(或第一列)就能知道它跟其他人的關係
target_cols = ['PM2.5_Value', 'RAINFALL', 'WIND_SPEED', 'RH']

# --- 4. 計算皮爾森相關係數 (Pearson Correlation) ---
corr_matrix = df[target_cols].corr()

# 印出數值
print("=== PM2.5 與氣象變數的相關係數 ===")
print(corr_matrix[['PM2.5_Value']])

# --- 5. 畫出熱力圖 ---
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1)
plt.title('PM2.5 與 氣象變數(雨量/風速/濕度) 相關性分析', fontsize=14)
plt.show()