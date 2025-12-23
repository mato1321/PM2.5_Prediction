import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# --- 1. 設定中文字體 (避免亂碼) ---
font_path = "/content/微軟正黑體-1.ttf"
plt.rcParams['axes.unicode_minus'] = False # 讓負號正常顯示

# --- 2. 讀取資料 ---
filename = 'FINAL_MODEL_TRAINING_DATA.csv'
if not df.empty:
    # --- 3. 指定要分析的變數 (不含測站) ---
    selected_cols = [
        'PM2.5_Value',    # 目標：當下 PM2.5
        'PM25_Lag_1h',    # 核心：前一小時 PM2.5
        'Hour',           # 時間：小時
        'RH',             # 氣象：濕度
        'AMB_TEMP',       # 氣象：氣溫
        'WIND_SPEED',     # 氣象：風速
        'Wind_Sin',       # 風向向量 Sin
        'Wind_Cos'        # 風向向量 Cos
    ]

    # 確認這些欄位真的存在於資料中
    existing_cols = [c for c in selected_cols if c in df.columns]

    # --- 4. 計算相關係數 ---
    corr_matrix = df[existing_cols].corr()

    # --- 5. 繪製熱力圖 ---
    plt.figure(figsize=(10, 8)) # 調整適當大小
    sns.heatmap(
        corr_matrix,
        annot=True,       # 顯示數值
        fmt=".2f",        # 兩位小數
        cmap='coolwarm',  # 冷暖色調
        vmin=-1, vmax=1,  # 固定範圍
        square=True,
        linewidths=0.5
    )

    plt.title('核心特徵皮爾森相關性', fontsize=16)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # --- 6. 數值列表 ---
    print("\n=== 各變數與 PM2.5 的相關係數排序 ===")
    print(corr_matrix['PM2.5_Value'].sort_values(ascending=False))