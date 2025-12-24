import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

font_path = "/content/微軟正黑體-1.ttf"
plt.rcParams['axes.unicode_minus'] = False

filename = 'FINAL_MODEL_TRAINING_DATA.csv'
if not df.empty:
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

    existing_cols = [c for c in selected_cols if c in df.columns]

    #計算相關係數
    corr_matrix = df[existing_cols].corr()

    # 繪製熱力圖
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        corr_matrix,
        annot=True,       
        fmt=".2f",      
        cmap='coolwarm',
        vmin=-1, vmax=1,  
        square=True,
        linewidths=0.5
    )

    plt.title('核心特徵皮爾森相關性', fontsize=16)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # 數值列表
    print("\n=== 各變數與 PM2.5 的相關係數排序 ===")
    print(corr_matrix['PM2.5_Value'].sort_values(ascending=False))