# ML Final Report: Taiwan PM2.5 Prediction System
## 項目概述

本項目使用 **5 大機器學習與深度學習模型** 預測臺灣的 **PM2.5 濃度**，並進行模型性能對比與特徵重要性分析。通過多層次的數據分析、特徵工程和可視化，建立了一套完整的空氣品質預測系統。

### 主要特點

- **5 大預測模型**:  LSTM、RNN、XGBoost、Random Forest 及對比分析
- **多層次分析**: 時間序列預測、多測站預測、特徵相關性分析
- **完整可視化**: 折線圖、柱狀圖、熱力圖、性能對比圖
- **生產級代碼**: 包含數據正規化、模型評估、早停機制、學習率衰減等優化
- **特徵工程**: 時間週期向量化、風向向量化、滯後特徵
- **詳細評估**: 2024/12/31 每小時預測詳細表格、多維度性能指標

---

## 項目結構

```
PM2.5_Prediction/
│
├── README.md                              # 項目說明文檔
├── requirements.txt                       # Python 依賴環境
│
├── src/                                   # 主要代碼目錄 (25 個 Python 檔)
│   │
│   ├── LSTM 系列 (長短期記憶網絡 - 深度學習)
│   │   ├── LSTM_TaipeiPM2.5.py            # LSTM 未來預測 (2026-2028 三年預測)
│   │   ├── LSTM_12_31_PM2.5.py            # LSTM 2024/12/31 詳細評估 (雙層架構)
│   │   ├── LSTM_station. py               # 多測站 LSTM 預測 (48h lookback, 特徵工程)
│   │   ├── LSTM_linechart.py              # LSTM 預測折線圖 (實際 vs 預測)
│   │   ├── LSTM_barchart.py               # LSTM 性能評估柱狀圖
│   │   └── LSTM_vs_RNN.py                 # 對比分析:  LSTM vs RNN 性能
│   │
│   ├── RNN 系列 (簡單循環神經網絡 - 深度學習)
│   │   ├── RNN_12_31_PM2.5.py             # RNN 2024/12/31 詳細評估
│   │   ├── RNN_station.py                 # SimpleRNN 按測站預測
│   │   ├── RNN_linechart.py               # RNN 預測結果對比圖
│   │   └── RNN_barchart.py                # RNN 性能柱狀圖 & 特徵重要性
│   │
│   ├── XGBoost 系列 (梯度提升決策樹 - 集成學習)
│   │   ├── XGBoost_12_31_PM2.5.py         # XGBoost 2024/12/31 詳細評估
│   │   ├── XGBoost_linechart.py           # XGBoost 預測折線圖 (10 天 240h)
│   │   ├── XGBoost_barchart.py            # XGBoost 各站精度 + 特徵排名
│   │   └── XGBoost_vs_Random_Forest.py    # 模型對決:  XGB vs RF
│   │
│   ├── Random Forest 系列 (隨機森林 - 集成學習)
│   │   ├── RandomForest_12_31_PM2.5.py    # Random Forest 2024/12/31 詳細評估 (基準模型)
│   │   ├── randomforest_linechart.py      # RF 預測結果折線圖
│   │   └── randomforest_barchart.py       # RF 特徵重要性 & 各站 MAE
│   │
│   └── 特徵分析 & 相關性研究
│       ├── All_Pearson_Correlation.py     # 核心特徵皮爾森相關性熱力圖
│       ├── Weather_Pearson_Correlation.py # 氣象因子對 PM2.5 的影響分析
│       ├── feature_barchart.py            # 特徵相關性柱狀圖
│       └── weather_barchart.py            # 氣象變數影響分數分析
│
└── content/                               # 數據目錄
    ├── FINAL_MODEL_TRAINING_DATA.csv      # 主訓練資料 
    ├── ALL_YEARS_PM25_TARGET_AND_LAG_FEATURES.csv  # PM2.5 滯後特徵 
    ├── ALL_YEARS_METEO_STANDARDIZED. csv   # 氣象標準化數據 
    └── 微軟正黑體-1. ttf                   # 中文字體用於繪圖
```

---

##  快速開始

### 1️⃣ 環境設置

```bash
# 克隆項目
git clone https://github.com/mato1321/PM2.5_Prediction.git
cd PM2.5_Prediction

# 安裝依賴
pip install -r requirements.txt
```

### 2️⃣ 數據準備

確保以下檔案已放入 `content/` 目錄：

- `FINAL_MODEL_TRAINING_DATA.csv` ← 主訓練資料
- `ALL_YEARS_PM25_TARGET_AND_LAG_FEATURES. csv` ← PM2.5 滯後特徵數據
- `ALL_YEARS_METEO_STANDARDIZED.csv` ← 氣象特徵數據
- `微軟正黑體-1.ttf` ← 中文字體 (可選，用於圖表展示)

### 3️⃣ 運行模型

####  **特徵分析** 
```bash
cd src
python All_Pearson_Correlation. py       # 查看所有特徵相關性
python Weather_Pearson_Correlation. py   # 分析氣象因子影響
python feature_barchart.py               # 特徵相關性柱狀圖
python weather_barchart.py               # 氣象變數影響分數
```

####  **Random Forest 預測** (性能最佳)
```bash
python RandomForest_12_31_PM2.5.py       # 2024/12/31 預測評估
python randomforest_linechart.py         # 折線圖可視化
python randomforest_barchart. py         # 性能和特徵重要性
```

####  **XGBoost 預測** (速度最快)
```bash
python XGBoost_12_31_PM2.5.py            # 2024/12/31 預測評估
python XGBoost_linechart. py             # 折線圖可視化
python XGBoost_barchart.py               # 各站精度分析
python XGBoost_vs_Random_Forest.py       # 與 Random Forest 對比
```

####  **LSTM 預測** (深度學習)
```bash
python LSTM_12_31_PM2.5.py               # 2024/12/31 詳細評估
python LSTM_linechart.py                 # 預測結果對比圖
python LSTM_station.py                   # 多測站詳細預測
python LSTM_barchart.py                  # 性能評估柱狀圖
python LSTM_TaipeiPM2.5.py               # 2026-2028 三年預測
```

####  **RNN 預測** (對比測試)
```bash
python RNN_12_31_PM2.5.py                # 2024/12/31 詳細評估
python RNN_linechart.py                  # 預測結果對比圖
python RNN_station.py                    # 多測站預測
python RNN_barchart.py                   # 性能評估
```

####  **模型對比**
```bash
python LSTM_vs_RNN.py                    # LSTM vs RNN 對比
```

---

##  模型詳細說明

### LSTM (長短期記憶網絡)

**特點**:  深度學習時間序列模型，擅長捕捉長期時間依賴

#### LSTM_TaipeiPM2.5.py
- **用途**: 預測 2026-2028 年未來三年 PM2.5 趨勢
- **模型**:  單層 LSTM (50 units)
- **訓練方式**: 模擬季節性數據 (每 4 個月數據一個點)
- **輸出**: 9 個季度的預測值和折線圖

#### LSTM_12_31_PM2.5.py 
- **用途**: 預測 2024/12/31 整天 PM2.5（每小時預測）
- **架構**:
  ```
  LSTM Layer 1: 256 units → BatchNormalization → Dropout(0.2)
  LSTM Layer 2: 128 units → BatchNormalization → Dropout(0.2)
  Dense Layer 1: 128 units → BatchNormalization → Dropout(0.2)
  Dense Layer 2: 64 units → Dropout(0.15)
  Dense Layer 3: 32 units → Dropout(0.1)
  Dense Layer 4: 16 units
  Output: 1 unit (PM2.5 預測值)
  ```
- **優化器**:  Nadam (learning_rate=0.001)
- **回調**: EarlyStopping (patience=30) + ReduceLROnPlateau (factor=0.5)
- **輸入**: 過去數據標準化後的特徵
- **輸出**: 
  - 詳細的每小時預測值對比表格
  - R² Score、MAE、RMSE 評估指標
  - 修正前後性能對比

#### LSTM_station. py
- **用途**: 多測站詳細預測
- **特徵數**: 12 個精心設計的特徵
  ```python
  feature_cols = [
      'PM25_Lag_1h',    # 前 1 小時 PM2.5
      'PM25_Lag_2h',    # 前 2 小時 PM2.5
      'PM25_Lag_24h',   # 前 24 小時 PM2.5
      'RAINFALL',       # 降雨量
      'WIND_SPEED',     # 風速
      'RH',             # 相對濕度
      'AMB_TEMP',       # 環境溫度
      'Wind_Sin',       # 風向 Sin 分量
      'Wind_Cos',       # 風向 Cos 分量
      'Hour_Sin',       # 時間 Sin 分量
      'Hour_Cos'        # 時間 Cos 分量
  ]
  ```
- **Lookback Window**: 48 小時 (過去 2 天數據)
- **預測期間**: 2024-12-20 ~ 2024-12-29

#### LSTM_linechart.py
- **功能**: 生成預測 vs 實際值的折線圖
- **時間範圍**: 最後 10 天 (240 小時)
- **視覺元素**:
  - 紅色實線:  真實 PM2.5
  - 藍色虛線: 預測結果

#### LSTM_barchart. py
- **功能**: 性能評估柱狀圖
- **左圖**: 各測站預測誤差 (MAE)
- **右圖**: 特徵重要性排名 (Log 尺度)
- **指標**: R²、MAE、RMSE

---

### RNN (簡單循環神經網絡)

**特點**: 輕量級循環神經網絡，用於與 LSTM 對比

#### RNN_12_31_PM2.5.py
- **用途**: 預測 2024/12/31 整天 PM2.5
- **架構**:
  ```
  SimpleRNN Layer 1: 256 units → BatchNormalization → Dropout(0.2)
  SimpleRNN Layer 2: 128 units → BatchNormalization → Dropout(0.2)
  Dense Layer 1: 128 units → BatchNormalization → Dropout(0.2)
  Dense Layer 2: 64 units → Dropout(0.15)
  Dense Layer 3: 32 units → Dropout(0.1)
  Dense Layer 4: 16 units
  Output: 1 unit
  ```
- **優化器**: Nadam (learning_rate=0.001)
- **性能**: R² ≈ 0.75-0.85 (略低於 LSTM)

#### RNN_linechart.py & RNN_barchart.py
- 功能同 LSTM 系列
- 綠色標記用於區分 RNN 預測結果

---

### XGBoost (梯度提升決策樹)

**特點**: 速度快、準確度高、特徵重要性清晰

#### XGBoost_12_31_PM2.5.py 
- **用途**: 預測 2024/12/31 整天 PM2.5
- **模型參數**:
  ```python
  XGBRegressor(
      n_estimators=500,        # 500 棵決策樹
      learning_rate=0.05,      # 學習率
      max_depth=6,             # 樹深度
      subsample=0.8,           # 樣本子集
      colsample_bytree=0.8,    # 特徵子集
      n_jobs=-1,               # 並行計算
      random_state=42
  )
  ```
- **性能**:
  ```
  R² Score: ~0.88 (全測試集)
  R² Score: ~0.88 (2024/12/31 修正版)
  MAE: ~0.45
  RMSE: ~0.62
  訓練時間: 極快 (< 1 分鐘)
  ```
- **輸出**: 24 小時每小時預測表、修正前後性能對比

#### XGBoost_linechart.py
- 功能:  時間序列折線圖
- 時間範圍: 2024-12-20 ~ 2024-12-29 (10 天)

#### XGBoost_barchart.py
- 左圖: 各測站 MAE 排名
- 右圖:  特徵重要性排名 (前 10)

#### XGBoost_vs_Random_Forest.py
- 並行訓練 XGBoost 和 Random Forest
- 詳細對比性能指標
- 彙總表格展示

---

### Random Forest (隨機森林)

**特點**: 無須特徵正規化、抗過擬合、性能最佳

#### RandomForest_12_31_PM2.5.py 
- **用途**: 預測 2024/12/31 整天 PM2.5（基準參考）
- **模型參數**:
  ```python
  RandomForestRegressor(
      n_estimators=300,        # 300 棵決策樹
      max_depth=15,            # 樹深度
      n_jobs=-1,               # 並行計算
      random_state=42
  )
  ```
- **性能**:
  ```
  全測試集
  R² Score: 0.9200 (最佳)
  MAE: ~0.40
  MSE: ~0.25
  
  2024/12/31 修正版
  R² Score: 0.9200
  MAE: ~0.40
  RMSE: ~0.65
  ```
- **輸出**: 
  - 24 小時每小時預測表（與實際值對比）
  - 詳細的誤差分析

#### randomforest_linechart.py
- 功能: 時間序列折線圖
- 配色: 紅色(實際) vs 藍色虛線(預測)

#### randomforest_barchart. py 📊 (最詳細)
- 左圖: 各測站預測誤差 (MAE)
- 右圖: 特徵重要性排名 (Log 尺度)
- 完整的模型評估報告

---

##  特徵分析

### All_Pearson_Correlation.py 

**目標**: 展示所有特徵與 PM2.5 的相關性

**分析特徵** (8 個):
```
PM2.5_Value      - 目標變數 (當下 PM2.5)
PM25_Lag_1h      - 相關性:  0.95+ (最強) 
PM25_Lag_24h     - 相關性:  0.80+
Hour             - 時間特徵
RH (濕度)        - 相關性: -0.20~0.20 (弱負相關)
AMB_TEMP (溫度)  - 相關性: 0.40~0.60 (正相關)
WIND_SPEED (風速)- 相關性: -0.30~0.30 (負相關)
Wind_Sin/Cos     - 風向向量化特徵
```

**視覺化**:  皮爾森相關係數熱力圖
- 紅色 = 正相關
- 藍色 = 負相關
- 強度 = 相關程度

**結論**: 
- PM25_Lag_1h 是最強預測因子 (0.95)
- 溫度與 PM2.5 呈正相關 (0.40-0.60)
- 風速和降雨呈負相關 (增加會降低 PM2.5)

---

### Weather_Pearson_Correlation.py 

**目標**: 專注分析氣象因子對 PM2.5 的影響

**分析變數**:
```
RAINFALL (降雨)     → 負相關 (-0.25)  [降雨增加，PM2.5 下降]
WIND_SPEED (風速)   → 負相關 (-0.15)  [風速增加，PM2.5 下降]
RH (相對濕度)       → 負相關 (-0.10)  [濕度增加，PM2.5 下降]
```

**視覺化**: 熱力圖或柱狀圖展示相關性大小

**應用**: 理解氣象因素對空氣品質的機制

---

### feature_barchart.py

**功能**: 特徵重要性排名柱狀圖

**排序**:  按相關係數從高到低
```
排序 1: PM25_Lag_1h    (0.95)
排序 2: AMB_TEMP        (0.50)
排序 3: Hour            (0.30)
排序 4: WIND_SPEED      (-0.25)
...
```

**配色**:  Coolwarm 漸變 (突出強弱相關)

---

### weather_barchart.py

**功能**: 氣象變數影響分數分析

**分類**:
- 紅色柱 = 正相關 (增加會提升 PM2.5)
- 藍色柱 = 負相關 (增加會降低 PM2.5)

**標籤**: 每根柱子上顯示相關係數數值

---

## 模型性能對比

### 2024/12/31 預測評估 (每小時預測)

| 模型 | 架構 | R² Score | MAE | RMSE | 訓練時間 |
|------|------|----------|-----|------|----------|
| **Random Forest** | 300 棵樹 | **0.9200** | **0.40** | **0.65** | 30 秒 | 
| **XGBoost** | 500 棵樹 | 0.8800 | 0.45 | 0.62 | 10 秒 | 
| **LSTM** | 雙層 256→128 | 0.8500 | 0.50 | 0.68 | 2 分鐘 | 
| **RNN** | 雙層 256→128 | 0.8000 | 0.55 | 0.72 | 1.5 分鐘 | 

**結論**:
- Random Forest 性能最佳 (R²=0.92)，適合生產環境
- XGBoost 速度最快，精度接近 Random Forest
- LSTM 穩定可靠，適合長期預測
- RNN 性能較弱，主要用於模型對比

---

### 聯繫方式

- GitHub: [mato1321/PM2.5_Prediction](https://github.com/mato1321/PM2.5_Prediction)
- Issues: 提交 GitHub Issues 報告問題

---

## License

本項目為教學用途 (ML Final Report)，遵循 MIT License