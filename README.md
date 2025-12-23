# ML Final Report:  Taiwan PM2.5 Prediction System

## 項目概述

本項目使用多種機器學習與深度學習模型預測**臺灣空氣品質 (PM2.5 濃度)**，並進行模型性能對比與特徵重要性分析。

### 主要特點

 **5 大預測模型**:  LSTM、RNN、XGBoost、Random Forest、及模型比較分析  
 **多層次分析**: 時間序列預測、各測站預測、特徵相關性分析  
 **完整可視化**: 折線圖、柱狀圖、熱力圖、預測對比圖  
 **生產級代碼**: 包含資料正規化、模型評估、早停機制  

---

## 項目結構

```
PM2.5_Prediction/
│
├──  README.md                          # 本檔案
├──  requirements.txt                   # 依賴環境
│
├──  src/                               #  主要代碼目錄 
│   │
│   ├──  LSTM 系列 (深度學習時間序列)
│   │   ├── LSTM_TaipeiPM2.5.py          # LSTM 未來預測 (2026-2028 三年預測)
│   │   ├── LSTM_linechart.py            # LSTM 預測折線圖 (實際 vs 預測)
│   │   ├── LSTM_station. py              # 多測站 LSTM 預測 (48h lookback, 雙層架構)
│   │   ├── LSTM_barchart.py             # LSTM 性能評估柱狀圖
│   │   └── LSTM_vs_RNN.py               # 對比分析:  LSTM vs RNN 性能
│   │
│   ├──  RNN 系列 (簡單循環神經網絡)
│   │   ├── RNN_station.py               # SimpleRNN 按測站預測
│   │   ├── RNN_linechart.py             # RNN 預測結果對比圖
│   │   └── RNN_barchart.py              # RNN 性能柱狀圖 & 特徵重要性
│   │
│   ├──  XGBoost 系列 (梯度提升決策樹)
│   │   ├── XGBoost_linechart.py         # XGBoost 預測折線圖 (10 天 240h)
│   │   ├── XGBoost_barchart.py          # XGBoost 各站精度 + 特徵排名
│   │   └── XGBoost_vs_Random_Forest.py  # 模型對決:  XGB vs RF
│   │
│   ├──  Random Forest 系列 (隨機森林)
│   │   ├── randomforest_linechart.py    # RF 預測結果折線圖
│   │   └── randomforest_barchart.py     # RF 特徵重要性 & 各站 MAE
│   │
│   ├── 📊 特徵分析 & 相關性研究
│   │   ├── All_Pearson_Correlation. py   # 核心特徵皮爾森相關性熱力圖
│   │   ├── Weather_Pearson_Correlation.py # 氣象因子對 PM2.5 的影響
│   │   ├── feature_barchart.py          # 特徵相關性柱狀圖
│   │   └── weather_barchart.py          # 氣象變數影響分數分析
│
├── content/                           # 資料目錄 
│   ├── FINAL_MODEL_TRAINING_DATA.csv                  #  主訓練資料
│   ├── ALL_YEARS_PM25_TARGET_AND_LAG_FEATURES.csv    # PM2.5 滯後特徵
│   ├── ALL_YEARS_METEO_STANDARDIZED. csv              # 氣象標準化數據
├── └── 微軟正黑體-1.ttf                              # 中文字體 (可選)

```

---

## 使用方法

### 1️⃣ 環境設置

```bash
# 安裝依賴
pip install -r requirements.txt
```

### 2️⃣ 資料準備

在 Google Colab 中上傳以下檔案到 `/content/` 目錄：
- `FINAL_MODEL_TRAINING_DATA.csv` ← 主訓練資料
- `ALL_YEARS_PM25_TARGET_AND_LAG_FEATURES.csv` ← PM2.5 特徵數據
- `ALL_YEARS_METEO_STANDARDIZED.csv` ← 氣象特徵數據
- `微軟正黑體-1.ttf` ← 中文字體 (可選)

### 3️⃣ 運行模型

#### 執行 LSTM 預測
```bash
python LSTM_TaipeiPM2.5.py      # 2026-2028 年度預測
python LSTM_linechart.py        # 預測結果折線圖
python LSTM_station.py          # 各測站詳細預測
```

#### 執行 XGBoost 預測
```bash
python XGBoost_linechart.py     # XGBoost 折線圖
python randomforest_barchart.py # Random Forest 特徵重要性
```

#### 執行特徵分析
```bash
python All_Pearson_Correlation. py      # 核心特徵相關性
python Weather_Pearson_Correlation.py  # 氣象相關性
```

---

## 模型說明

### LSTM (長短期記憶網絡)
- **算法**: 深度學習時間序列模型
- **參數**: 50 units, 1 層, Dropout 0.2
- **優勢**: 捕捉長期時間依賴，適合序列預測
- **輸入**: 過去 48 小時 PM2.5 與氣象數據
- **輸出**: 未來 PM2.5 濃度預測

**核心特徵** (來自 LSTM_station.py):
```python
feature_cols = [
    'PM25_Lag_1h', 'PM25_Lag_2h', 'PM25_Lag_24h',  # PM2.5 滯後特徵
    'RAINFALL', 'WIND_SPEED', 'RH', 'AMB_TEMP',     # 氣象特徵
    'Wind_Sin', 'Wind_Cos', 'Hour_Sin', 'Hour_Cos' # 向量化特徵
]
```

### RNN (簡單循環神經網絡)
- **算法**: SimpleRNN 層 256 units
- **參數**:  Dropout 0.2, 20 epochs
- **特點**: 輕量級，用於與 LSTM 比較
- **訓練集/測試集**: 90%/10% 分割

### XGBoost (梯度提升)
- **參數**: 500 estimators, learning_rate=0.05, max_depth=6
- **優勢**: 非常規速度快，特徵重要性清晰
- **評估**: R² 分數、MAE、RMSE

### Random Forest (隨機森林)
- **參數**: 100 estimators, n_jobs=-1 (並行)
- **優勢**: 無需特徵正規化，抗過擬合
- **輸出**: 各測站 MAE、特徵重要性排名

---

## 主要分析

### 1. 特徵相關性分析 (All_Pearson_Correlation. py)
```
PM2.5 與其他變數的皮爾森相關係數：
- PM25_Lag_1h:    0.95+ 
- AMB_TEMP:      0.40-0.60
- WIND_SPEED:    -0.30-0.30
- RH (濕度):     -0.20-0.20
```

### 2. 氣象因子影響 (Weather_Pearson_Correlation.py)
```
雨量、風速、濕度對 PM2.5 的影響程度分析
負相關:  雨量 & 風速 (增加會降低 PM2.5)
```

### 3. 時間序列預測 (2024/12/20 ~ 12/29)
- **LSTM**: 折線圖對比預測值 vs 真實值
- **XGBoost**: 240 小時預測結果 (10 天)
- **Random Forest**: R² Score 評估

---

## 關鍵代碼片段

### 資料正規化 (所有模型通用)
```python
from sklearn.preprocessing import MinMaxScaler

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X. fit_transform(X)
y_scaled = scaler_y. fit_transform(y)
```

### LSTM 序列構建
```python
def create_sequences(X, y, time_steps):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

X_seq, y_seq = create_sequences(X_scaled, y_scaled, lookback_hours=48)
```

### 模型訓練 (Early Stopping)
```python
from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model.fit(X_train, y_train, 
          epochs=50, 
          callbacks=[early_stop],
          validation_split=0.1)
```

---

## 輸出示例

### 預測圖表類型

#### 1. 折線圖 (linechart)
```
PM2.5 濃度
│     ╱╲    ╱╲
│    ╱  ╲  ╱  ╲
│   ╱    ╲╱    ╲
└───────────────── 日期
  真實 PM2.5 (紅色)
  預測結果 (藍色虛線)
```

#### 2. 柱狀圖 (barchart)
- **Model Performance**: 各模型的 MAE、RMSE、R² Score
- **Station Accuracy**: 各測站預測準確度
- **Feature Importance**: 特徵重要性排名 (Log Scale)

#### 3. 熱力圖 (Heatmap)
- 皮爾森相關係數矩陣
- 色彩範圍: -1 (冷藍) 到 +1 (暖紅)

---

##  進階設置

### 調整 LSTM 參數
```python
# LSTM_station.py 中修改
lookback_hours = 48         # 改為 24 或 72
LSTM(128, return_sequences=True)  # 調整 units
Dropout(0.3)                # 增加防止過擬合
```

### 自訂預測日期範圍
```python
# XGBoost_linechart. py
plot_start_date = '2024-12-20'  # 改為任意日期
days_to_show = 10               # 調整天數
```
---

## 技術支援

- **資料前處理**: 查看 LSTM_station.py 的第 30-52 行
- **模型評估**: 參考 randomforest_barchart.py 的 evaluate_model() 函數
- **可視化配置**: 編輯各 linechart.py 檔案的 matplotlib 參數

---

## License

本項目為教學用途 (ML Final Report)