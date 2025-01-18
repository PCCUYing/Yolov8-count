# YOLOv8 區域偵測與追蹤

此專案使用 YOLOv8 物件偵測模型追蹤進入定義區域的物件（例如：人）。程式會計算每個物件在指定區域內停留的時間，並在影片畫面中顯示相關資訊。

## 功能特點

- 使用 YOLOv8 進行物件偵測。
- 追蹤進入定義多邊形區域的物件。
- 計算並顯示物件在區域內的停留時間。
- 在畫面上顯示邊界框、物件類別名稱、信心指數以及停留時間。
- 支援動態追蹤物件，並設有超時機制移除未活動的物件。

## 先決條件

### 所需函式庫

- Python 3.8+
- OpenCV
- NumPy
- Shapely
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)

安裝所需函式庫，執行以下指令：

```bash
pip install opencv-python-headless numpy shapely ultralytics
```

### 輸入影片

- 確保目標影片檔案位於 `testfile/Counter.mp4`，若路徑不同，請更新程式中的 `target` 變數。

### YOLOv8 模型

- 下載並將 YOLOv8 模型檔案（`yolov8m.pt`）放置於程式相同目錄，或更新 `model` 變數指定模型的路徑。

## 使用方法

### 1. 建立 Conda 環境

1. 創建 Conda 環境：
   ```bash
   conda create -n yolov8 python=3.8 anaconda
   ```
2. 啟用環境：
   ```bash
   conda activate yolov8
   ```

### 2. 複製專案並安裝依賴

1. 複製專案：
   ```bash
   git clone https://github.com/你的帳號/Flask-YOLOv8-App.git
   cd Flask-YOLOv8-App
   ```
2. 安裝必要依賴：
   ```bash
   pip install -r requirements.txt
   ```

### 3. 執行程式

1. 確保已安裝必要函式庫並設置正確的影片檔案路徑。
2. 執行程式：
   ```bash
   python main.py
   ```
3. 程式會開啟一個視窗，顯示帶有偵測結果的影片畫面。

## 程式概覽

### 主要函式

#### `drawArea`

- 在影像框架上繪製定義區域。

#### `inarea`

- 計算偵測到的物件與定義區域的重疊百分比。

### 物件追蹤邏輯

- 使用物件的中心座標進行追蹤。
- 透過字典 `tracked_objects` 記錄每個物件的進入時間（`start_time`）與最後一次出現時間（`last_seen`）。
- 若物件在設定超時期間（預設 5 秒）未被偵測到，將從追蹤字典中移除。

### 畫面顯示資訊

- 邊界框：在偵測到的物件周圍繪製矩形。
- 類別名稱：顯示偵測到的物件名稱（例如：`person`）。
- 信心指數：顯示模型對偵測結果的信心分數。
- 停留時間：顯示物件在指定區域內的停留時間（以秒為單位）。

## 可調整參數

- **超時時間（********`timeout`********）**：未活動物件從追蹤移除的時間間隔（預設：5 秒）。
- **信心門檻（********`r`********）**：顯示偵測結果所需的最低信心分數（預設：0.5）。
- **定義區域（********`area`********）**：可更新 `area` 變數以修改多邊形檢測區域。

## 已知限制

- 追蹤依賴物件中心座標，若物件移動迅速或尺寸變化大，可能導致追蹤不準確。
- 多邊形檢測區域需以像素座標手動定義。

## 範例輸出

- 顯示帶有偵測物件的影片視窗，包括邊界框、類別名稱、信心分數與停留時間。

## 授權

此專案為開源專案，遵循 MIT 授權。

## 致謝

- 感謝 [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) 提供的物件偵測框架。
- 感謝 OpenCV 提供的影像處理功能。

