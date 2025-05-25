# YOLOv8 人臉偵測與情緒分析系統

這是一個基於 YOLOv8 的人臉偵測與情緒分析系統，能夠即時偵測人臉並分析情緒狀態。

## 功能特點

- 即時人臉偵測
- 情緒狀態分析
- 支援多種 YOLOv8 模型
- 即時視訊串流處理

## 系統需求

- Python 3.8 穩定 (不支援更高版本)
- CUDA 支援（建議用於 GPU 加速）

## 安裝步驟

1. 克隆專案：
```bash
git clone [專案網址]
cd yolov8
```

2. 建立虛擬環境（建議）：
```bash
python -m venv .venv
```

3. 啟動虛擬環境：
- Windows:
```bash
.venv\Scripts\activate
```
- Linux/Mac:
```bash
source .venv/bin/activate
```

4. 安裝依賴套件：
```bash
pip install -r requirements.txt
```

5. 下載必要的模型檔案：
- YOLOv8n 模型（輕量版）：[下載連結](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt)
- YOLOv8x 模型（完整版）：[下載連結](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt)
- YOLOv8-face 模型：[下載連結](https://github.com/kanosawa/yolov8-face/releases/download/v0.0.0/yolov8-face.pt)

將下載的模型檔案放在專案根目錄下。

## 使用方法

1. 啟動主程式：
```bash
python main.py
```

2. 使用不同的模型：
- 使用 YOLOv8n 模型（輕量版）：
```bash
python yolo.py --model yolov8n.pt
```
- 使用 YOLOv8x 模型（完整版）：
```bash
python yolo.py --model yolov8x.pt
```

## 專案結構

- `app.py`: 主應用程式
- `yolo.py`: YOLO 模型處理核心
- `emotion_detection.py`: 情緒分析模組
- `cursor.py`: 滑鼠控制模組
- `dataset/`: 資料集目錄
- `runs/`: 執行結果輸出目錄

## 模型說明

- `yolov8n.pt`: 輕量級模型，適合一般使用
- `yolov8x.pt`: 完整版模型，提供更精確的偵測結果
- `yolov8-face.pt`: 專門用於人臉偵測的模型

## 注意事項

- 確保系統已安裝 CUDA 和對應的 GPU 驅動程式
- 建議使用虛擬環境以避免套件衝突
- 首次執行時會自動下載所需的模型檔案
- 模型檔案較大，請確保有足夠的網路頻寬和儲存空間

## 授權說明

本專案採用 MIT 授權條款
 
