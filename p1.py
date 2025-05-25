from ultralytics import YOLO

# 載入預訓練模型（你可以換成 yolov8n.pt / yolov8s.pt）
model = YOLO('yolov8n.pt')

# 開始訓練
model.train(
    data='data.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    device=0  # 0 表示使用 GPU（若有），若要用 CPU 則寫 device='cpu'
)
