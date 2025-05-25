from ultralytics import YOLO
import cv2

# 載入你訓練好的模型
model = YOLO('runs/detect/train28/weights/best.pt')  # 修改為你的.pt路徑

# 開啟攝影機（0 是內建攝影機）
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 偵測影像
    results = model.predict(source=frame, save=False, imgsz=640, conf=0.5)

    # 繪製結果
    annotated_frame = results[0].plot()

    # 顯示
    cv2.imshow('YOLOv8 Real-time Detection', annotated_frame)

    # 按 Q 結束
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 清理
cap.release()
cv2.destroyAllWindows()
