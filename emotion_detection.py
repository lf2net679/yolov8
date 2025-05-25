import cv2
import torch
from ultralytics import YOLO
import numpy as np

# 載入預訓練的YOLOv8模型
model = YOLO('yolov11n.pt')  # 使用最小的模型版本

# 初始化視訊攝影機
cap = cv2.VideoCapture(0)  # 0 代表筆電的內建攝影機

# 表情標籤
emotions = ['開心', '生氣', '難過', '驚訝']

def process_frame(frame):
    # 使用YOLOv8進行偵測
    results = model(frame)
    
    # 處理偵測結果
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # 取得邊界框座標
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # 取得信心分數
            conf = float(box.conf[0])
            
            # 取得類別
            cls = int(box.cls[0])
            
            # 在畫面上繪製邊界框和標籤
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f'{emotions[cls]} {conf:.2f}'
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return frame

def main():
    while True:
        # 讀取視訊畫面
        ret, frame = cap.read()
        if not ret:
            break
            
        # 處理畫面
        processed_frame = process_frame(frame)
        
        # 顯示結果
        cv2.imshow('表情偵測', processed_frame)
        
        # 按'q'退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # 釋放資源
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main() 