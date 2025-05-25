import cv2
import torch
from ultralytics import YOLO
import time


def detect_objects_webcam(model_path='yolov8x.pt'):
    """
    使用YOLOv8x模型進行即時視訊物件偵測，針對RTX 4060筆電最佳化
    """
    try:
        # 判斷是否有CUDA可用，選擇運行裝置
        if torch.cuda.is_available():
            device = 'cuda'
            # 啟用cuDNN自動最佳化
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True
            torch.cuda.empty_cache()
        else:
            device = 'cpu'
        print(f"使用設備: {device}")
        if device == 'cuda':
            print(f"GPU型號: {torch.cuda.get_device_name(0)}")

        # 載入YOLOv8模型
        print("正在載入模型...")
        model = YOLO(model_path)
        model.to(device)  # 模型搬到指定裝置
        model.eval()  # 設定為推論模式
        print("模型載入完成！")

        # 開啟視訊鏡頭
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("無法開啟視訊鏡頭")
            return

        # 降低解析度以增加FPS
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

        print("程式已啟動，按 'q' 鍵退出")

        prev_time = time.time()  # 上次時間，用來計算FPS
        frame_count = 0  # 幀計數
        fps = 0  # 每秒幀數

        while True:
            ret, frame = cap.read()  # 讀取影像
            if not ret:
                print("無法讀取視訊幀")
                break

            frame = cv2.flip(frame, 1)  # 鏡像效果

            frame = frame[:, :, :3]  # 只保留BGR三通道

            # 禁用梯度以加速推論
            with torch.no_grad():
                results = model(frame, conf=0.5, device=device, verbose=False)  # 推論

            # 計算FPS，每20幀更新一次
            frame_count += 1
            if frame_count >= 20:
                now = time.time()
                fps = frame_count / (now - prev_time)
                prev_time = now
                frame_count = 0

            # 繪製推論結果
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    conf = float(box.conf[0])  # 置信度
                    cls = int(box.cls[0])  # 類別索引
                    class_name = result.names[cls]  # 類別名稱

                    if conf < 0.5:  # 忽略低置信度
                        continue

                    color = (0, int(255 * conf), 0)  # 綠色深淺代表置信度
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)  # 畫框
                    label = f'{class_name} {conf:.2f}'  # 顯示文字
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # 顯示FPS與設備
            cv2.putText(frame, f'FPS: {fps:.1f} | Device: {device}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow('Real-time Object Detection', frame)  # 顯示畫面

            if cv2.waitKey(1) & 0xFF == ord('q'):  # 按q離開
                print("使用者按下 'q' 鍵，程式結束")
                break

    except KeyboardInterrupt:
        print("\n使用者中斷程式")
    except Exception as e:
        print(f"發生錯誤: {str(e)}")
    finally:
        if 'cap' in locals():  # 釋放視訊資源
            cap.release()
        cv2.destroyAllWindows()
        print("程式已結束")


def detect_objects_image(image_path, model_path='yolov8x.pt'):
    """
    使用YOLOv8模型進行圖像物件偵測

    參數:
        image_path: 輸入圖像的路徑
        model_path: YOLOv8模型的路徑
    """
    try:
        # 設定裝置
        if torch.cuda.is_available():
            device = 'cuda'
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True
            torch.cuda.empty_cache()
            torch.cuda.set_per_process_memory_fraction(0.8)
        else:
            device = 'cpu'
        print(f"使用設備: {device}")
        if device == 'cuda':
            print(f"GPU型號: {torch.cuda.get_device_name(0)}")

        # 載入模型
        print("正在載入模型...")
        model = YOLO(model_path)
        model.to(device)
        model.eval()
        print("模型載入完成！")

        # 讀取圖像
        image = cv2.imread(image_path)
        if image is None:
            print("無法讀取圖像")
            return

        # 預測
        with torch.no_grad():
            results = model(image, conf=0.3, device=device)

        # 繪製結果
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                class_name = result.names[cls]
                color = (0, int(255 * conf), 0)
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                label = f'{class_name} {conf:.2f}'
                cv2.putText(image, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # 顯示圖像
        cv2.imshow('Object Detection', image)
        print("按任意鍵關閉視窗")
        cv2.waitKey(0)

    except Exception as e:
        print(f"發生錯誤: {str(e)}")
    finally:
        cv2.destroyAllWindows()  # 關閉所有OpenCV視窗


if __name__ == '__main__':
    detect_objects_webcam()  # 主函式呼叫即時鏡頭偵測
