import cv2
import mss
import numpy as np
from ultralytics import YOLO
from datetime import datetime
import os

# Загружаем модель YOLO (используем предобученную или свою)
model = YOLO("C:\\Users\\Katana\\Desktop\\yolo_motion_prediction\\runs\\detect\\train7\\weights\\best.onnx")  # Можно заменить на кастомную модель

# Определяем параметры захвата экрана
monitor = {"top": (mss.mss().monitors[1]["height"] - 640) // 2, "left": (mss.mss().monitors[1]["width"] - 640) // 2, "width": 640, "height": 640}


# Создаем необходимые директории, если они не существуют
os.makedirs("images", exist_ok=True)
os.makedirs("labels", exist_ok=True)

with mss.mss() as sct:
    while True:
        # Захватываем экран
        screenshot = sct.grab(monitor)

        # Преобразуем изображение в numpy-формат
        frame = np.array(screenshot)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)  # Убираем альфа-канал

        # Применяем YOLO для детекции
        results = model(frame)

        # Отображаем результаты
        annotated_frame = results[0].plot()  # Отрисовка рамок и меток
        cv2.imshow("YOLO Screen Detection", annotated_frame)

        # Сохраняем изображение и данные рамок при нажатии клавиши 's'
        if cv2.waitKey(1) & 0xFF == ord('s'):
            # Генерируем уникальное имя файла с использованием времени
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            image_filename = f"images/screenshot_{timestamp}.png"
            text_filename = f"labels/screenshot_{timestamp}.txt"

            # Сохраняем изображение без рамок
            cv2.imwrite(image_filename, frame)
            print(f"Изображение сохранено как '{image_filename}'")

            # Сохраняем данные рамок в формате YOLO
            img_height, img_width, _ = frame.shape
            with open(text_filename, 'w') as file:
                for result in results:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    for box in boxes:
                        # Нормализуем координаты
                        x_center = (box[0] + box[2]) / 2 / img_width
                        y_center = (box[1] + box[3]) / 2 / img_height
                        width = (box[2] - box[0]) / img_width
                        height = (box[3] - box[1]) / img_height
                        file.write(f"0 {x_center} {y_center} {width} {height}\n")
            print(f"Данные рамок сохранены в '{text_filename}'")

        # Выход по нажатию клавиши 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()
