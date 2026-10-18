from ultralytics import YOLO

def main():
    # Загружаем базовые веса YOLOv11 nano
    model = YOLO("yolo11n.pt")

    # Запуск тренировки на GPU
    model.train(
        data="data.yaml",
        epochs=100,
        device=0,        # Видеокарта RTX 3060
        batch=16,        # При 12 ГБ VRAM можно поднять до 32
        workers=4,       # Потоки CPU для чтения диска
        imgsz=640
    )

if __name__ == '__main__':
    main()