import ultralytics
import torch


def main():
    print(torch.cuda.is_available())
    print(torch.cuda.current_device())

    model= ultralytics.YOLO("yolo11n.pt")


    # model.export(format="onnx")
    # train=model.train(
    #     data="datasets\data.yaml",
    #     epochs=40,
    #     imgsz=640,
    #     device=0,
    #     val=False,
    #     batch=-1
        
    # )

    # path = model.export(format="onnx") 

    # # Проверяем точность модели на валидационном наборе данных
    # metrics = model.val()

    # # Выводим метрики точности
    # print(f"Precision: {metrics.precision}")
    # print(f"Recall: {metrics.recall}")
    # print(f"mAP50: {metrics.map50}")
    # print(f"mAP50-95: {metrics.map}")


    # source = "https://www.youtube.com/watch?v=Zi9PwNb3I_I"
    # source1 = "https://www.youtube.com/watch?v=RukFNgmPLVI"


    # results = model.predict(source=source1, show=True) 

if __name__ == "__main__":
    
    main()
