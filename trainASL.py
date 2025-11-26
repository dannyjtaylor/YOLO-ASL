from ultralytics import YOLO
#finetune apretrained model for ASL

model = YOLO("yolov8n.pt")
if __name__ == "__main__":
    #train model
    results = model.train(
        data='asl_yolo_dataset/data.yaml',  #Path to your dataset YAML
        epochs=50,                          #number of training epochs
        imgsz=640,                          #image size
        batch=32,                           #batch size, adjust based on GPU
        name='asl_yolo',                    #name for the run
        device=0,                           #GPU ur using (can put "cpu" if no GPU)
    )
    # an epoch is one complete pass through the dataset

    print("Training complete!")
    print(f"Best model saved to: {results.save_dir}/weights/best.pt")