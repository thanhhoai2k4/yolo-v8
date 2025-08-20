from yolov8.model.model_yolo_v8 import create_yolo_v8_model


"""
    co 3 cai model :
                d       w       r
        nano:   0.33    0.25    2.0
        small:  0.33    0.5     2.0
        medium:  0.67   0.75    1.5
        large:  1.0     1.0     1.0
        X:      1.0     1.25    1.0
    
"""
INPUT_SHAPE = [640, 640, 3]
NUM_CLASSES = 1



model_nano = create_yolo_v8_model(INPUT_SHAPE, NUM_CLASSES, 0.33, 0.5)
model_nano.summary()