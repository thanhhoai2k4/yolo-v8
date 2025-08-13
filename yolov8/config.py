from yolov8.dataset.dataset import getlistfile

BATCH_SIZE = 1
NUM_CLASSES = 1
EPOCHS = 100
INPUT_SHAPE = [640,640,3]
LEARNING_RATE = 0.0001
N_MAX_Bboxes = 10



images_path_train = getlistfile("dataset", "train")
Total_Train = len(images_path_train) //5


image_path_val = getlistfile("dataset", "val")
Total_Val = len(image_path_val) // 5

