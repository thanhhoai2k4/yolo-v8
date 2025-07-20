from yolov8.dataset import dataset
import tensorflow as tf



dataset = dataset("dataset", "train")

for image,label in dataset:
    print(image.shape, label.shape)
    break
