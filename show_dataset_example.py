from yolov8 import get_prepared_dataset, dataset
from yolov8.utils.plot import plot_image
import numpy as np

batch_size = 4
n_max_bboxes = 10
input_shape = [640, 640, 3]

data_train = get_prepared_dataset(
    data_dir="dataset",
    training="train", 
    batch_size=batch_size,
    n_max_bboxes=n_max_bboxes,
    input_shape=tuple(input_shape)
)


images, labels, gt_masks = next(iter(data_train.take(1)))
plot_image(images, labels, gt_masks, xywh=True) # xuất ảnh với định dạng đầu vào là xywh.