from yolov8 import get_prepared_dataset, dataset
from yolov8.utils.plot import plot_image

data_train = get_prepared_dataset(
    data_dir="dataset",
    training="train", 
    batch_size=4,
    n_max_bboxes=10,
    input_shape=[640,640,3]
)




for images, labels, gt_masks in data_train.take(1):
    plot_image(images, labels, gt_masks, xywh=True)
    break