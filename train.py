import time
from tqdm import tqdm
from yolov8.model.model_yolo_v8 import create_yolo_v8_model
from yolov8.losses.losses import losses
from yolov8.dataset.dataset import get_prepared_dataset
import tensorflow as tf
from yolov8.config import BATCH_SIZE, NUM_CLASSES, EPOCHS, INPUT_SHAPE, LEARNING_RATE, N_MAX_Bboxes, Total_Train, Total_Val

from rich.console import Console
from rich.live import Live
from rich.table import Table


# Load dữ liệu - chỉ cần 1 dòng!
# get du lieu training
data_train = get_prepared_dataset(
    data_dir    ="dataset",
    training    ="train",
    batch_size  =BATCH_SIZE,
    n_max_bboxes=N_MAX_Bboxes,
    input_shape =tuple(INPUT_SHAPE)
)

data_val = get_prepared_dataset(
    data_dir ="dataset",
    training ="val",
    batch_size =BATCH_SIZE,
    n_max_bboxes=N_MAX_Bboxes,
    input_shape =tuple(INPUT_SHAPE)
)



# khai bao ham loss
loss_fn = losses(num_classes=NUM_CLASSES)
optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE) # cap nhat trong so
model =create_yolo_v8_model(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES)



# ---------------------------------------------------------------------------------------

@tf.function
def train_step(images, labels, gt_masks):
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss_value = loss_fn(images, labels, gt_masks, predictions)
    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))

    # đảm bảo trả scalar float32
    return tf.reduce_mean(loss_value)

# for data_val
@tf.function
def val_step(images, labels, gt_masks):
    predictions = model(images, training=False) # training=False
    loss_value = loss_fn(images, labels, gt_masks, predictions)
    return tf.reduce_mean(loss_value)

def train(data_train, model):

    best_val_loss = float('inf')
    for epoch in tf.range(EPOCHS):

        print(f"\n--- Bắt đầu Epoch {epoch + 1}/{EPOCHS} ---")
        # train
        total_loss, num_batches = 0, 0
        for images, labels, gt_masks in data_train:
            loss_value = train_step(images, labels, gt_masks)
            total_loss += float(loss_value.numpy())
            num_batches += 1
            avg_train = total_loss / num_batches if num_batches > 0 else 0.0

        # val
        total_val_loss, num_val_batches = 0, 0
        for images, labels, gt_masks in data_val:
            val_loss_value = val_step(images, labels, gt_masks)
            total_val_loss += float(val_loss_value.numpy())
            num_val_batches += 1
            avg_val = total_val_loss / num_val_batches if num_val_batches > 0 else 0.0

        # save model
        if best_val_loss > avg_val:
            model.save_weights("my_weights.weights.h5")    # lưu định dạng weight HDF5
            # update value loss
            best_val_loss = avg_val

        print("loss: {}".format(avg_train))
        print("val_loss: {}".format(avg_val))



# -----------------------------------------------------------------------------------------
# Training model
train(data_train, model)