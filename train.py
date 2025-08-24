from tqdm import tqdm
from yolov8.model.model_yolo_v8 import create_yolo_v8_model
from yolov8.losses.losses import losses
from yolov8.dataset.dataset import get_prepared_dataset
import tensorflow as tf
from yolov8.config import BATCH_SIZE, NUM_CLASSES, EPOCHS, INPUT_SHAPE, LEARNING_RATE, N_MAX_Bboxes, Total_Train, Total_Val
import os

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

model = create_yolo_v8_model(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES, depth_multiple=0.25, width_multiple=0.25)
if os.path.exists("my_weights.weights.h5"):
    model.load_weights("my_weights.weights.h5")
# ---------------------------------------------------------------------------------------

# data training
@tf.function
def train_step(images, labels, gt_masks):
    """
        thuc hien tinh loss va backpropagation voi du lieu training

    """
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
    """
        Thuc hien danh gia du du lieu kiem dinh. ma ko cap nhat bat ki trong so gi o day.
    """
    predictions = model(images, training=False) # training=False
    loss_value = loss_fn(images, labels, gt_masks, predictions)
    return tf.reduce_mean(loss_value)

def train(data_train, model):
    """
        huan luyen model yolo v8

        Arguments:
            - data_trai: tf.data.Dataset day la du lieu voi lan luong la anh va nhan cua anh
                    image , labels (=) 640,640,3   10,5 voi moi hang la id,x,y,w,h o dang chuan hoa
            - model: model yolo v8.
        Returns:

    """
    best_val_loss = float('inf')
    avg_val = float("inf")
    for epoch in tf.range(EPOCHS):

        print(f"\n--- Bắt đầu Epoch {epoch + 1}/{EPOCHS} ---")

        total_loss, num_batches = 0, 0
        train_bar = tqdm(data_train,
                            total = Total_Train,
                            desc=f"Train {epoch+1}/{EPOCHS}",
                            unit="batch")
        for images, labels, gt_masks in train_bar:
            loss_value = train_step(images, labels, gt_masks)
            total_loss += float(loss_value.numpy())
            num_batches += 1
            avg_train = total_loss / num_batches if num_batches > 0 else 0.0
            train_bar.set_postfix({"loss": f"{avg_train:.4f}"})

        total_val_loss, num_val_batches = 0, 0
        val_bar = tqdm(data_val,
                       desc=f"Val {epoch + 1}/{EPOCHS}",
                       unit="batch",
                       total=Total_Val)
        for images, labels, gt_masks in val_bar:
            val_loss_value = val_step(images, labels, gt_masks)
            total_val_loss += float(val_loss_value.numpy())
            num_val_batches += 1
            avg_val = total_val_loss / num_val_batches if num_val_batches > 0 else 0.0
            val_bar.set_postfix({"val_loss": f"{avg_val:.4f}"})

        # save model
        if best_val_loss > avg_val:
            model.save_weights("my_weights.weights.h5")    # lưu định dạng weight HDF5
            best_val_loss = avg_val


# -----------------------------------------------------------------------------------------
# Training model
if __name__ == "__main__":
    # run model
    train(data_train, model)