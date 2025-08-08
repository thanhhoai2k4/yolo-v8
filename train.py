import time
from tqdm import tqdm
from yolov8.model.model_yolo_v8 import create_yolo_v8_model
from yolov8.losses.losses import losses
from yolov8.dataset.dataset import get_prepared_dataset
import tensorflow as tf
from yolov8.config import BATCH_SIZE, NUM_CLASSES, EPOCHS, INPUT_SHAPE, LEARNING_RATE, N_MAX_Bboxes, Total_Train


# Load dữ liệu - chỉ cần 1 dòng!
data_train = get_prepared_dataset(
    data_dir    ="dataset",
    training    ="train",
    batch_size  =BATCH_SIZE,
    n_max_bboxes=N_MAX_Bboxes,
    input_shape =tuple(INPUT_SHAPE)
)

# khai bao ham loss
loss_fn = losses(num_classes=NUM_CLASSES)
optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE) # cap nhat trong so
model =create_yolo_v8_model(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES)

# huan luyen
# ---------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------
for epoch in tf.range(EPOCHS):
    print(f"\n--- Bắt đầu Epoch {epoch + 1}/{EPOCHS} ---")
    start_time_epoch = time.time()
    total_loss, num_batches = 0, 0

    pbar = tqdm(enumerate(data_train), total=Total_Train // BATCH_SIZE, desc=f"Epoch {epoch + 1}/{EPOCHS}")
    for step, (images, labels, gt_masks) in pbar:
        with tf.GradientTape() as tape:

            predictions = model(images, training=True)
            loss_value = loss_fn(images, labels, gt_masks, predictions)

        # tinh gradient vao nhung bien kha bien
        grads = tape.gradient(loss_value, model.trainable_weights)
        # update weight
        optimizer.apply_gradients(zip(grads, model.trainable_weights))


        total_loss += loss_value.numpy()
        num_batches += 1

        pbar.set_postfix({"loss": total_loss / num_batches})


    print("loss: ",total_loss)

# -----------------------------------------------------------------------------------------