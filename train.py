import time
from yolov8 import dataset
from yolov8.model.model_yolo_v8 import create_yolo_v8_model
from yolov8.losses.losses import losses
from yolov8.dataset.dataset import get_prepared_dataset
import tensorflow as tf

# Parameter
# -----------------------------------
batch_size = 2
num_classes = 1
epochs = 10
input_shape = [640,640,3]
learning_rate = 0.001
n_max_bboxes = 10
# ------------------------------------

# Load dữ liệu - chỉ cần 1 dòng!
data_train = get_prepared_dataset(
    data_dir="dataset",
    training="train", 
    batch_size=batch_size,
    n_max_bboxes=n_max_bboxes,
    input_shape=tuple(input_shape)
)

loss_fn = losses(num_classes=1)# tinh loss
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate) # cap nhat trong so

model =create_yolo_v8_model(input_shape = input_shape, num_classes=num_classes)

# huan luyen
# -------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------
for epoch in tf.range(epochs):

    print(f"\n--- Bắt đầu Epoch {epoch + 1}/{epochs} ---")
    start_time_epoch = time.time()
    total_loss, num_batches = 0, 0
    for step, (images, labels, gt_masks) in enumerate(data_train):
        with tf.GradientTape() as tape:
            predictions = model(images, training=True)

            loss_value = loss_fn(images, labels, gt_masks, predictions)

        # tinh gradient vao nhung bien kha bien
        grads = tape.gradient(loss_value, model.trainable_weights)

        #cap nhat trong so
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        total_loss += loss_value
        num_batches += 1


    print("loss: ",total_loss)

# -----------------------------------------------------------------------------------------