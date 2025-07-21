import time
from yolov8 import dataset
from yolov8 import create_yolo_v8_model
from yolov8 import losses
import tensorflow as tf

# Parameter
# -----------------------------------
batch_size = 2
num_classes = 1
epochs = 10
input_shape = [640,640,3]
learning_rate = 0.001
# ------------------------------------

# load du lieu
# --------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------
data_train = tf.data.Dataset.from_generator(
    lambda: dataset("dataset"),
    output_signature =(
        tf.TensorSpec(shape=(640, 640, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(None, 5), dtype=tf.float32),
    )
)
data_train = data_train.cache()
data_train = data_train.shuffle(100)
data_train = data_train.padded_batch(
    batch_size = batch_size,
    padded_shapes=([640, 640, 3], [None, 5]),
    padding_values = (tf.constant(0.0, dtype=tf.float32), tf.constant(-1.0, dtype=tf.float32)),
    drop_remainder=True
)
data_train = data_train.prefetch(tf.data.experimental.AUTOTUNE)
# --------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------


loss_fn = losses(num_classes=num_classes)# tinh loss
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate) # cap nhat trong so


model =create_yolo_v8_model(input_shape = input_shape, num_classes=num_classes)




# huan luyen
# -------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------
for epoch in tf.range(epochs):

    print(f"\n--- Bắt đầu Epoch {epoch + 1}/{epochs} ---")
    start_time_epoch = time.time()
    total_loss, num_batches = 0, 0
    for step, (images, labels) in enumerate(data_train):
        with tf.GradientTape() as tape:
            predictions = model(images, training=True)

            loss_value = loss_fn(labels, predictions)

        # tinh gradient vao nhung bien kha bien
        grads = tape.gradient(loss_value, model.trainable_weights)

        #cap nhat trong so
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        total_loss += loss_value
        num_batches += 1

# -----------------------------------------------------------------------------------------