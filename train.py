# from yolov8.model_yolo_v8 import create_yolo_v8_model
# from yolov8.losses import losses
# import tensorflow as tf
# from yolov8.uitls import *
# import time
# 
# 
# 
# 
# 
# images , labels = create_dataset_cache(True)
# model_yolov8 = create_yolo_v8_model(num_classes=80)
# loss = losses(num_classes=80)
# images = tf.convert_to_tensor(images)
# labels = tf.convert_to_tensor(labels)
# optimizer = tf.keras.optimizers.Adam(0.001)
# 
# data = tf.data.Dataset.from_tensor_slices((images, labels))
# data = data.repeat(20)
# data = data.batch(2)
# #
# # model_yolov8.summary()
# 
# 
# for epoch in range(10):
#     start_time = time.time()
#     for step, (image, label) in enumerate(data):
#         with tf.GradientTape() as tape:
#             y_pred = model_yolov8(image, training=True)
# 
#             # tinh loss
#             total_loss = loss(label, y_pred)
#         grads = tape.gradient(total_loss,model_yolov8.trainable_variables)
#         optimizer.apply_gradients(zip(grads, model_yolov8.trainable_variables))
#     print(f"Epoch {epoch+1} - Thời gian: {time.time() - start_time:.2f}s - Loss: {total_loss:.4f}")
# 
# 
