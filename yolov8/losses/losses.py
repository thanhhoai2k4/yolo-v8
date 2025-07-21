import tensorflow as tf
import numpy as np
from yolov8.losses.task_aligned_assigner import Label_Assignment

np.random.seed(42)

BOX_REGRESSION_CHANNELS = 64
def decode_regression_to_boxes(preds):
    """
        giai ma ket qua cua model cho box

        tra ve
    """
    preds_bbox = tf.keras.layers.Reshape((-1,4,BOX_REGRESSION_CHANNELS // 4)) (
        preds
    ) # batch, num_cell, num_cell, 4, 16
    preds_bbox = tf.nn.softmax(preds_bbox,axis=-1) * tf.keras.ops.arange(start=0, stop=BOX_REGRESSION_CHANNELS//4, dtype=tf.float32)
    return tf.keras.ops.sum(preds_bbox, axis=-1)

def get_anchors(
        image_shape,
        strides=[8,16,32],
        base_anchors=[0.5, 0.5],
):
    """

    """
    base_anchors = tf.keras.ops.array(base_anchors, dtype=tf.float32)

    all_anchors = []
    all_strides = []

    for stride in strides:

        # image_shape = [height, width]
        hh_centers = tf.keras.ops.arange(start=0, stop=image_shape[0], step=stride, dtype=tf.float32)
        ww_centers = tf.keras.ops.arange(start=0, stop=image_shape[1], step=stride, dtype=tf.float32)

        ww_grid, hh_grid = tf.keras.ops.meshgrid(ww_centers, hh_centers)
        grid = tf.keras.ops.cast(
            tf.keras.ops.reshape(tf.keras.ops.stack([hh_grid, ww_grid], 2), [-1, 1, 2]),
            "float32",
        )

        anchors = (
                tf.keras.ops.expand_dims(
                    base_anchors * tf.keras.ops.array([stride, stride], "float32"), 0
                )
                + grid
        )
        anchors = tf.keras.ops.reshape(anchors, [-1,2])
        temp_stride = tf.keras.ops.repeat([stride], anchors.shape[0])
        all_anchors.append(anchors)
        all_strides.append(temp_stride)

    all_anchors = tf.keras.ops.cast(tf.keras.ops.concatenate(all_anchors, axis=0), "float32")
    all_strides = tf.keras.ops.cast(tf.keras.ops.concatenate(all_strides, axis=0), "float32")

    #chuan hoa
    all_anchors = all_anchors / all_strides[:, None]

    # vi su ly trong anh thi toa y,x => hoang doi vi tri
    all_anchors = tf.keras.ops.concatenate(
        [all_anchors[:, 1, None], all_anchors[:, 0, None]], axis=-1
    )

    return all_anchors, all_strides

def dist2bbox(distance, anchor_points):
    """Decodes distance predictions into xyxy boxes.

    Input left / top / right / bottom predictions are transformed into xyxy box
    predictions based on anchor points.

    The resulting xyxy predictions must be scaled by the stride of their
    corresponding anchor points to yield an absolute xyxy box.
    """
    left_top, right_bottom = tf.keras.ops.split(distance, 2, axis=-1)
    x1y1 = anchor_points - left_top
    x2y2 = anchor_points + right_bottom
    return tf.keras.ops.concatenate((x1y1, x2y2), axis=-1)  # xyxy bbox





def losses(num_classes=1):
    def compute_loss(y_true, y_pred):



    return compute_loss
