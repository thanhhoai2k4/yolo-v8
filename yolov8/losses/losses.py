import tensorflow as tf
from yolov8.losses.task_aligned_assigner import task_aligned_assigner
import math


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



def losses(num_classes=1, weight = [5.0, 1.0, 0.5, 1.0]):
    tal = task_aligned_assigner(num_classes=num_classes)
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=False, reduction=tf.keras.losses.Reduction.NONE)

    @tf.function
    def compute_loss(images: tf.Tensor, labels: tf.Tensor, gt_masks: tf.Tensor, y_pred):
        """
            Tinh toan loss cho model yolo v8.

            Args:
                images: Tensor đầu vào thử nghiệm. shape (B, H, W, 3)
                labels: Tensor đại điện cho ids, x,y,w,h. shape (B, N_MAX_BBOXES, 5)
                gt_masks: Tensor đại diện cho nó có phải là dữ liệu thật sự hay ko hay chỉ là padding. shape (B, N_MAX_BBOXES, 1)
                y_pred: danh sách 6 phần tử Tensor. lần lước là đầu ra box và class theo diện tích giảm dần.
            Returns:
                Loss.
        """

        # giai ma dau ra cua du doan
        box_pred_p3, cls_pred_p3, box_pred_p4, cls_pred_p4, box_pred_p5, cls_pred_p5 = y_pred
        
        # gop tat ca box pred
        all_box_pred = [box_pred_p3, box_pred_p4, box_pred_p5]
        # gop tat ca class pred
        all_cls_preds = [cls_pred_p3, cls_pred_p4, cls_pred_p5]

        # ----------------------------------------------------------------------------------------
        reshape_box_preds = []
        for box_pred in all_box_pred:
            batch_size = tf.shape(box_pred)[0] # batch_size
            # reshape: (B, H, W,  4*reg_max) =) (B, H*W, 4*reg_max)
            reshaped = tf.reshape(box_pred, shape=(batch_size, -1, box_pred.shape[-1]))
            reshape_box_preds.append(reshaped)
        all_box_preds_concat = tf.concat(reshape_box_preds, axis=1) # shape (B, 8400, 64)

        # --------------------------------------------------------------------------------
        reshape_cls_preds = []
        for cls_pred in all_cls_preds:
            batch_size = tf.shape(cls_pred)[0]
            # Reshape: (batch, H, W, num_classes) -> (batch, H*W, num_classes)
            reshaped = tf.reshape(cls_pred, [batch_size, -1, cls_pred.shape[-1]])
            reshape_cls_preds.append(reshaped)
        all_cls_preds_concat = tf.concat(reshape_cls_preds, axis=1) # shape: (B, 8400, 1) chua duoc ma hoa
        all_cls_preds_concat = tf.nn.sigmoid(all_cls_preds_concat) # shape: (B, 8400, 1) da duoc ma hoa
        #--------------------------------------------------------------------------------------

        # giai ma boxes su that
        boxes_True = labels[..., 1:] * 640
        boxes_True = converbox(boxes_True, True) # chuyen doi xywh sang xyxy
        class_id_true = labels[..., 0] 



        # mỗi dòng là 4 phần tử đại diện cho khoảng cách tình từ tâm tương ứng đến 4 tọa độ (top - left) và (right - bottom)
        #                   ****************-----------------
        #                   *                               |
        #                   *              . (x_c, y_c)     *      x_c : x anchros
        #                   |                               *      y_c : y anchors
        #                   ---------------******************
        # khoang cach tu tâm tương ứng đến 4 tọa độ (top - left) và (right - bottom)
        decoded_boxes = decode_regression_to_boxes(all_box_preds_concat) # day la box xyxy da duoc ma hoa shape (B, num_acnhors, 4)
        
        # all_anchors: shape(8400,2)  all_strides : shape(8400,)
        all_anchors, all_strides = get_anchors(image_shape=images.shape[1:3])

        # chuyển anchors từ tọa độ tương đối sang tọa độ tuyệt đối.
        anchor_point = all_anchors * all_strides[:,None] # shape: (8400,2) notes: 8400 còn được gọi là n_anchors.

        pred_bboxes_xyxy = dist2bbox(decoded_boxes, anchor_point) # Tọa độ thực của pred_box shape: (B, 8400, 4)

        bbox_labels, class_labels, gt_box_matches_per_anchor = tal(scores = all_cls_preds_concat,
                                                                   decode_bboxes = pred_bboxes_xyxy,
                                                                   anchors = tf.keras.ops.tile(
                                                                       tf.keras.ops.expand_dims(anchor_point, axis=0),
                                                                       [labels.shape[0], 1, 1]),
                                                                   gt_labels = class_id_true,
                                                                   gt_bboxes = boxes_True,
                                                                   gt_mask = gt_masks
                                                                   )

        fg_mask = (gt_box_matches_per_anchor > 0)

        loss_cls_positive = tf.keras.ops.cond(
            tf.reduce_sum(tf.cast(fg_mask, tf.float32)) > 0,
            lambda: tf.keras.ops.sum(bce(
                tf.boolean_mask(class_labels, fg_mask),
                tf.boolean_mask(all_cls_preds_concat, fg_mask))
            ),
            lambda: tf.keras.ops.sum(0* class_labels)
        )


        boxes_loss = tf.keras.ops.cond(
            tf.reduce_sum(tf.cast(fg_mask, tf.float32)) > 0,
            lambda: tf.keras.ops.sum(1 - compute_ciou(
                tf.boolean_mask(pred_bboxes_xyxy, fg_mask),
                tf.boolean_mask(bbox_labels, fg_mask))
            ),
            lambda: tf.keras.ops.sum(0.0*pred_bboxes_xyxy)
        )

        loss_cls_negative = tf.keras.ops.sum(bce(
            tf.boolean_mask(class_labels, ~fg_mask),
            tf.boolean_mask(all_cls_preds_concat, ~fg_mask)
        ))

        # dfl loss # tich hop dfl loss vao
        loss_dfl = 0.0




        total_loss = weight[1]*loss_cls_positive + weight[0]*boxes_loss + weight[2]*loss_cls_negative + weight[3]*loss_dfl
        return total_loss



    return compute_loss


def compute_ciou(boxes1, boxes2):

    x_min1, y_min1, x_max1, y_max1 = tf.keras.ops.split(boxes1[..., :4], 4, axis=-1)
    x_min2, y_min2, x_max2, y_max2 = tf.keras.ops.split(boxes2[..., :4], 4, axis=-1)

    width_1 = x_max1 - x_min1
    height_1 = y_max1 - y_min1 + 1e-5
    width_2 = x_max2 - x_min2
    height_2 = y_max2 - y_min2 + 1e-5

    intersection_area = tf.keras.ops.maximum(
        tf.keras.ops.minimum(x_max1, x_max2) - tf.keras.ops.maximum(x_min1, x_min2), 0
    ) * tf.keras.ops.maximum(
        tf.keras.ops.minimum(y_max1, y_max2) - tf.keras.ops.maximum(y_min1, y_min2), 0
    )
    union_area = (
        width_1 * height_1
        + width_2 * height_2
        - intersection_area
        + 1e-5
    )
    iou = tf.keras.ops.squeeze(
        tf.keras.ops.divide(intersection_area, union_area + 1e-5),
        axis=-1,
    )

    convex_width = tf.keras.ops.maximum(x_max1, x_max2) - tf.keras.ops.minimum(x_min1, x_min2)
    convex_height = tf.keras.ops.maximum(y_max1, y_max2) - tf.keras.ops.minimum(y_min1, y_min2)
    convex_diagonal_squared = tf.keras.ops.squeeze(
        convex_width**2 + convex_height**2 + tf.keras.backend.epsilon(),
        axis=-1,
    )
    centers_distance_squared = tf.keras.ops.squeeze(
        ((x_min1 + x_max1) / 2 - (x_min2 + x_max2) / 2) ** 2
        + ((y_min1 + y_max1) / 2 - (y_min2 + y_max2) / 2) ** 2,
        axis=-1,
    )

    v = tf.keras.ops.squeeze(
        tf.keras.ops.power(
            (4 / math.pi**2)
            * (tf.keras.ops.arctan(width_2 / height_2) - tf.keras.ops.arctan(width_1 / height_1)),
            2,
        ),
        axis=-1,
    )
    alpha = v / (v - iou + (1 + tf.keras.backend.epsilon()))

    return iou - (
        centers_distance_squared / convex_diagonal_squared + v * alpha
    )


def converbox(boxes,xyxy=True):
    """
        chuyen doi dinh dang cua hop gioi han.

        Args:
            - boxes: co shape -1,4
            - xyxy: xet xem can chuyen doi dinh dang gi

    """
    if xyxy:
        x,y,w,h = tf.split(boxes, 4, axis=-1)

        x1 = x - w/2
        y1 = y - h/2
        x2 = x + w/2
        y2 = y + h/2

        my_result = tf.concat([x1,y1, x2, y2], axis=-1)
    else:
        x1, y1, x2, y2 = tf.split(boxes, 4, axis = -1)

        x = (x1 + x2)/2
        y = (y1 + y2)/2
        w = x2 - x1
        h = y2 - y1

        my_result = tf.concat([x,y,w,h], axis=-1)
    return my_result