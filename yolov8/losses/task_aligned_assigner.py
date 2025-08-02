import tensorflow as tf

def compute_iou(boxes1, boxes2):
    """
    boxes1: Tensor [N, 4] (x1, y1, x2, y2)
    boxes2: Tensor [M, 4] (x1, y1, x2, y2)
    return: Tensor [N, M] IoU
    """
    boxes1 = tf.cast(boxes1, tf.float32)
    boxes2 = tf.cast(boxes2, tf.float32)

    # Expand dimensions for broadcasting
    boxes1 = tf.expand_dims(boxes1, 1)  # [N, 1, 4]
    boxes2 = tf.expand_dims(boxes2, 0)  # [1, M, 4]

    # Calculate intersection coords
    inter_x1 = tf.maximum(boxes1[..., 0], boxes2[..., 0])
    inter_y1 = tf.maximum(boxes1[..., 1], boxes2[..., 1])
    inter_x2 = tf.minimum(boxes1[..., 2], boxes2[..., 2])
    inter_y2 = tf.minimum(boxes1[..., 3], boxes2[..., 3])

    inter_w = tf.maximum(inter_x2 - inter_x1, 0)
    inter_h = tf.maximum(inter_y2 - inter_y1, 0)
    inter_area = inter_w * inter_h

    # Calculate area of each box
    area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])  # [N, 1]
    area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])  # [1, M]

    union_area = area1 + area2 - inter_area
    iou = inter_area / (union_area + 1e-7)

    return iou  # shape [N, M]



def Label_Assignment(pd_bboxes,
            pd_scores,
            anc_points,
            gt_labels,
            gt_bboxes,
            mask_gt):
    """
    Args:
        pred_bboxe: shape (B, Num_anchors, 4)
        pred_scores: shape (B, Num_acnhors, 1)
        anc_points: shape (Num_acnhors, 2)
        gt_labels: shape (B, N_MAX_BBOXES, 1)
        gt_bboxes: shape (B, N_MAX_BBOXES, 4)
        mask_gt: shape (B, Num_anchors, 1)
    """

    batch = tf.shape(pd_bboxes)[0]

    for i in tf.range(batch):
        scores = pd_scores[i]
        boxes = pd_bboxes[i]

        labelTrue = tf.boolean_mask(tf.expand_dims(gt_labels[i], axis=-1), tf.cast(mask_gt[i], tf.bool))# 10
        boxesTrue = tf.boolean_mask(gt_bboxes[i], tf.squeeze(tf.cast(mask_gt[i], tf.bool),axis=-1), axis=0)

        mask = mask_gt[i] # (10,1)


        ious = compute_iou(boxesTrue, boxes) # shape (Num_object in image, Num_anchors) = (5, 8400)
        class_scores = scores
        class_score = tf.transpose(class_scores)

        aligned_scores = tf.pow(ious,1) * tf.pow(class_score, 1)