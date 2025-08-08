import tensorflow as tf
from yolov8.losses.task_aligned_assigner import task_aligned_assigner
from yolov8.losses.losses import get_anchors


batch_size      = 1
num_anchors     = 16
num_classes     = 4
num_gt_boxes    = 10



scores = tf.constant(
    [
        [
            [0.1, 0.2, 0.3, 0.4],
            [0.1, 0.2, 0.3, 0.4],
            [0.1, 0.2, 0.3, 0.4],
            [0.1, 0.2, 0.3, 0.4],
            [0.1, 0.2, 0.3, 0.4],
            [0.1, 0.2, 0.3, 0.4],
            [0.1, 0.2, 0.3, 0.4],
            [0.1, 0.2, 0.3, 0.4],
            [0.1, 0.2, 0.3, 0.4],
            [0.1, 0.2, 0.3, 0.4],
            [0.1, 0.2, 0.3, 0.4],
            [0.1, 0.2, 0.3, 0.4],
            [0.1, 0.2, 0.3, 0.4],
            [0.1, 0.2, 0.3, 0.4],
            [0.1, 0.2, 0.3, 0.4],
            [0.1, 0.2, 0.3, 0.4],
        ]
    ], dtype=tf.float32
)


decode_bboxes = tf.constant(
    [[
        [50, 50, 200, 200],
        [60, 50, 300, 300],
        [150, 50, 500, 500],
        [200, 200, 400, 400],
        [0, 0, 200, 200],
        [1, 1, 200, 200],
        [2, 2, 200, 200],
        [3, 4, 200, 200],
        [5, 6, 200, 200],
        [7, 8, 200, 200],
        [2, 2, 200, 200],
        [1, 1, 200, 200],
        [2, 2, 200, 200],
        [3, 3, 200, 200],
        [4, 5, 200, 200],
        [1, 2, 200, 200],
    ]],
    dtype=tf.float32,
)

anchors,strides = get_anchors([640,640], [160])
anchors = anchors * strides[:,None]
anchors = anchors[None,:]



gt_labels = tf.constant(
    [
        [0, 1, 2, -1, -1, -1, -1, -1, -1, -1]
    ], dtype=tf.int32, shape=[batch_size, num_gt_boxes]
)



gt_bboxes = tf.constant(
    [
        [10, 10 , 150, 150],
        [150, 150, 200, 200],
        [-1, -1, -1, -1],
        [-1, -1, -1, -1],
        [-1, -1, -1, -1],
        [-1, -1, -1, -1],
        [-1, -1, -1, -1],
        [-1, -1, -1, -1],
        [-1, -1, -1, -1],
        [-1, -1, -1, -1],
    ], dtype=tf.float32, shape=[batch_size, num_gt_boxes, 4]
)


gt_mask = tf.constant(
    [True, True, False, False, False, False, False, False, False, False], dtype=tf.bool, shape=[batch_size, num_gt_boxes]
)


#
tal = task_aligned_assigner(num_classes = num_classes)

bbox_labels, class_labels, gt_box_matches_per_anchor = tal(
    scores = scores,
    decode_bboxes = decode_bboxes,
    anchors = anchors,
    gt_labels = gt_labels,
    gt_bboxes = gt_bboxes,
    gt_mask = gt_mask

)

print(bbox_labels)
print(class_labels)
print(gt_box_matches_per_anchor)

