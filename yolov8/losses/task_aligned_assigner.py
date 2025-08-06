import tensorflow as tf
import math

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

def is_anchor_center_within_box(anchors, gt_bboxes):
    return tf.keras.ops.all(
        tf.keras.ops.logical_and(
            gt_bboxes[:, :, None, :2] < anchors,
            gt_bboxes[:, :, None, 2:] > anchors,
        ),
        axis=-1,
    )


class task_aligned_assigner(tf.keras.layers.Layer):
    def __init__(
            self,
            num_classes,
            max_anchor_matches : int = 10,
            alpha : float = 0.5,
            beta : float = 6.0,
            epsilon : float = 1e-5,
            **kwargs
                 ):
        super().__init__(**kwargs)
        self.max_anchor_matches = max_anchor_matches
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon

    def assign(
            self,
            scores,
            decode_bboxes,
            anchors,
            gt_labels,
            gt_bboxes,
            gt_mask
    ):
        num_anchors = anchors.shape[1]

        bbox_scores = tf.keras.ops.take_along_axis(
            scores,
            tf.keras.ops.cast(tf.keras.ops.maximum(gt_labels[:, None, :], 0), "int32"),
            axis=-1,
        )# batch, num_anchors, num_gt_bboxes

        bbox_scores = tf.keras.ops.transpose(bbox_scores, (0, 2, 1)) # batch, num_gt_bboxes, num_anchors

        # Overlaps are the IoUs of each predicted box and each GT box.
        # Shape: (batch, num_gt_bboxes, num_anchors)
        overlaps = compute_ciou(
            tf.keras.ops.expand_dims(gt_bboxes, axis=2),
            tf.keras.ops.expand_dims(decode_bboxes, axis=1)
        )

        # batch, num_gt_bboxes, num_anchors
        alignment_metrics = tf.keras.ops.power(bbox_scores, self.alpha) * tf.keras.ops.power(
            overlaps, self.beta
        )
        alignment_metrics = tf.keras.ops.where(gt_mask, alignment_metrics, 0) #


        # anchors inside cell
        # batch_size, num_gt_bboxes, num_anchors
        # boolean tensor
        matching_anchors_in_gt_boxes = is_anchor_center_within_box(
            anchors, gt_bboxes
        )
        alignment_metrics = tf.keras.ops.where(matching_anchors_in_gt_boxes, alignment_metrics, 0)


        # the top-k
        candidate_metrics, candidate_idxs = tf.keras.ops.top_k(
            alignment_metrics, self.max_anchor_matches
        )
        candidate_idxs = tf.keras.ops.where(candidate_metrics > 0, candidate_idxs, -1)



        anchors_matched_gt_box = tf.keras.ops.zeros_like(overlaps)
        for k in range(self.max_anchor_matches):

            onehot = tf.keras.ops.one_hot(
                candidate_idxs[:, k], num_anchors
            )
            anchors_matched_gt_box += onehot

        # remove zero-out( xoa bo chong cheo)
        overlaps *= anchors_matched_gt_box # batch, num_gt_bboxes, num_anchors

        # In cases where one anchor matches to 2 GT boxes, we pick the GT box
        # with the highest overlap as a max.
        gt_box_matches_per_anchor = tf.keras.ops.argmax(overlaps, axis=1)
        gt_box_matches_per_anchor_mask = tf.keras.ops.max(overlaps, axis=1) > 0




        gt_box_matches_per_anchor = tf.keras.ops.cast(gt_box_matches_per_anchor, "int32")

        # We select the GT boxes and labels that correspond to anchor matches.
        bbox_labels = tf.keras.ops.take_along_axis(
            gt_bboxes, gt_box_matches_per_anchor[:, :, None], axis=1
        )

        bbox_labels = tf.keras.ops.where(
            gt_box_matches_per_anchor_mask[:, :, None], bbox_labels, -1
        )
        class_labels = tf.keras.ops.take_along_axis(
            gt_labels, gt_box_matches_per_anchor, axis=1
        )
        class_labels = tf.keras.ops.where(
            gt_box_matches_per_anchor_mask, class_labels, -1
        )

        class_labels = tf.keras.ops.one_hot(
            tf.keras.ops.cast(class_labels, "int32"), self.num_classes
        )

        # Finally, we normalize an anchor's class labels based on the relative
        # strength of the anchors match with the corresponding GT box.
        alignment_metrics *= anchors_matched_gt_box
        max_alignment_per_gt_box = tf.keras.ops.max(
            alignment_metrics, axis=-1, keepdims=True
        )
        max_overlap_per_gt_box = tf.keras.ops.max(overlaps, axis=-1, keepdims=True)

        normalized_alignment_metrics = tf.keras.ops.max(
            alignment_metrics
            * max_overlap_per_gt_box
            / (max_alignment_per_gt_box + self.epsilon),
            axis=-2,
        )
        class_labels *= normalized_alignment_metrics[:, :, None]

        # On TF backend, the final "4" becomes a dynamic shape so we include
        # this to force it to a static shape of 4. This does not actually
        # reshape the Tensor.
        bbox_labels = tf.keras.ops.reshape(bbox_labels, (-1, num_anchors, 4))
        return (
            tf.keras.ops.stop_gradient(bbox_labels),
            tf.keras.ops.stop_gradient(class_labels),
            tf.keras.ops.stop_gradient(
                tf.keras.ops.cast(gt_box_matches_per_anchor > -1, "float32")
            ),
        )




    def call(
            self,
            scores,
            decode_bboxes,
            anchors,
            gt_labels,
            gt_bboxes,
            gt_mask
    ):

        """
            Args:
                - scores: a float tensor with shape [batch_size, num_anchors, num_classes].
                - decode_bboxes: a float tensor with shape [batch_size, num_anchors, 4]. representing predicted bboxes for each anchor.
                - anchors: a float tensor with shape [batch_size, num_anchors, 2]. representing the xy of the center of the anchor.
                - gt_labels: a int tensor with shape [batch_size, num_gt_bboxes].
                - gt_bboxes: a float tensor with shape [batch_size, num_gt_bboxes, 4].
                - gt_mask: A Boolean Tensor of shape (batch_size, num_gt_boxes)
                            representing whether a box in `gt_bboxes` is a real box or a
                            non-box that exists due to padding.

                các ảnh không có số lượng box = num_gt_bboxes: thường là 10 (Khac với max_anchor_matches: đây là số lượng được lấy mẫu tích cực)
                Vì thế các ma trận sẽ được đệm 1 giá trị nhất định thường là -1 vào trong các ma trận và trong gt_mask: thì là true false.
                Từ đó lọc ra các boxes được tính toán để điều chỉnh cho model học

        """

        max_num_boxes = tf.keras.ops.shape(
            gt_bboxes
        )[1] # number of bounding box per an image

        return tf.keras.ops.cond(
            tf.keras.ops.array(max_num_boxes) > 0,
            # True: have an object in image
            lambda: self.assign(scores, decode_bboxes, anchors, gt_labels, gt_bboxes, gt_mask),
            # False : haven't an object
            lambda: (
                tf.keras.ops.zeros_like(decode_bboxes),
                tf.keras.ops.zeros_like(scores),
                tf.keras.ops.zeros_like(scores[..., 0])
            )
        )