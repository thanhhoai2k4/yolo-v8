import tensorflow as tf

def xywh2xyxy(boxes):
    """
    Chuyển đổi hộp bao từ định dạng [x_center, y_center, width, height]
    sang [x_min, y_min, x_max, y_max].
    """
    xy = boxes[..., :2]
    wh = boxes[..., 2:]
    x1y1 = xy - wh / 2
    x2y2 = xy + wh / 2
    return tf.keras.ops.concatenate([x1y1, x2y2], axis=-1)

def bboxes_iou(boxes1, boxes2):
    """
    Tính IoU (Intersection over Union) giữa hai tập hợp hộp bao.
    boxes1: (N, 4)
    boxes2: (M, 4)
    Returns: (N, M) ma trận IoU
    """
    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    # Tính tọa độ vùng giao nhau
    intersect_mins = tf.keras.ops.maximum(boxes1[..., None, :2], boxes2[..., :2])
    intersect_maxes = tf.keras.ops.minimum(boxes1[..., None, 2:], boxes2[..., 2:])
    intersect_wh = tf.keras.ops.maximum(intersect_maxes - intersect_mins, 0.0)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]

    union_area = boxes1_area[..., None] + boxes2_area - intersect_area

    # Tính IoU
    iou = intersect_area / (union_area + 1e-7) # Thêm epsilon để tránh chia cho 0
    return iou


def Label_Assignment(pred_bboxes, pred_scores, anchors, y_true, num_classes=80, top_k=10, alpha=1.0, beta=6.0):
    """
    Thực hiện gán nhãn theo phương pháp Task-Aligned Assigner.

    Args:
        pred_bboxes (tf.Tensor): Hộp dự đoán từ mô hình, shape (batch, total_anchors, 4).
        pred_scores (tf.Tensor): Điểm lớp dự đoán, shape (batch, total_anchors, num_classes).
        anchors (tf.Tensor): Tọa độ các anchor point, shape (total_anchors, 2).
        y_true (tf.Tensor): Nhãn thật, shape (batch, N, 5) với 5 là (class_id, x,y,w,h).
        num_classes (int): Số lượng lớp.
        top_k (int): Số lượng ứng viên tốt nhất được chọn cho mỗi đối tượng thật.
        alpha (float): Trọng số cho điểm phân loại trong alignment metric.
        beta (float): Trọng số cho IoU trong alignment metric.

    Returns:
        Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
            - target_bboxes (batch, total_anchors, 4): Hộp bao mục tiêu cho mỗi anchor.
            - target_scores (batch, total_anchors, num_classes): Lớp mục tiêu cho mỗi anchor.
            - fg_mask (batch, total_anchors): Mặt nạ xác định các mẫu dương.
    """
    batch_size = tf.shape(pred_bboxes)[0]
    total_anchors = tf.shape(anchors)[0]

    # Chuẩn bị các tensor đầu ra
    target_bboxes_list = []
    target_scores_list = []
    fg_mask_list = []

    # Xử lý từng ảnh trong batch
    for i in tf.range(batch_size):
        single_pred_bboxes = pred_bboxes[i]  # (total_anchors, 4)
        single_pred_scores = pred_scores[i]  # (total_anchors, num_classes)
        single_y_true = y_true[i]  # (N, 5)

        # 1. Lọc các ground-truth box hợp lệ (loại bỏ padding)
        mask_gt = single_y_true[:, 4] > 0  # Giả sử width > 0 là hợp lệ
        gt_boxes_xywh = tf.boolean_mask(single_y_true[:, 1:], mask_gt)
        gt_class_ids = tf.cast(tf.boolean_mask(single_y_true[:, 0], mask_gt), dtype=tf.int32)

        num_gt = tf.shape(gt_class_ids)[0]
        if num_gt == 0:
            # Nếu không có đối tượng thật, tất cả đều là mẫu âm
            target_bboxes_list.append(tf.zeros((total_anchors, 4)))
            target_scores_list.append(tf.zeros((total_anchors, num_classes)))
            fg_mask_list.append(tf.zeros(total_anchors, dtype=tf.bool))
            continue

        gt_boxes_xyxy = xywh2xyxy(gt_boxes_xywh)

        # 2. Lấy các ứng viên sơ bộ (Preliminary Candidates)
        # Chỉ chọn các anchor có tâm nằm trong ground-truth box
        # anchor_points: (total_anchors, 2), gt_boxes_xyxy: (num_gt, 4)
        # is_in_gt_mask: (total_anchors, num_gt)
        is_in_gt_mask = tf.logical_and(
            tf.logical_and(anchors[:, None, 0] >= gt_boxes_xyxy[:, 0], anchors[:, None, 0] <= gt_boxes_xyxy[:, 2]),
            tf.logical_and(anchors[:, None, 1] >= gt_boxes_xyxy[:, 1], anchors[:, None, 1] <= gt_boxes_xyxy[:, 3])
        )

        # 3. Tính Alignment Metric
        iou_matrix = bboxes_iou(single_pred_bboxes, gt_boxes_xyxy)  # (total_anchors, num_gt)

        # Lấy điểm dự đoán cho đúng lớp ground-truth
        cls_scores = tf.gather(single_pred_scores, gt_class_ids, axis=1)  # (total_anchors, num_gt)

        # Metric = (score^alpha) * (iou^beta)
        alignment_metric = tf.pow(cls_scores, alpha) * tf.pow(iou_matrix, beta)

        # Chỉ giữ lại metric của các ứng viên sơ bộ
        alignment_metric = alignment_metric * tf.cast(is_in_gt_mask, dtype=tf.float32)

        # 4. Chọn top 'k' ứng viên cho mỗi ground-truth
        # (Chuyển vị để tìm top_k trên chiều anchor)
        top_k_metrics, top_k_indices = tf.nn.top_k(tf.transpose(alignment_metric), k=top_k)

        # 5. Xử lý trùng lặp (nếu một anchor được gán cho nhiều gt)
        # Anchor được gán cho gt có IoU cao nhất
        fg_mask_per_image = tf.zeros(total_anchors, dtype=tf.bool)
        assigned_gt_idx = tf.ones(total_anchors, dtype=tf.int32) * -1

        # (num_gt, top_k)
        candidate_gt_idx = tf.broadcast_to(tf.range(num_gt)[:, None], tf.shape(top_k_indices))

        # Sắp xếp các ứng viên theo metric giảm dần để ưu tiên gán các match tốt nhất trước
        flat_top_k_indices = tf.reshape(top_k_indices, [-1])
        flat_candidate_gt_idx = tf.reshape(candidate_gt_idx, [-1])
        flat_top_k_metrics = tf.reshape(top_k_metrics, [-1])

        sorted_indices = tf.argsort(flat_top_k_metrics, direction='DESCENDING')
        flat_top_k_indices = tf.gather(flat_top_k_indices, sorted_indices)
        flat_candidate_gt_idx = tf.gather(flat_candidate_gt_idx, sorted_indices)

        # Lấy các cặp (anchor, gt) duy nhất, giữ lại cặp đầu tiên (tốt nhất)
        unique_anchor_indices, unique_idx = tf.unique(flat_top_k_indices)
        unique_gt_indices = tf.gather(flat_candidate_gt_idx, unique_idx)

        # Gán kết quả
        fg_mask_per_image = tf.tensor_scatter_nd_update(
            fg_mask_per_image,
            unique_anchor_indices[:, None],
            tf.ones_like(unique_anchor_indices, dtype=tf.bool)
        )
        assigned_gt_idx = tf.tensor_scatter_nd_update(
            assigned_gt_idx,
            unique_anchor_indices[:, None],
            unique_gt_indices
        )

        # 6. Tạo các tensor mục tiêu
        target_bboxes_per_image = tf.zeros((total_anchors, 4))
        target_scores_per_image = tf.zeros((total_anchors, num_classes))

        positive_indices = tf.where(fg_mask_per_image)
        positive_gt_idx = tf.gather_nd(assigned_gt_idx, positive_indices)

        positive_target_bboxes = tf.gather(gt_boxes_xyxy, positive_gt_idx)
        positive_target_scores_iou = tf.gather_nd(iou_matrix,
                                                  tf.stack([tf.squeeze(positive_indices, -1), positive_gt_idx],
                                                           axis=-1))
        positive_target_labels = tf.one_hot(tf.gather(gt_class_ids, positive_gt_idx), num_classes)
        positive_target_scores = positive_target_labels * positive_target_scores_iou[:, None]

        target_bboxes_per_image = tf.tensor_scatter_nd_update(target_bboxes_per_image, positive_indices,
                                                              positive_target_bboxes)
        target_scores_per_image = tf.tensor_scatter_nd_update(target_scores_per_image, positive_indices,
                                                              positive_target_scores)

        target_bboxes_list.append(target_bboxes_per_image)
        target_scores_list.append(target_scores_per_image)
        fg_mask_list.append(fg_mask_per_image)

    # Ghép kết quả của cả batch lại
    target_bboxes = tf.stack(target_bboxes_list, axis=0)
    target_scores = tf.stack(target_scores_list, axis=0)
    fg_mask = tf.stack(fg_mask_list, axis=0)

    return target_bboxes, target_scores, fg_mask