import tensorflow as tf
from yolov8.model.model_yolo_v8 import create_yolo_v8_model
from yolov8.config import  *
from yolov8.losses.losses import decode_regression_to_boxes, get_anchors, dist2bbox


def load_model(path: str = "my_weights.weights.h5"):
    """
        load model
        Args:
            path: model path: vd: model.h5
        Returns:
            model: keras model
    """
    # load model
    model =create_yolo_v8_model(input_shape=INPUT_SHAPE,
                                num_classes=NUM_CLASSES,
                                width_multiple=0.25,
                                depth_multiple=0.25)
    model.load_weights(path)
    print("✅ load weights successfully!")
    return model


def load_image(path: str, target_size : [int, int] = [640, 640]):
    """
        read image.

        Args:
            path: image path: vd dataset/test/image/text01.jpg shape: [BATCH, HEIGHT, WIDTH, CHANNEL]
        Returns:
            imgs with shape [1, HEIGHT, WIDTH, CHANNEL]
    """
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, channels=3)  # convert 3 channels 0-255
    img = tf.image.convert_image_dtype(img, tf.float32)  # any to float: 0-1
    height_org, width_org = img.shape[:2] # original height and width
    heiht_after, width_after = target_size

    img = tf.image.resize(img, [heiht_after, width_after])
    ratio_height = heiht_after / height_org
    ratio_width = width_after / width_org
    """ 
        height_affter: 640
        height_org: 320
        
        ratio: 640 : 320 = 2
        
        =) cac gia tri y tai from height_org into height_org: se phai x2
        
        Width tuong ung
    """

    return tf.keras.ops.expand_dims(img, 0), [ratio_height, ratio_width]


def inference(path_model: str="my_weights.weights.h5", path_image: str="dataset/images/test/n02085620_368 - Copy.jpg", target_size : [int, int] = [640, 640], iou_threshold: int = 0.2, confidence_threshold: int = 0.2):
    """
        show Corresponding boxes with an images

    """
    # load model and an image
    model = load_model(path_model) # load model
    image, [height_ratio, width_ratio] = load_image(path_image, target_size) # load image with shape: [1, *target_size, 3]

    # prediction:
    # box_pred_p3: [1, 80,80, 64]
    # cls_pred_p3: [1, 80, 80, 1]
    # box_pred_p4: [1, 40, 40,64]
    # cls_pred_p4: [1, 40, 40, 1]
    # box_pred_p5: [1, 20, 20,64]
    # cls_pred_p5: [1, 20, 20, 1]
    box_pred_p3, cls_pred_p3, box_pred_p4, cls_pred_p4, box_pred_p5, cls_pred_p5 = model.predict(image)


    all_box_pred = [box_pred_p3, box_pred_p4, box_pred_p5]
    reshape_box_preds = []
    for box_pred in all_box_pred:
        batch_size = tf.shape(box_pred)[0]  # batch_size
        # reshape: (B, H, W,  4*reg_max) =) (B, H*W, 4*reg_max)
        reshaped = tf.reshape(box_pred, shape=(batch_size, -1, box_pred.shape[-1]))
        reshape_box_preds.append(reshaped)
    all_box_preds_concat = tf.concat(reshape_box_preds, axis=1)  # shape (B, 8400, 64)

    all_cls_preds = [cls_pred_p3, cls_pred_p4, cls_pred_p5]
    reshape_cls_preds = []
    for cls_pred in all_cls_preds:
        batch_size = tf.shape(cls_pred)[0]
        # Reshape: (batch, H, W, num_classes) -> (batch, H*W, num_classes)
        reshaped = tf.reshape(cls_pred, [batch_size, -1, cls_pred.shape[-1]])
        reshape_cls_preds.append(reshaped)
    all_cls_preds_concat = tf.concat(reshape_cls_preds, axis=1)  # shape: (B, 8400, 1) chua duoc ma hoa
    all_cls_preds_concat = tf.nn.sigmoid(all_cls_preds_concat)  # shape: (B, 8400, 1) da duoc ma hoa


    decoded_boxes = decode_regression_to_boxes(all_box_preds_concat)

    # all_anchors: shape(8400,2)  all_strides : shape(8400,)
    all_anchors, all_strides = get_anchors(image_shape=target_size)

    # chuyển anchors từ tọa độ tương đối sang tọa độ tuyệt đối.
    anchor_points = all_anchors * all_strides[:, None]  # shape: (8400,2) notes: 8400 còn được gọi là n_anchors.

    # Tính tọa độ (x1,y1,x2,y2) tuyệt đối trên ảnh 640x640
    pred_bboxes_xyxy = dist2bbox(decoded_boxes, anchor_points)
    all_cls_preds_concat = all_cls_preds_concat

    # 5. Áp dụng Non-Max Suppression (NMS)
    final_boxes, final_scores, final_classes, num_detections  = tf.image.combined_non_max_suppression(
        boxes=tf.expand_dims(pred_bboxes_xyxy, axis=2),  # Shape: (B, 8400, 1, 4)
        scores=all_cls_preds_concat,  # Shape: (B, 8400, num_classes)
        max_output_size_per_class=50,  # Số hộp tối đa mỗi lớp
        max_total_size=50,  # Tổng số hộp tối đa
        iou_threshold=iou_threshold,
        score_threshold=confidence_threshold
    )



    x = 0




model_path = "my_weights.weights.h5"
image_path = "dataset/images/test/n02085620_368 - Copy.jpg"
target_size = [640, 640] # height, width

if __name__ == "__main__":
    inference(model_path, image_path, target_size)

