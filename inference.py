import tensorflow as tf


def load_model(path: str = "model.h5"):
    """
        load model
        Args:
            path: model path: vd: model.h5
        Returns:
            model: keras model
    """
    # load model
    model = tf.keras.models.load_model(path)
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


def inference(path_model, path_image, target_size : [int, int] = [640, 640]):
    """
        show Corresponding boxes with an images

    """

    model = load_model(path_model) # load model
    image, [height_ratio, width_ratio] = load_image(path_image, target_size) # load image



    #predcit boxes and class
    box_pred_p3, cls_pred_p3, box_pred_p4, cls_pred_p4, box_pred_p5, cls_pred_p5 = model.predict(image)


    print(box_pred_p3.shape)


model_path = "model.h5"
image_path = "D:\yolo-v8\dataset\images\test\n02085936_1556.jpg"
target_size = [640, 640] # height, width

if __name__ == "__main__":
    load_model(model_path)

