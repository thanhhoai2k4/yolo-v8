import tensorflow as tf
from yolov8.dataset.data_augmentation import mosaic, mixup

def read_img(path):
    """
        doc 1 anh

        Parameters:
            path: Duong dan den anh.
        Returns:
            Tensor 0-1
    """
    if not tf.io.gfile.exists(path): # kiem tra tot tai cua duong dan
        raise FileNotFoundError(f"Error file not exist: {path}")
    images = tf.io.read_file(path) # read
    images = tf.image.decode_png(images, channels=3) # convert 3 channels
    images = tf.image.convert_image_dtype(images, tf.float32) # any to float
    images /= 255.0 # 0-255 =) 0-1

    return images

def read_label(path):
    """
        doc 1 file text
        Parameters:
            path: Duong dan den txt
        Returns:
            Tensor (N 5): N la so luong box co trong anh.
    """

    if not tf.io.gfile.exists(path): # kiem tra duong dan
        raise FileNotFoundError(f"Error file not exist: {path}")
    labels = tf.io.read_file(path)
    labels = tf.strings.strip(tf.strings.split(labels, sep='\n'))
    labels = tf.strings.split(labels, sep=' ')
    labels = tf.strings.to_number(labels, out_type=tf.float32)
    final_labels = labels.to_tensor()

    return final_labels

def getlistfile(path, training : str="train" ):
    """
        lay ds tu path
        Parameters:
            path: Duong dan den thu muc dataset
            training: train or test
        Returns:
            tra ve 2 bien chua ds tuong ung anh va nhan.
    """
    import os
    path2imgdataset = os.path.join(os.getcwd(), path, "images", training, "*")
    path2txtdataset = os.path.join(os.getcwd(), path, "labels", training, "*")
    images_dataset = tf.io.gfile.glob(path2imgdataset)
    texts_dataset = tf.io.gfile.glob(path2txtdataset)

    return images_dataset, texts_dataset

def dataset(path, training : str="train" ):

    images_dataset, texts_dataset = getlistfile(path, training)

    image_list = []
    labels_list = []
    for i in tf.range(len(images_dataset)):

        image = read_img(images_dataset[i])
        label = read_label(texts_dataset[i])

        if len(image_list) != 4 and len(labels_list) != 4:
            image_list.append(image)
            labels_list.append(label)

        else: # thuc hien argument co mosaic
            images, labels = mosaic(image_list, labels_list, output_size=(640,640))
            mixed_image, combined_labels = mixup(images, labels, image, label, 3.0)
            yield mixed_image, combined_labels

            image_list = []
            labels_list = []