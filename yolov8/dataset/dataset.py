import tensorflow as tf
from yolov8.dataset.data_augmentation import mosaic, mixup
import os
import numpy as np

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
    labels = tf.strings.split(labels, sep='\n')
    non_empty_lines = tf.boolean_mask(labels, tf.strings.length(labels) > 0)
    labels = tf.strings.split(non_empty_lines, sep=' ')
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
    """
    Generator function để yield dữ liệu (image, label, mask)
    """
    images_dataset, texts_dataset = getlistfile(path, training)
    Quotient = len(images_dataset) // 5  # lay phan guyen
    images_dataset = images_dataset[:Quotient * 5]
    texts_dataset = texts_dataset[:Quotient * 5]

    print("len images: " + str(len(images_dataset)) +  "    len datatext: "  + str(len(texts_dataset)) )


    if len(images_dataset) != len(texts_dataset):
        raise ValueError("so luong anh({}) va nhan({}) tuong ung ko khop".format(str(len(images_dataset)), str(len(texts_dataset))))

    if len(images_dataset) < 5:
        raise ValueError("So luong anh nho hon 5 cu the la: " + str(len(images_dataset)))



    image_list = []
    labels_list = []
    for i in tf.range(len(images_dataset)):
        try:
            image = read_img(images_dataset[i])
            label = read_label(texts_dataset[i])
        except:
            print("co van de o file" + texts_dataset[i])
            continue

        # Tạo mask với shape đúng (N, 1)
        num_boxes = tf.shape(label)[0]
        mask = tf.ones((num_boxes, 1), dtype=tf.int16)

        if len(image_list) != 4 and len(labels_list) != 4:
            image_list.append(image)
            labels_list.append(label)
        else:
            # Thực hiện mosaic và mixup
            images, labels = mosaic(image_list, labels_list, output_size=(640,640))
            mixed_image, combined_labels = mixup(images, labels, image, label, 3.0)

            # Tạo mask cho combined_labels
            combined_num_boxes = tf.shape(combined_labels)[0]
            combined_mask = tf.ones((combined_num_boxes, 1), dtype=tf.int16)
            combined_mask = tf.cast(combined_mask, dtype=tf.bool)
            combined_mask = tf.reshape(combined_mask, [combined_num_boxes,])

            yield mixed_image, combined_labels, combined_mask
            image_list = []
            labels_list = []

def get_prepared_dataset(
    data_dir="dataset",
    training="train",
    batch_size=2,
    n_max_bboxes=10,
    input_shape=[640, 640, 3],
    shuffle_buffer=10,
    drop_remainder=True
):
    """
    Hàm đóng gói toàn bộ pipeline load và chuẩn bị dữ liệu
    
    Parameters:
        data_dir: Đường dẫn đến thư mục dataset
        training: "train" hoặc "test"
        batch_size: Kích thước batch
        n_max_bboxes: Số lượng bbox tối đa sau khi padding
        input_shape: Kích thước ảnh đầu vào
        shuffle_buffer: Buffer size cho shuffle
        drop_remainder: Có drop batch cuối không đủ size không
    
    Returns:
        tf.data.Dataset đã được chuẩn bị sẵn sàng cho training
    """
    
    def generator():
        return dataset(data_dir, training)
    
    # Tạo dataset từ generator
    ds = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=input_shape, dtype=tf.float32), 
            tf.TensorSpec(shape=(None, 5), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.bool),
        )
    )
    
    # Áp dụng các transformation
    ds = ds.repeat(1)
    ds = ds.shuffle(shuffle_buffer)
    ds = ds.padded_batch(
        batch_size=batch_size,
        padded_shapes=(input_shape, (n_max_bboxes, 5), (n_max_bboxes, )),
        padding_values=(
            tf.constant(0.0, dtype=tf.float32),
            tf.constant(-1.0, dtype=tf.float32),
            tf.constant(False, dtype=tf.bool)
        ),
        drop_remainder=drop_remainder
    )
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
    
    return ds


def dataset_catch(path, training : str="train" ):
    """
        Generator function để yield dữ liệu (image, label, mask)
        """
    images_dataset, texts_dataset = getlistfile(path, training)
    Quotient = len(images_dataset) // 5  # lay phan guyen
    images_dataset = images_dataset[:Quotient * 5]
    texts_dataset = texts_dataset[:Quotient * 5]

    print("len images: " + str(len(images_dataset)) + "    len datatext: " + str(len(texts_dataset)))

    if len(images_dataset) != len(texts_dataset):
        raise ValueError(
            "so luong anh({}) va nhan({}) tuong ung ko khop".format(str(len(images_dataset)), str(len(texts_dataset))))

    if len(images_dataset) < 5:
        raise ValueError("So luong anh nho hon 5 cu the la: " + str(len(images_dataset)))

    images_list = list() # ds chua ket da tra ve
    labels_list = list()
    gt_masks_list = list()





    for i in range(0, len(texts_dataset), 5):

        # read images
        image1 = read_img(images_dataset[i])
        image2 = read_img(images_dataset[i+1])
        image3 = read_img(images_dataset[i+2])
        image4 = read_img(images_dataset[i+3])
        image5 = read_img(images_dataset[i+4])

        # read label
        label1 = read_label(texts_dataset[i])
        label2 = read_label(texts_dataset[i+1])
        label3 = read_label(texts_dataset[i+2])
        label4 = read_label(texts_dataset[i+3])
        label5 = read_label(texts_dataset[i+4])


        # mosaic argumention
        image, label = mosaic(
            [image1, image2, image3, image4, image5],
            [label1, label2, label3, label4], output_size=(640, 640)) # shape: (640, 640, 3) , (num_gt_box, 5)

        # mixup
        image, label = mixup(image, label, image5, label5, 3.0)

        # gt_mask
        combined_num_boxes = tf.shape(label)[0]
        combined_mask = tf.ones((combined_num_boxes, 1), dtype=tf.int16)
        combined_mask = tf.cast(combined_mask, dtype=tf.bool)

        images_list.append(image)
        labels_list.append(label)
        gt_masks_list.append(combined_mask)

    images_list = np.array(images_list) #  batch, 640, 640, 3
    labels_list = np.array(labels_list) # batch, num_gt, 5
    gt_masks_list = np.array(gt_masks_list) # batch, num_gt, 1

    # padding
    images_list_padding = images_list
    labels_list_padding = np.pad(
        labels_list,
        pad_width=10,
        mode="constant",
        constant_values=-1
    )

    return None