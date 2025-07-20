import tensorflow as tf


def converlabels(labels_x,scale, paddingx=0, paddingy=0):

    """
        chuyen doi box sang 1 ty le.
        Parameters:
            labels_x: tf.Tensor N,5 : moi dong la id, x, y, w, h
            scale: tuple (height, width)
            paddingx: doi voi anh o vi tri top right thi can padding 1 gia tri nhat dinh.Tuong tu voi nhung vi tri khac
            ``
        Returns:
            tra ve du lieu da duoc duoc nhan voi ti le
    """
    ids = labels_x[...,0:1]
    x = labels_x[...,1:2] * scale[0] + tf.cast(paddingx, tf.float32)
    y = labels_x[...,2:3] * scale[1] + tf.cast(paddingy, tf.float32)
    w = labels_x[...,3:4] * scale[0]
    h = labels_x[...,4:5] * scale[1]

    my = tf.concat([ids,x, y, w, h], axis=-1)
    return my


def mosaic(list_images: list, labels_list: list, output_size=(640, 640)):

    """
        ghep 4 anh vao  1 anh

        Parameters:
            list_images: list 4 phan tu.
            labels_list: list 4 phan tu.
            output_size: tuple (height, width)
        Returns:
            final_image: anh duoc ghep
            final_labels: label da duoc chinh sua theo anh da ghep.
    """
    height, width = output_size

    # chon ra 1 diem de chia thanh 4 anh
    # 160 den 480
    yc = tf.cast(tf.random.uniform([], minval=height // 4, maxval=height * 3 // 4), dtype=tf.int32)
    xc = tf.cast(tf.random.uniform([], minval=width // 4, maxval=width * 3 // 4), dtype=tf.int32)

    # su ly anh 1
    image_1, labels_1 = list_images[0], labels_list[0]
    scaley, scalex = tf.cast(yc / height, tf.float32), tf.cast(xc / width, tf.float32)
    image_1 = tf.image.resize(image_1, (yc, xc), method=tf.image.ResizeMethod.BILINEAR)
    labels_1 = converlabels(labels_1, (scalex,scaley),paddingx=0, paddingy=0)

    # su ly anh 2
    image_2, labels_2 = list_images[1], labels_list[1]
    scaley, scalex = tf.cast(yc / height, tf.float32), tf.cast((width - xc)/ width ,tf.float32)
    image_2 = tf.image.resize(image_2, (yc, width-xc), method=tf.image.ResizeMethod.BILINEAR)
    labels_2 = converlabels(labels_2,(scalex,scaley), paddingx=xc/width, paddingy=0)

    # su ly anh 3
    image_3, labels_3 = list_images[2], labels_list[2]
    scaley, scalex = tf.cast((height - yc) / height, tf.float32), tf.cast(xc / width ,tf.float32)
    image_3 = tf.image.resize(image_3, (height - yc, xc), method=tf.image.ResizeMethod.BILINEAR)
    labels_3 = converlabels(labels_3,(scalex,scaley), paddingx=0, paddingy = yc/height)

    #su ly anh 4
    image_4, labels_4 = list_images[3], labels_list[3]
    scaley, scalex = tf.cast((height - yc) / height, tf.float32), tf.cast((width - xc) / width, tf.float32)
    image_4 = tf.image.resize(image_4, (height - yc, width-xc), method=tf.image.ResizeMethod.BILINEAR)
    labels_4 = converlabels(labels_4,(scalex,scaley), paddingx = xc/width, paddingy = yc/height)

    # ghep anh 1 va 2
    image12 = tf.concat([image_1, image_2], axis=1)
    image34 = tf.concat([image_3, image_4], axis=1)

    # ghep 4 anh
    final_image = tf.concat([image12, image34], axis=0)
    final_labels = tf.concat([labels_1, labels_2, labels_3, labels_4], axis=0)

    return final_image, final_labels

def mixup(image_1, labels_1, image_2, labels_2, alpha=0.4):
    """
    Áp dụng MixUp augmentation bằng cách dùng tf.random.gamma để tạo tỉ lệ trộn.
    """

    image_2 = tf.image.resize(image_2,(640, 640), method=tf.image.ResizeMethod.BILINEAR)

    gamma_1 = tf.random.gamma(shape=[], alpha=alpha)
    gamma_2 = tf.random.gamma(shape=[], alpha=alpha)

    lam = gamma_1 / (gamma_1 + gamma_2)

    mixed_image = lam * image_1 + (1 - lam) * image_2

    combined_labels = tf.concat([labels_1, labels_2], axis=0)

    return mixed_image, combined_labels



