import tensorflow as tf
import math

def conv_block(input_tensor, filters, kernel_size, strides, name=None):
    """Một khối tích chập chuẩn: Conv -> BatchNormalization -> SiLU."""
    x = tf.keras.layers.Conv2D(filters, kernel_size, strides=strides, padding="same", use_bias=False, name=f'{name}.conv')(input_tensor)
    x = tf.keras.layers.BatchNormalization(name=f'{name}.bn')(x)
    x = tf.keras.layers.Activation(tf.nn.silu, name=f'{name}.silu')(x)
    return x

def bottleneck(c1: int, c2: int, shortcut: bool, name=None):
    """
        day la ham dinh nghia bootleneck trong yolo v8

        args:
            c1: so nguyen
            c2: so nguyen
            shortcut: True su dung ket noi tat. False khong su lu dung ket tat
        return:
            tra ve 1 lambda (ham an danh) khi nhan 1 dau vao la x

    """
    def call(x):
        c_ = int(c2 * 0.5)

        y = conv_block(x, c_, 1, 1, name=f'{name}.conv1')
        y = conv_block(y, c2, 3, 1, name=f'{name}.conv2')
        if shortcut == True and c1 == c2:
            return tf.keras.layers.Add(name=f'{name}.add')([x, y])
        else:
            return y
    return call

def sppf(input_tensor, c2, name=None):

    c1 = input_tensor.shape[-1]
    c_ = int(c1 * 0.5)

    cv1 = conv_block(input_tensor, c_, 1, 1, name=f'{name}.conv1')

    m1 = tf.keras.layers.MaxPooling2D(pool_size=5, strides=1, padding='same')(cv1)
    m2 = tf.keras.layers.MaxPooling2D(pool_size=5, strides=1, padding='same')(m1)
    m3 = tf.keras.layers.MaxPooling2D(pool_size=5, strides=1, padding='same')(m2)

    concat = tf.keras.layers.Concatenate(axis=-1)([cv1, m1, m2, m3])

    cv2 = conv_block(concat, c2, 1, 1, name=f'{name}.conv2')

    return cv2


def c2f(c1: int, c2: int, n: int=1, shortcut: bool=False, name=None):
    """"""
    c = int(c2 * 0.5)

    bottlenecks = [bottleneck(c, c, shortcut, name=f'{name}.m.{i}') for i in range(n)]

    def call(x):
        y = conv_block(x, filters=2 * c, kernel_size=1, strides=1, name=f'{name}.cv1')
        y_splits = tf.keras.layers.Lambda(
            lambda t: tf.split(t, 2, axis=-1), name=f'{name}.split'
        )(y) # khong the su dung ham rieng le vi no chi nhan 1 ham
        y_processed = [y_splits[0]]
        current_split = y_splits[1]

        for m in bottlenecks:
            current_split = m(current_split)
            y_processed.append(current_split)

        concatenated = tf.keras.layers.Concatenate(axis=-1)(y_processed)
        kq = conv_block(concatenated, filters=c2, kernel_size=1, strides=1, name=f'{name}.cv2')
        return kq
    return call

def create_head(input_tensor, num_classes, num_box_preds, name):
    """
        detection for yolo v8

    """
    box_head = conv_block(input_tensor, input_tensor.shape[-1], 3, 1, name=f'{name}.box_head.1')
    box_head = conv_block(box_head, input_tensor.shape[-1], 3, 1, name=f'{name}.box_head.2')
    box_pred = tf.keras.layers.Conv2D(num_box_preds, 1, 1, name=f'{name}.box_pred')(box_head)

    cls_head = conv_block(input_tensor, input_tensor.shape[-1], 3, 1, name=f'{name}.cls_head.1')
    cls_head = conv_block(cls_head, input_tensor.shape[-1], 3, 1, name=f'{name}.cls_head.2')
    cls_pred = tf.keras.layers.Conv2D(num_classes, 1, 1, name=f'{name}.cls_pred')(cls_head)
    return box_pred, cls_pred

def create_yolo_v8_model(input_shape: list[int, int, int]=[640,640,3], num_classes = 80, depth_multiple = 1.0, width_multiple=1.0):

    inputs = tf.keras.layers.Input(input_shape)

    def scale_channels(c):
        if c > 0:
            kq = math.ceil(c * width_multiple / 2.) * 2
        else:
            kq = c
        return kq

    def scale_depth(d):
        return max(round(d * depth_multiple), 1) if d > 0 else d

    # --- BACKBONE ---
    channels = [64, 128, 256, 512, 1024]
    scaled_channels = [scale_channels(c) for c in channels]

    x = conv_block(inputs, scaled_channels[0], 3, 2, name='backbone.stem') # 16

    # Stage 1
    x = conv_block(x, scaled_channels[1], 3, 2, name='backbone.stage1')
    x = c2f(x.shape[-1], scaled_channels[1], n=scale_depth(3), shortcut=True, name='backbone.stage2')(x)

    # stage 2
    x = conv_block(x, scaled_channels[2], 3, 2, name='backbone.stage3.downsample')
    p3 = c2f(x.shape[-1], scaled_channels[2], n=scale_depth(6), shortcut=True, name='backbone.stage3.c2f')(x)

    # Stage 3 -> Đầu ra P4
    x = conv_block(p3, scaled_channels[3], 3, 2, name='backbone.stage4.downsample')
    p4 = c2f(x.shape[-1], scaled_channels[3], n=scale_depth(6), shortcut=True, name='backbone.stage4.c2f')(x)

    # Stage 4 -> Đầu ra P5
    x = conv_block(p4, scaled_channels[4], 3, 2, name='backbone.stage5.downsample')
    x = c2f(x.shape[-1], scaled_channels[4], n=scale_depth(3), shortcut=True, name='backbone.stage5.c2f')(x)
    p5 = sppf(x, scaled_channels[4], name='backbone.sppf')
    #
    # # --- NECK (PANet) ---
    x = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear', name='neck.upsample1')(p5)
    x = tf.keras.layers.Concatenate(axis=-1, name='neck.concat1')([x, p4])
    neck_p4 = c2f(x.shape[-1], scaled_channels[3], n=scale_depth(3), shortcut=False, name='neck.c2f1')(x)

    x = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear', name='neck.upsample2')(neck_p4)
    x = tf.keras.layers.Concatenate(axis=-1, name='neck.concat2')([x, p3])
    head_p3 = c2f(x.shape[-1], scaled_channels[2], n=scale_depth(3), shortcut=False, name='neck.c2f2')(x)

    # Đường từ dưới lên (Bottom-up path)
    x = conv_block(head_p3, scaled_channels[2], 3, 2, name='neck.downsample1')
    x = tf.keras.layers.Concatenate(axis=-1, name='neck.concat3')([x, neck_p4])
    head_p4 = c2f(x.shape[-1], scaled_channels[3], n=scale_depth(3), shortcut=False, name='neck.c2f3')(x)

    x = conv_block(head_p4, scaled_channels[3], 3, 2, name='neck.downsample2')
    x = tf.keras.layers.Concatenate(axis=-1, name='neck.concat4')([x, p5])
    head_p5 = c2f(x.shape[-1], scaled_channels[4], n=scale_depth(3), shortcut=False, name='neck.c2f4')(x)

    reg_max = 16
    num_box_preds = 4 * reg_max

    box_pred_p3, cls_pred_p3 = create_head(head_p3, num_classes, num_box_preds, name='head.p3')
    box_pred_p4, cls_pred_p4 = create_head(head_p4, num_classes, num_box_preds, name='head.p4')
    box_pred_p5, cls_pred_p5 = create_head(head_p5, num_classes, num_box_preds, name='head.p5')

    raw_outputs = [box_pred_p3, cls_pred_p3, box_pred_p4, cls_pred_p4, box_pred_p5, cls_pred_p5]

    final_outputs = raw_outputs

    model = tf.keras.Model(inputs=inputs, outputs=final_outputs)
    return model


