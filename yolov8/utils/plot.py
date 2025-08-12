import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
import tensorflow as tf
import cv2


classid = {0:"dog"}

def converImageFrom01Into0255(images: np.ndarray):
    """
        Chuyển đổi ảnh từ 0-1 veef 0-255.

        Args:
            images: array shape (B,H,W,C) (0-1)
        Returns:
            images: array shape (B,H,W,C) (0-255)
    """
    return (images*255.0).astype(np.uint8)

def converLabels(labels: np.ndarray, height: int,width: int):
    """
        Trả về tọa độ xywh ở dạng thực.
        Args:
            labels: numpy. shape (B, N_MAX_BBOXES, 5)
            height: int. chiều cao của ảnh
            width: chiều rộng của ảnh
        Returns:
            labels:  
    """
    ids = labels[...,0]
    x_center = (labels[...,1] * width).astype(np.uint32)
    y_center = (labels[...,2] * height).astype(np.uint32)
    w = (labels[...,3] * width).astype(np.uint32)
    h = (labels[...,4] * height).astype(np.uint32)


    labels = tf.stack([ids, x_center, y_center, w, h], axis=-1)

    return labels



def plot_image(images: tf.Tensor, labels: tf.Tensor, gt_masks: tf.Tensor, xywh: bool = True):
    """
        show anh len.

        Args:
            images: tensor hoac array. shape (B, H, W, C)
            labels: tensor hoac array. shape (B, N_MAX_BBOXES, 5)
            gt_masks: tensor hoac array. shape (B, N_MAX_BBOXES)
            xywh: xem xét label đang ở dạng nào. True or False
        Returns:
            None

        Notes:
            - Vui long load du lieu co batch = 4 de tuong thich de dang
            - Đây là 1
    """
    images = images.numpy() if tf.is_tensor(images) else images
    labels = labels.numpy() if tf.is_tensor(labels) else labels
    gt_masks = gt_masks.numpy() if tf.is_tensor(gt_masks) else gt_masks

    images = converImageFrom01Into0255(images)
    labels = converLabels(labels, 640,640)


    num_image = tf.shape(images)[0] # so luong anh = B
    plt.figure(figsize=(num_image*10,10))
    for i in range(num_image):
        plt.subplot(2, 2, i + 1)
        img = images[i]
        cv2.imwrite("image_result/{}.png".format(str(i)), img)
        if img.shape[-1] == 1:
            img = img.squeeze(-1)
            plt.imshow(img,cmap="gray")
        else:
            plt.imshow(img)

        # ve bboxes (labels)
        ax = plt.gca()
        for (box, mask) in zip(labels[i], gt_masks[i]):
            if mask == 0:
                continue
            else:
                ids,x,y,w,h = box
                assert x >= 0.0 and x <= 640
                assert y >= 0.0 and y <= 640
                assert w >=0
                assert h >= 0


                if xywh:
                    x1 = int(x - w / 2)
                    y1 = int(y - h / 2)
                    w = int(w)
                    h = int(h)
                else:
                    x1 = int(x)
                    y1 = int(y)
                    w = int(w-x)
                    h = int(h - y)

                rect = patches.Rectangle(
                    (x1,y1), w, h, linewidth=2, edgecolor="red", facecolor="none"
                )
                ax.add_patch(rect)

                # ve nhan that te
                label_text = classid[int(ids)]
                plt.text(
                    x1, y1, label_text, color="yellow", fontsize=12, backgroundcolor='black'
                )
                
    plt.tight_layout()  
    plt.show()





