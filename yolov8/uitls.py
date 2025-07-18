import os
import glob
import tensorflow as tf
from yolov8.config import *
import cv2

def create_dataset_cache( training = True):
    """

    :return:
    """
    if not os.path.exists( labels_path ):
        print("Duong dan den nhan ko ton tai")
        return
    if not os.path.exists(images_path):
        print("khong tim thay thu muc anh.")
        return

    list_labels = glob.glob(labels_path+"/train/*.txt") if training else glob.glob(labels_path+"/val/*.txt")
    list_labels = [xx.replace("\\", "/") for xx in list_labels[:2]] # chuan hoa

    list_images = [] # luu tru anh cho toan bo du lieu
    targets = [] # luu tru du lieu cho toan bo du lieu
    for el in list_labels:
        group_target = [] # luu tru du lieu cho moi anh
        img = None
        with open(el, "r") as filetxt:
            for line in filetxt:
                classid, x_normal, y_normal, w_normal, h_normal = line.strip().split()

                classid = int(classid)
                x_normal = float(x_normal)
                y_normal = float(y_normal)
                w_normal = float(w_normal)
                h_normal = float(h_normal)

                group_target.append([classid, x_normal, y_normal, w_normal, h_normal])

            for ext in ['.jpg', '.jpeg', '.png']:
                name_base = os.path.splitext(el.split("/")[-1])[0]
                image_path = os.path.join(images_path,"train", name_base + ext) if training else os.path.join(images_path,"val", name_base + ext)
                if not os.path.exists(image_path):
                    continue
                else:
                    img = cv2.imread(image_path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (640,640))/ 255.0
                    break
        list_images.append(img)
        targets.append(group_target)

    return list_images, targets

def decode_dfl(preds, reg_max=16):

    """
        giai ma cac gia tri tho thanh toa do thuc

        Args:
            preds: shape(-1,reg_max*4): trong ngu canh nay la 64
            reg_max: so luong gia tri dai dien cho 1 toa do
        Returns:
            tra ve: (-1,4) chua x,y,w,h da giai ma nhung luc nay van la tuong doi
    """

    # tach gia tri ra thanh 4 phan
    preds = tf.reshape(preds, shape=(-1,4,reg_max))

    # chuan hoa phan phoi
    prob = tf.nn.softmax(preds, axis=-1)

    # tao ra gia tri de nhan voi xac xuat
    bins = tf.range(reg_max, dtype=tf.float32)

    # tinh gia tri trung binh de tim ra vi tri
    coords = tf.reduce_sum(prob * bins, axis=-1)  # (..., 4)

    return coords


dummy = tf.random.normal((1, 64))
decoded = decode_dfl(dummy)