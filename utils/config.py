import numpy as np
from os.path import join

epochs = 300
batch_size = 16
image_size = 1024
data_dir = join('..', 'Dataset', 'VOC2012')
image_dir = 'IMAGES'
label_dir = 'LABELS'
voc_classes = {'aeroplane': 0,
               'bicycle': 1,
               'bird': 2,
               'boat': 3,
               'bottle': 4,
               'bus': 5,
               'car': 6,
               'cat': 7,
               'chair': 8,
               'cow': 9,
               'diningtable': 10,
               'dog': 11,
               'horse': 12,
               'motorbike': 13,
               'person': 14,
               'pottedplant': 15,
               'sheep': 16,
               'sofa': 17,
               'train': 18,
               'tvmonitor': 19}
classes = voc_classes
anchors = np.array([[9, 8], [12, 11], [14, 15],
                    [17, 8], [18, 24], [25, 13],
                    [35, 29], [46, 13], [91, 24]], np.float32)
