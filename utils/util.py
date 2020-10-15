from os.path import join
from xml.etree.ElementTree import ParseError
from xml.etree.ElementTree import parse as parse_fn

import cv2
import numpy as np
from six import raise_from

from utils import config


def resize(image, boxes=None):
    ih, iw = config.image_size, config.image_size
    h, w, _ = image.shape

    scale = min(iw / w, ih / h)
    nw, nh = int(scale * w), int(scale * h)
    image_resized = cv2.resize(image, (nw, nh))

    image_padded = np.zeros(shape=[ih, iw, 3], dtype=np.uint8)
    dw, dh = (iw - nw) // 2, (ih - nh) // 2
    image_padded[dh:nh + dh, dw:nw + dw, :] = image_resized.copy()

    if boxes is None:
        return image_padded, scale, dw, dh

    else:
        boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale + dw
        boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale + dh

        return image_padded, boxes


def find_node(parent, name, debug_name=None, parse=None):
    if debug_name is None:
        debug_name = name

    result = parent.find(name)
    if result is None:
        raise ValueError('missing element \'{}\''.format(debug_name))
    if parse is not None:
        try:
            return parse(result.text)
        except ValueError as e:
            raise_from(ValueError('illegal value for \'{}\': {}'.format(debug_name, e)), None)
    return result


def parse_annotation(element):
    truncated = find_node(element, 'truncated', parse=int)
    difficult = find_node(element, 'difficult', parse=int)

    class_name = find_node(element, 'name').text
    if class_name not in config.classes:
        raise ValueError('class name \'{}\' not found in classes: {}'.format(class_name, list(config.classes.keys())))

    label = config.classes[class_name]

    box = find_node(element, 'bndbox')
    x_min = find_node(box, 'xmin', 'bndbox.xmin', parse=int)
    y_min = find_node(box, 'ymin', 'bndbox.ymin', parse=int)
    x_max = find_node(box, 'xmax', 'bndbox.xmax', parse=int)
    y_max = find_node(box, 'ymax', 'bndbox.ymax', parse=int)

    return truncated, difficult, [x_min, y_min, x_max, y_max], label


def parse_annotations(xml_root):
    boxes = []
    labels = []
    for i, element in enumerate(xml_root.iter('object')):
        truncated, difficult, box, label = parse_annotation(element)
        boxes.append(box)
        labels.append(label)

    boxes = np.asarray(boxes, np.float32)
    labels = np.asarray(labels, np.int32)
    return boxes, labels


def load_image(f_name):
    path = join(config.data_dir, config.image_dir, f_name + '.jpg')
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def load_label(f_name):
    try:
        tree = parse_fn(join(config.data_dir, config.label_dir, f_name + '.xml'))
        return parse_annotations(tree.getroot())
    except ParseError as error:
        raise_from(ValueError('invalid annotations file: {}: {}'.format(f_name, error)), None)
    except ValueError as error:
        raise_from(ValueError('invalid annotations file: {}: {}'.format(f_name, error)), None)


def random_horizontal_flip(image, boxes):
    if np.random.random() < 0.5:
        _, w, _ = image.shape
        image = image[:, ::-1, :]
        boxes[:, [0, 2]] = w - boxes[:, [2, 0]]

    return image, boxes


def random_crop(image, boxes):
    if np.random.random() < 0.7:
        h, w, _ = image.shape
        max_bbox = np.concatenate([np.min(boxes[:, 0:2], axis=0), np.max(boxes[:, 2:4], axis=0)], axis=-1)

        max_l_trans = max_bbox[0]
        max_u_trans = max_bbox[1]
        max_r_trans = w - max_bbox[2]
        max_d_trans = h - max_bbox[3]

        crop_x_min = max(0, int(max_bbox[0] - np.random.uniform(0, max_l_trans)))
        crop_y_min = max(0, int(max_bbox[1] - np.random.uniform(0, max_u_trans)))
        crop_x_max = max(w, int(max_bbox[2] + np.random.uniform(0, max_r_trans)))
        crop_y_max = max(h, int(max_bbox[3] + np.random.uniform(0, max_d_trans)))

        image = image[crop_y_min: crop_y_max, crop_x_min: crop_x_max]

        boxes[:, [0, 2]] = boxes[:, [0, 2]] - crop_x_min
        boxes[:, [1, 3]] = boxes[:, [1, 3]] - crop_y_min

    return image, boxes


def random_translate(image, boxes):
    if np.random.random() < 0.7:
        h, w, _ = image.shape
        max_bbox = np.concatenate([np.min(boxes[:, 0:2], axis=0), np.max(boxes[:, 2:4], axis=0)], axis=-1)

        max_l_trans = max_bbox[0]
        max_u_trans = max_bbox[1]
        max_r_trans = w - max_bbox[2]
        max_d_trans = h - max_bbox[3]

        tx = np.random.uniform(-(max_l_trans - 1), (max_r_trans - 1))
        ty = np.random.uniform(-(max_u_trans - 1), (max_d_trans - 1))

        matrix = np.array([[1, 0, tx], [0, 1, ty]])
        image = cv2.warpAffine(image, matrix, (w, h))

        boxes[:, [0, 2]] = boxes[:, [0, 2]] + tx
        boxes[:, [1, 3]] = boxes[:, [1, 3]] + ty

    return image, boxes


def process_box(boxes, labels):
    anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    anchors = config.anchors
    box_centers = (boxes[:, 0:2] + boxes[:, 2:4]) / 2
    box_size = boxes[:, 2:4] - boxes[:, 0:2]

    y_true_1 = np.zeros((config.image_size // 32, config.image_size // 32, 3, 6 + len(config.classes)), np.float32)
    y_true_2 = np.zeros((config.image_size // 16, config.image_size // 16, 3, 6 + len(config.classes)), np.float32)
    y_true_3 = np.zeros((config.image_size // 8, config.image_size // 8, 3, 6 + len(config.classes)), np.float32)

    y_true_1[..., -1] = 1.
    y_true_2[..., -1] = 1.
    y_true_3[..., -1] = 1.

    y_true = [y_true_1, y_true_2, y_true_3]

    box_size = np.expand_dims(box_size, 1)

    min_np = np.maximum(- box_size / 2, - anchors / 2)
    max_np = np.minimum(box_size / 2, anchors / 2)

    whs = max_np - min_np

    overlap = whs[:, :, 0] * whs[:, :, 1]
    union = box_size[:, :, 0] * box_size[:, :, 1] + anchors[:, 0] * anchors[:, 1] - whs[:, :, 0] * whs[:, :, 1] + 1e-10

    iou = overlap / union
    best_match_idx = np.argmax(iou, axis=1)

    ratio_dict = {1.: 8., 2.: 16., 3.: 32.}
    for i, idx in enumerate(best_match_idx):
        feature_map_group = 2 - idx // 3
        ratio = ratio_dict[np.ceil((idx + 1) / 3.)]
        x = int(np.floor(box_centers[i, 0] / ratio))
        y = int(np.floor(box_centers[i, 1] / ratio))
        k = anchors_mask[feature_map_group].index(idx)
        c = labels[i]

        y_true[feature_map_group][y, x, k, :2] = box_centers[i]
        y_true[feature_map_group][y, x, k, 2:4] = box_size[i]
        y_true[feature_map_group][y, x, k, 4] = 1.
        y_true[feature_map_group][y, x, k, 5 + c] = 1.
        y_true[feature_map_group][y, x, k, -1] = boxes[i, -1]

    return y_true_1, y_true_2, y_true_3
