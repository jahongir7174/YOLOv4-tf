import numpy as np
import tensorflow as tf
from tensorflow import nn
from tensorflow.keras import backend
from tensorflow.keras import layers

from utils import config

initializer = tf.random_normal_initializer(stddev=0.01)


def activation_fn(x):
    return x * backend.tanh(backend.softplus(x))


def convolution_block(inputs, filters, kernel_size, strides=1):
    if strides == 2:
        inputs = layers.ZeroPadding2D(((1, 0), (1, 0)))(inputs)
        padding = 'valid'
    else:
        padding = 'same'

    x = layers.Conv2D(filters, kernel_size, strides, padding, use_bias=False, kernel_initializer=initializer)(inputs)

    x = layers.BatchNormalization()(x)
    x = layers.Activation(activation_fn)(x)

    return x


def residual_block(inputs, filter_num1, filter_num2):
    x = convolution_block(inputs, filter_num1, 1)
    x = convolution_block(x, filter_num2, 3)

    return inputs + x


def backbone(inputs):
    x = convolution_block(inputs, 32, 3)
    x = convolution_block(x, 63, 3, 2)

    skip = convolution_block(x, 64, 1)
    x = convolution_block(x, 64, 1)
    for i in range(1):
        x = residual_block(x, 32, 64)
    x = convolution_block(x, 64, 1)

    x = tf.concat([x, skip], axis=-1)
    x = convolution_block(x, 64, 1)
    x = convolution_block(x, 128, 3, 2)

    skip = convolution_block(x, 64, 1)
    x = convolution_block(x, 64, 1)
    for i in range(2):
        x = residual_block(x, 64, 64)
    x = convolution_block(x, 64, 1)
    x = tf.concat([x, skip], axis=-1)

    x = convolution_block(x, 128, 1)
    x = convolution_block(x, 256, 3, 2)

    skip = convolution_block(x, 128, 1)
    x = convolution_block(x, 128, 1)
    for i in range(8):
        x = residual_block(x, 128, 128)
    x = convolution_block(x, 128, 1)
    x = tf.concat([x, skip], axis=-1)

    x = convolution_block(x, 256, 1)
    skip1 = x
    x = convolution_block(x, 512, 3, 2)

    skip = convolution_block(x, 256, 1)
    x = convolution_block(x, 256, 1)
    for i in range(8):
        x = residual_block(x, 256, 256)
    x = convolution_block(x, 256, 1)
    x = tf.concat([x, skip], axis=-1)

    x = convolution_block(x, 512, 1)
    skip2 = x
    x = convolution_block(x, 1024, 3, 2)

    skip = convolution_block(x, 512, 1)
    x = convolution_block(x, 512, 1)
    for i in range(4):
        x = residual_block(x, 512, 512)
    x = convolution_block(x, 512, 1)
    x = tf.concat([x, skip], axis=-1)

    x = convolution_block(x, 1024, 1)
    x = convolution_block(x, 512, 1)
    x = convolution_block(x, 1024, 3)
    x = convolution_block(x, 512, 1)

    x = tf.concat([nn.max_pool(x, 9, 1, 'SAME'),
                   nn.max_pool(x, 5, 1, 'SAME'),
                   nn.max_pool(x, 13, 1, 'SAME'), x], axis=-1)
    x = convolution_block(x, 512, 1)
    x = convolution_block(x, 1024, 3)
    x = convolution_block(x, 512, 1)

    return skip1, skip2, x


def build_model(inputs, nb_classes):
    x1, x2, x3 = backbone(inputs)

    x = convolution_block(x3, 256, 1)
    x = layers.UpSampling2D(interpolation='bilinear')(x)
    x2 = convolution_block(x2, 256, 1)
    x = layers.concatenate([x2, x])

    x = convolution_block(x, 256, 1)
    x = convolution_block(x, 512, 3)
    x = convolution_block(x, 256, 1)
    x = convolution_block(x, 512, 3)
    x = convolution_block(x, 256, 1)

    x2 = x
    x = convolution_block(x, 128, 1)
    x = layers.UpSampling2D(interpolation='bilinear')(x)
    x1 = convolution_block(x1, 128, 1)
    x = layers.concatenate([x1, x])

    x = convolution_block(x, 128, 1)
    x = convolution_block(x, 256, 3)
    x = convolution_block(x, 128, 1)
    x = convolution_block(x, 256, 3)
    x = convolution_block(x, 128, 1)

    x1 = x
    x = convolution_block(x, 256, 3)
    small = layers.Conv2D(3 * (nb_classes + 5), 1, kernel_initializer=initializer)(x)

    x = convolution_block(x1, 256, 3, 2)
    x = layers.concatenate([x, x2])

    x = convolution_block(x, 256, 1)
    x = convolution_block(x, 512, 3)
    x = convolution_block(x, 256, 1)
    x = convolution_block(x, 512, 3)
    x = convolution_block(x, 256, 1)

    x2 = x
    x = convolution_block(x, 512, 3)
    medium = layers.Conv2D(3 * (nb_classes + 5), 1, kernel_initializer=initializer)(x)

    x = convolution_block(x2, 512, 3, 2)
    x = layers.concatenate([x, x3])

    x = convolution_block(x, 512, 1)
    x = convolution_block(x, 1024, 3)
    x = convolution_block(x, 512, 1)
    x = convolution_block(x, 1024, 3)
    x = convolution_block(x, 512, 1)

    x = convolution_block(x, 1024, 3)
    large = layers.Conv2D(3 * (nb_classes + 5), 1, kernel_initializer=initializer)(x)

    return small, medium, large


def decode(output, i=0, nb_classes=20):
    shape = tf.shape(output)
    batch_size = shape[0]
    output_size = shape[1]

    output = tf.reshape(output, (batch_size, output_size, output_size, 3, 5 + nb_classes))

    raw_xy = output[:, :, :, :, 0:2]
    raw_wh = output[:, :, :, :, 2:4]
    raw_conf = output[:, :, :, :, 4:5]
    raw_prob = output[:, :, :, :, 5:]

    y = tf.tile(tf.range(output_size, dtype=tf.int32)[:, tf.newaxis], [1, output_size])
    x = tf.tile(tf.range(output_size, dtype=tf.int32)[tf.newaxis, :], [output_size, 1])

    xy_grid = tf.concat([x[:, :, tf.newaxis], y[:, :, tf.newaxis]], axis=-1)
    xy_grid = tf.tile(xy_grid[tf.newaxis, :, :, tf.newaxis, :], [batch_size, 1, 1, 3, 1])
    xy_grid = tf.cast(xy_grid, tf.float32)

    pred_xy = (tf.sigmoid(raw_xy) + xy_grid) * config.strides[i]
    pred_wh = (tf.exp(raw_wh) * config.anchors[i]) * config.strides[i]
    pred_xy_wh = tf.concat([pred_xy, pred_wh], axis=-1)

    pred_conf = tf.sigmoid(raw_conf)
    pred_prob = tf.sigmoid(raw_prob)

    return tf.concat([pred_xy_wh, pred_conf, pred_prob], axis=-1)


def box_iou(boxes1, boxes2):
    boxes1_area = boxes1[..., 2] * boxes1[..., 3]
    boxes2_area = boxes2[..., 2] * boxes2[..., 3]

    boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                        boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
    boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                        boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

    left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = tf.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area

    return 1.0 * inter_area / union_area


def box_g_iou(boxes1, boxes2):
    boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                        boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
    boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                        boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

    boxes1 = tf.concat([tf.minimum(boxes1[..., :2], boxes1[..., 2:]),
                        tf.maximum(boxes1[..., :2], boxes1[..., 2:])], axis=-1)
    boxes2 = tf.concat([tf.minimum(boxes2[..., :2], boxes2[..., 2:]),
                        tf.maximum(boxes2[..., :2], boxes2[..., 2:])], axis=-1)

    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = tf.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    iou = inter_area / union_area

    enclose_left_up = tf.minimum(boxes1[..., :2], boxes2[..., :2])
    enclose_right_down = tf.maximum(boxes1[..., 2:], boxes2[..., 2:])
    enclose = tf.maximum(enclose_right_down - enclose_left_up, 0.0)
    enclose_area = enclose[..., 0] * enclose[..., 1]

    return iou - 1.0 * (enclose_area - union_area) / enclose_area


def compute_loss(pred, output, label, boxes, i=0):
    shape = tf.shape(output)
    batch_size = shape[0]
    output_size = shape[1]
    input_size = config.strides[i] * output_size
    output = tf.reshape(output, (batch_size, output_size, output_size, 3, 5 + 20))

    raw_conf = output[:, :, :, :, 4:5]
    raw_prob = output[:, :, :, :, 5:]

    pred_xy_wh = pred[:, :, :, :, 0:4]
    pred_conf = pred[:, :, :, :, 4:5]

    label_xy_wh = label[:, :, :, :, 0:4]
    respond_bbox = label[:, :, :, :, 4:5]
    label_prob = label[:, :, :, :, 5:]

    g_iou = tf.expand_dims(box_g_iou(pred_xy_wh, label_xy_wh), axis=-1)
    input_size = tf.cast(input_size, tf.float32)

    bbox_loss_scale = 2.0 - 1.0 * label_xy_wh[:, :, :, :, 2:3] * label_xy_wh[:, :, :, :, 3:4] / (input_size ** 2)
    g_iou_loss = respond_bbox * bbox_loss_scale * (1 - g_iou)

    iou = box_iou(pred_xy_wh[:, :, :, :, np.newaxis, :], boxes[:, np.newaxis, np.newaxis, np.newaxis, :, :])
    max_iou = tf.expand_dims(tf.reduce_max(iou, axis=-1), axis=-1)

    respond_bgd = (1.0 - respond_bbox) * tf.cast(max_iou < 0.5, tf.float32)

    conf_focal = tf.pow(respond_bbox - pred_conf, 2)

    conf_loss = conf_focal * (respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(respond_bbox, raw_conf)
                              +
                              respond_bgd * tf.nn.sigmoid_cross_entropy_with_logits(respond_bbox, raw_conf))

    prob_loss = respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(label_prob, raw_prob)

    g_iou_loss = tf.reduce_mean(tf.reduce_sum(g_iou_loss, axis=[1, 2, 3, 4]))
    conf_loss = tf.reduce_mean(tf.reduce_sum(conf_loss, axis=[1, 2, 3, 4]))
    prob_loss = tf.reduce_mean(tf.reduce_sum(prob_loss, axis=[1, 2, 3, 4]))

    return g_iou_loss, conf_loss, prob_loss
