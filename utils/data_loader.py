import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import OrderedEnqueuer
from tensorflow.keras.utils import Sequence

from utils import config
from utils.util import load_image
from utils.util import load_label
from utils.util import process_box
from utils.util import random_crop
from utils.util import random_horizontal_flip
from utils.util import random_translate
from utils.util import resize


class Generator(Sequence):
    def __init__(self, f_names):
        self.f_names = f_names
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.f_names) / config.batch_size))

    def __getitem__(self, index):
        image = load_image(self.f_names[index])
        boxes, label = load_label(self.f_names[index])
        boxes = np.concatenate((boxes, np.full(shape=(boxes.shape[0], 1), fill_value=1., dtype=np.float32)), axis=-1)
        image, boxes = random_horizontal_flip(image, boxes)
        image, boxes = random_crop(image, boxes)
        image, boxes = random_translate(image, boxes)
        image, boxes = resize(image, boxes)

        image = image.astype(np.float32)
        image /= 255.
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        image -= mean
        image /= std
        y_true_1, y_true_2, y_true_3 = process_box(boxes, label)
        return image, y_true_1, y_true_2, y_true_3

    def on_epoch_end(self):
        np.random.shuffle(self.f_names)


def input_fn(f_names):
    def generator_fn():
        generator = OrderedEnqueuer(Generator(f_names), True)
        generator.start(workers=8, max_queue_size=10)
        while True:
            image, y_true_1, y_true_2, y_true_3 = generator.get().__next__()
            yield image, y_true_1, y_true_2, y_true_3

    output_types = (tf.float32, tf.float32, tf.float32, tf.float32)
    output_shapes = ((config.image_size, config.image_size, 3),
                     (config.image_size // 32, config.image_size // 32, 3, len(config.classes) + 6),
                     (config.image_size // 16, config.image_size // 16, 3, len(config.classes) + 6),
                     (config.image_size // 8, config.image_size // 8, 3, len(config.classes) + 6),)

    dataset = tf.data.Dataset.from_generator(generator=generator_fn,
                                             output_types=output_types,
                                             output_shapes=output_shapes)

    dataset = dataset.repeat(config.epochs + 1)
    dataset = dataset.batch(config.batch_size)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset
