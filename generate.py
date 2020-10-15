import os
from os.path import join

import numpy as np

from utils import config
from utils.util import load_label


class AnchorGenerator:
    def __init__(self, cluster_number):
        self.cluster_number = cluster_number

    def iou(self, boxes, clusters):
        n = boxes.shape[0]
        k = self.cluster_number

        box_area = boxes[:, 0] * boxes[:, 1]
        box_area = box_area.repeat(k)
        box_area = np.reshape(box_area, (n, k))

        cluster_area = clusters[:, 0] * clusters[:, 1]
        cluster_area = np.tile(cluster_area, [1, n])
        cluster_area = np.reshape(cluster_area, (n, k))

        box_w_matrix = np.reshape(boxes[:, 0].repeat(k), (n, k))
        cluster_w_matrix = np.reshape(np.tile(clusters[:, 0], (1, n)), (n, k))
        min_w_matrix = np.minimum(cluster_w_matrix, box_w_matrix)

        box_h_matrix = np.reshape(boxes[:, 1].repeat(k), (n, k))
        cluster_h_matrix = np.reshape(np.tile(clusters[:, 1], (1, n)), (n, k))
        min_h_matrix = np.minimum(cluster_h_matrix, box_h_matrix)
        inter_area = np.multiply(min_w_matrix, min_h_matrix)

        result = inter_area / (box_area + cluster_area - inter_area)
        return result

    def avg_iou(self, boxes, clusters):
        accuracy = np.mean([np.max(self.iou(boxes, clusters), axis=1)])
        return accuracy

    def generator(self, boxes, k, dist=np.median):
        box_number = boxes.shape[0]
        last_nearest = np.zeros((box_number,))
        clusters = boxes[np.random.choice(box_number, k, replace=False)]
        while True:
            distances = 1 - self.iou(boxes, clusters)

            current_nearest = np.argmin(distances, axis=1)
            if (last_nearest == current_nearest).all():
                break
            for cluster in range(k):
                clusters[cluster] = dist(boxes[current_nearest == cluster], axis=0)
            last_nearest = current_nearest

        return clusters

    def generate_anchor(self):
        boxes = self.get_boxes()
        result = self.generator(boxes, k=self.cluster_number)
        result = result[np.lexsort(result.T[0, None])]
        print("K anchors:\n {}".format(result))
        print("Accuracy: {:.2f}%".format(self.avg_iou(boxes, result) * 100))

    @staticmethod
    def get_boxes():
        boxes = []
        f_names = [f_name[:-4] for f_name in os.listdir(join(config.data_dir, config.label_dir))]
        for f_name in f_names:
            annotations = load_label(f_name)
            for annotation in annotations[0]:
                x_min = annotation[0]
                y_min = annotation[1]
                x_max = annotation[2]
                y_max = annotation[3]

                boxes.append([x_max - x_min, y_max - y_min])
        return np.array(boxes)


if __name__ == "__main__":
    generator = AnchorGenerator(9)
    generator.generate_anchor()
