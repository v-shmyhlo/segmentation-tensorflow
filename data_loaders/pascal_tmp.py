import os
import xml.etree.ElementTree as ET
import numpy as np
import cv2
from data_loaders.base import Base
from multiprocessing import Pool
from tqdm import tqdm

classes = [
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
    'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

class_to_color = {
    'background': (0, 0, 0),
    'aeroplane': (128, 0, 0),
    'bicycle': (0, 128, 0),
    'bird': (128, 128, 0),
    'boat': (0, 0, 128),
    'bottle': (128, 0, 128),
    'bus': (0, 128, 128),
    'car': (128, 128, 128),
    'cat': (64, 0, 0),
    'chair': (192, 0, 0),
    'cow': (64, 128, 0),
    'diningtable': (192, 128, 0),
    'dog': (64, 0, 128),
    'horse': (192, 0, 128),
    'motorbike': (64, 128, 128),
    'person': (192, 128, 128),
    'pottedplant': (0, 64, 0),
    'sheep': (128, 64, 0),
    'sofa': (0, 192, 0),
    'train': (128, 192, 0),
    'tvmonitor': (0, 64, 128)
}


class Pascal(Base):
    def __init__(self, path, subset):  # FIXME:
        self._path = path
        self._subset = subset

    @property
    def num_classes(self):
        return len(classes)

    def _load_segmentation(self, image_name):
        segmentation_file = os.path.join(self._path, 'SegmentationClass', image_name + '.png')
        segmentation = cv2.imread(segmentation_file)
        segmentation = cv2.cvtColor(segmentation, cv2.COLOR_BGR2RGB)

        segmentation_flat = np.zeros(segmentation.shape[:2])
        for c in classes:
            mask = np.all(segmentation == np.reshape(class_to_color[c], (1, 1, 3)), -1)
            segmentation_flat[mask] = classes.index(c)

        return segmentation_flat

    def _load_boxes(self, image_name):
        tree = ET.parse(os.path.join(self._path, 'Annotations', image_name + '.xml'))

        boxes = []
        class_ids = []
        for obj in tree.getroot().iter('object'):
            t = float(obj.find('bndbox/ymin').text)
            l = float(obj.find('bndbox/xmin').text)
            b = float(obj.find('bndbox/ymax').text)
            r = float(obj.find('bndbox/xmax').text)

            boxes.append([t, l, b, r])
            class_ids.append(classes.index(obj.find('name').text))

        boxes = np.array(boxes).reshape((-1, 4))
        class_ids = np.array(class_ids).reshape(-1)

        return boxes, class_ids

    def _load_sample(self, image_name):
        image_file = os.path.join(self._path, 'JPEGImages', image_name + '.jpg')

        segmentation = self._load_segmentation(image_name)

        # boxes, class_ids = self._load_boxes(image_name)
        return {
            'image_file': image_file.encode('utf-8'),
            'segmentation': segmentation,
            # 'class_ids': boxes,
            # 'boxes': class_ids
        }

    def __iter__(self):
        with open(os.path.join(self._path, 'ImageSets', 'Segmentation', self._subset + '.txt')) as f:
            lines = f.readlines()
            image_names = [line.strip().split()[0] for line in lines]

        with Pool(min(os.cpu_count(), 4)) as pool:
            yield from pool.imap_unordered(self._load_sample, image_names)


if __name__ == '__main__':
    dl = Pascal(os.path.expanduser('~/Datasets/pascal/VOCdevkit/VOC2012'), 'trainval')

    for _ in tqdm(dl):
        pass
