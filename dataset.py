import tensorflow as tf
from data_loaders.shapes import Shapes
import matplotlib.pyplot as plt
from data_loaders.pascal_tmp import Pascal
import os


def build_dataset(data_loader, batch_size, shuffle=None):
    def mapper(input):
        image = tf.read_file(input['image_file'])
        image = tf.image.decode_png(image, channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)

        segmentation = tf.read_file(input['segmentation_file'])
        segmentation = tf.image.decode_png(segmentation, channels=1)
        segmentation = tf.squeeze(segmentation, -1)
        segmentation = tf.one_hot(segmentation, data_loader.num_classes)

        features = {'image': image}
        labels = {'segmentation': segmentation}

        return features, labels

    ds = tf.data.Dataset.from_generator(
        lambda: data_loader,
        output_types={'image_file': tf.string, 'segmentation_file': tf.string},
        output_shapes={'image_file': [], 'segmentation_file': []})
    ds = ds.map(mapper)
    if shuffle is not None:
        ds = ds.shuffle(shuffle)
    ds = ds.padded_batch(
        batch_size,
        ({'image': [None, None, 3]}, {'segmentation': [None, None, data_loader.num_classes]}))
    ds = ds.prefetch(1)

    return ds


def main():
    # dl = Shapes('./shapes-dataset', 10, (500, 500))
    dl = Pascal(os.path.expanduser('~/Datasets/pascal/VOCdevkit/VOC2012'), 'trainval')
    ds = build_dataset(dl, batch_size=1)

    features, labels = ds.make_one_shot_iterator().get_next()

    with tf.Session() as sess:
        f, l = sess.run([features, labels])

        for image, segmentation in zip(f['image'], l['segmentation']):
            import numpy as np
            print(np.unique(segmentation.argmax(-1)))
            plt.subplot(121)
            plt.imshow(image)
            plt.subplot(122)
            plt.imshow(segmentation.argmax(-1))
            plt.show()


if __name__ == '__main__':
    main()
