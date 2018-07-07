import tensorflow as tf
from data_loaders.shapes import Shapes
import matplotlib.pyplot as plt
from data_loaders.pascal_tmp import Pascal
import os


def build_dataset(data_loader):
    def mapper(input):
        image = tf.read_file(input['image_file'])
        image = tf.image.decode_png(image, channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)

        # mask = tf.read_file(input['mask_file'])
        # mask = tf.image.decode_png(mask, channels=1)
        # mask = tf.squeeze(mask, -1)
        segmentation = tf.one_hot(input['segmentation'], data_loader.num_classes)

        features = {'image': image}
        labels = {'segmentation': segmentation}

        return features, labels

    ds = tf.data.Dataset.from_generator(
        lambda: data_loader,
        output_types={'image_file': tf.string, 'segmentation': tf.int32},
        output_shapes={'image_file': [], 'segmentation': [None, None]})
    ds = ds.map(mapper)

    return ds


def main():
    # dl = Shapes('./shapes-dataset', 10, (500, 500))
    dl = Pascal(os.path.expanduser('~/Datasets/pascal/VOCdevkit/VOC2012'), 'trainval')
    ds = build_dataset(dl)

    features, labels = ds.make_one_shot_iterator().get_next()

    with tf.Session() as sess:
        f, l = sess.run([features, labels])

        plt.subplot(121)
        plt.imshow(f['image'])
        plt.subplot(122)
        plt.imshow(l['segmentation'].argmax(-1))
        plt.show()


if __name__ == '__main__':
    main()
