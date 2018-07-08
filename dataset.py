import tensorflow as tf
from data_loaders.shapes import Shapes
import matplotlib.pyplot as plt
from data_loaders.pascal_tmp import Pascal
import os
from itertools import count
from tqdm import tqdm


def build_dataset(data_loader, batch_size, shuffle=None):
    def mapper(input):
        image = tf.read_file(input['image_file'])
        image = tf.image.decode_png(image, channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)

        segmentation = tf.read_file(input['segmentation_file'])
        segmentation = tf.image.decode_png(segmentation, channels=1)
        segmentation = tf.squeeze(segmentation, -1)
        segmentation = tf.one_hot(segmentation, data_loader.num_classes)

        image = tf.image.resize_images(image, (224, 224), method=tf.image.ResizeMethod.BILINEAR)
        segmentation = tf.image.resize_images(segmentation, (224, 224), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        features = {'image': image}
        labels = {'segmentation': segmentation}

        return features, labels

    ds = tf.data.Dataset.from_generator(
        lambda: data_loader,
        output_types={'image_file': tf.string, 'segmentation_file': tf.string},
        output_shapes={'image_file': [], 'segmentation_file': []})
    ds = ds.map(mapper, num_parallel_calls=min(os.cpu_count(), 4))
    if shuffle is not None:
        ds = ds.shuffle(shuffle)

    ds = ds.batch(batch_size)
    # ds = ds.padded_batch(
    #     batch_size,
    #     ({'image': [None, None, 3]}, {'segmentation': [None, None, data_loader.num_classes]}))

    return ds


def main():
    # dl = Shapes('./shapes-dataset', 10, (500, 500))
    dl = Pascal(os.path.expanduser('~/Datasets/pascal/VOCdevkit/VOC2012'), 'trainval')
    ds = build_dataset(dl, batch_size=4)

    features, labels = ds.make_one_shot_iterator().get_next()

    with tf.Session() as sess:
        for _ in tqdm(count()):
            f, l = sess.run([features, labels])

        for image, segmentation in zip(f['image'], l['segmentation']):
            plt.figure(figsize=(10, 5))
            plt.subplot(121)
            plt.imshow(image)
            plt.subplot(122)
            plt.imshow(segmentation.argmax(-1))
            plt.show()


if __name__ == '__main__':
    main()
