import tensorflow as tf


def draw_segmentation(input):
    bg_color = tf.zeros((1, 1, 1, 1, 3))
    class_colors = tf.random_uniform([1, 1, 1, input.shape[-1].value - 1, 3], 0., 1., seed=42)
    colors = tf.concat([bg_color, class_colors], -2)

    input = tf.expand_dims(input, -1)
    input *= colors
    input = tf.reduce_sum(input, -2)

    return input
