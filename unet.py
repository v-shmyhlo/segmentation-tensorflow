import tensorflow as tf
from mobilenet_v2 import MobileNetV2


class Conv(tf.layers.Layer):
    def __init__(self, filters, name='conv'):
        super().__init__(name=name)

        self._filters = filters

    def build(self, input_shape):
        self._conv = tf.layers.Conv2D(self._filters, 3, padding='same', use_bias=False)

        super().build(input_shape)

    def call(self, input):
        input = self._conv(input)

        return input


class ConvBNRelu(tf.layers.Layer):
    def __init__(self, filters, name='conv_bn_relu'):
        super().__init__(name=name)

        self._filters = filters

    def build(self, input_shape):
        self._conv = Conv(self._filters)
        self._bn = tf.layers.BatchNormalization()

        super().build(input_shape)

    def call(self, input, training):
        input = self._conv(input)
        input = self._bn(input, training=training)
        input = tf.nn.elu(input)

        return input


class UpsampleMerge(tf.layers.Layer):
    def __init__(self, filters, name='upsample_merge'):
        super().__init__(name=name)

        self._filters = filters

    def build(self, input_shape):
        self._squeeze = ConvBNRelu(self._filters)
        self._output_conv = ConvBNRelu(self._filters)

        super().build(input_shape)

    def call(self, input, lateral, training):
        input = self._squeeze(input, training=training)
        input = tf.image.resize_bilinear(input, (input.shape[1] * 2, input.shape[2] * 2), align_corners=True)

        input = tf.concat([input, lateral], -1)
        input = self._output_conv(input, training=training)

        return input


class Decoder(tf.layers.Layer):
    def __init__(self, num_classes, name='decoder'):
        super().__init__(name=name)

        self._num_classes = num_classes

    def build(self, input_shape):
        self._upsample_merge_c5_c4 = UpsampleMerge(512)
        self._upsample_merge_c4_c3 = UpsampleMerge(256)
        self._upsample_merge_c3_c2 = UpsampleMerge(128)
        self._upsample_merge_c2_c1 = UpsampleMerge(64)
        self._output_conv = Conv(self._num_classes)

        super().build(input_shape)

    def call(self, input, training):
        output = input['C5']

        output = self._upsample_merge_c5_c4(output, input['C4'], training=training)
        output = self._upsample_merge_c4_c3(output, input['C3'], training=training)
        output = self._upsample_merge_c3_c2(output, input['C2'], training=training)
        output = self._upsample_merge_c2_c1(output, input['C1'], training=training)
        output = tf.image.resize_bilinear(output, (output.shape[1] * 2, output.shape[2] * 2), align_corners=True)
        output = self._output_conv(output)

        return output


class Unet(tf.layers.Layer):
    def __init__(self, num_classes, name='unet'):
        super().__init__(name=name)

        self._num_classes = num_classes

    def build(self, input_shape):
        self._encoder = MobileNetV2(dropout_rate=0.2, activation=tf.nn.elu)  # TODO:
        self._decoder = Decoder(num_classes=self._num_classes)

        super().build(input_shape)

    def call(self, input, training):
        encoded = self._encoder(input, training=training)
        decoded = self._decoder(encoded, training=training)

        return decoded
