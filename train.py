import tensorflow as tf
import losses
import unet
from data_loaders.shapes import Shapes
from dataset import build_dataset
import argparse
import os
import utils
from data_loaders.pascal_tmp import Pascal


# TODO: rename logits after softmax to prob
# TODO: check num_classes usage

def build_summary(image, labels, logits):
    segmentation_true = utils.draw_segmentation(labels)
    segmentation_pred = utils.draw_segmentation(tf.nn.softmax(logits))

    tf.summary.image('image', image)
    tf.summary.image('segmentation_true', segmentation_true)
    tf.summary.image('segmentation_pred', segmentation_pred)


# TODO: regularization, initialization

def model_fn(features, labels, mode, params, config):
    global_step = tf.train.get_or_create_global_step()
    training = mode == tf.estimator.ModeKeys.TRAIN

    net = unet.Unet(num_classes=params['data_loader'].num_classes)
    logits = net(features['image'], training=training)
    loss = losses.segmentation_loss(labels=labels['segmentation'], logits=logits, losses=params['losses'])

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(params['learning_rate'])

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_step = optimizer.minimize(loss, global_step=global_step)

        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_step)

    elif mode == tf.estimator.ModeKeys.EVAL:
        mask = tf.argmax(labels['segmentation'], -1)
        predictions = tf.argmax(logits, -1)

        metrics = {'iou': tf.metrics.mean_iou(
            labels=mask, predictions=predictions, num_classes=params['data_loader'].num_classes + 1)}

        build_summary(features['image'], labels['segmentation'], logits)

        summary_hook = tf.train.SummarySaverHook(
            save_steps=10,
            # save_secs=60,
            output_dir=os.path.join(config.model_dir, 'eval'),
            summary_op=tf.summary.merge_all())

        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics, evaluation_hooks=[summary_hook])


def input_fn(params):
    ds = build_dataset(params['data_loader']).batch(params['batch_size']).prefetch(1)
    features, labels = ds.make_one_shot_iterator().get_next()

    return features, labels


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=str, required=True)
    parser.add_argument('--losses', type=str, required=True, nargs='+')
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--batch-size', type=int, default=32)

    return parser


def main():
    EPOCHS = 1000

    args = build_parser().parse_args()

    config = tf.estimator.RunConfig(
        model_dir=args.experiment,
        # save_checkpoints_steps=LOG_INTERVAL,
        save_summary_steps=100)

    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        params={
            # 'data_loader': Shapes('./shapes-dataset', args.batch_size * 100, (224, 224)),
            'data_loader': Pascal('/data/Vlad/code/retinanet-tensorflow/data/pascal/VOCdevkit/VOC2012', 'trainval'),
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'losses': args.losses

        },
        config=config)

    for epoch in range(EPOCHS):
        estimator.train(input_fn)
        estimator.evaluate(input_fn, steps=10)


if __name__ == '__main__':
    main()
