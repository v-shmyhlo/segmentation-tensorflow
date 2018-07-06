import tensorflow as tf
import losses
import unet
from data_loaders.shapes import Shapes
from dataset import build_dataset


# TODO: regularization, initialization

def model_fn(features, labels, mode, params):
    global_step = tf.train.get_or_create_global_step()
    training = mode == tf.estimator.ModeKeys.TRAIN

    net = unet.Unet(num_classes=params['data_loader'].num_classes + 1)
    logits = net(features['image'], training=training)
    predictions = tf.argmax(logits, -1)
    loss = losses.segmentation_loss(labels=labels['mask'], logits=logits)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer()
        train_step = optimizer.minimize(loss, global_step=global_step)

        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_step)

    if mode == tf.estimator.ModeKeys.EVAL:
        tf.summary.image('image', features['image'])

        mask = tf.argmax(labels['mask'], -1)
        tf.summary.image('mask', mask)

        metrics = {'iou': tf.metrics.mean_iou(
            labels=mask, predictions=predictions, num_classes=params['data_loader'].num_classes + 1)}

        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)


def input_fn(params):
    ds = build_dataset(params['data_loader']).batch(4).prefetch(1)
    features, labels = ds.make_one_shot_iterator().get_next()

    return features, labels


def main():
    EPOCHS = 1000

    config = tf.estimator.RunConfig(
        model_dir='./tf_log')
    # save_checkpoints_steps=LOG_INTERVAL,
    # save_summary_steps=LOG_INTERVAL)

    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        params={
            'data_loader': Shapes('./shapes-dataset', 1024, (224, 224))
        },
        config=config)

    for epoch in range(EPOCHS):
        estimator.train(input_fn)
        estimator.evaluate(input_fn, steps=10)


if __name__ == '__main__':
    main()
