import tensorflow as tf
import numpy as np


def model_fn(features, labels, mode):
    net = tf.reshape(features['x'], (-1, 784))
    logits = tf.layers.dense(net, 10)
    predictions = {'classes': tf.argmax(input=logits, axis=1),
                   'probabilities': tf.nn.softmax(logits)}
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    loss = tf.losses.sparse_softmax_cross_entropy(labels, logits)
    if mode == tf.estimator.ModeKeys.TRAIN:
        train_op = tf.train.\
            AdamOptimizer(0.001).\
            minimize(loss, tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss,
                                          train_op=train_op)
    if mode == tf.estimator.ModeKeys.EVAL:
        eval_metric_ops = {
            'accuracy': tf.metrics.accuracy(labels, predictions['classes']),
        }
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss,
                                          eval_metric_ops=eval_metric_ops)


def main():
    (train_images, train_labels), (test_images, test_labels) = tf.keras.\
        datasets.mnist.load_data()

    classifier = tf.estimator.Estimator(model_fn=model_fn,
                                        model_dir='model_dir')
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'x': np.asarray(train_images / 255, np.float32)},
        y=np.asarray(train_labels, np.int32),
        batch_size=100,
        num_epochs=None,
        shuffle=True,
    )
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'x': np.asarray(test_images / 255, np.float32)},
        y=np.asarray(test_labels, np.int32),
        num_epochs=1,
        shuffle=False,
    )
    while True:
        for i in range(10):
            classifier.train(input_fn=train_input_fn, steps=200)
        print(classifier.evaluate(input_fn=eval_input_fn))


if __name__ == "__main__":
    main()
