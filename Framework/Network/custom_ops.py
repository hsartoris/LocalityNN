import tensorflow as tf

def modified_mse(labels, predictions):
    """This is... weird"""
    mse = tf.losses.mean_squared_error(labels, predictions,
            reduction = tf.losses.Reduction.NONE)

    return tf.divide(tf.reduce_sum(mse),
            tf.reduce_sum(labels))
