"""CTC loss for training the lip-reading model."""
import tensorflow as tf


def ctc_loss_fn(y_true, y_pred):
    """Connectionist Temporal Classification loss.

    Args:
        y_true: int label ids of shape [batch, max_text_length], right-padded
            with 0 (the reserved padding id). Real label length is inferred from
            the count of non-zero entries per row.
        y_pred: softmax probabilities of shape [batch, time, BLANK_TOKEN + 1].
            The CTC blank is the final class (index BLANK_TOKEN), as expected by
            tf.keras.backend.ctc_batch_cost.

    Returns:
        Scalar mean CTC loss over the batch.
    """
    y_true = tf.cast(y_true, tf.int64)
    batch_size = tf.shape(y_pred)[0]
    time_steps = tf.shape(y_pred)[1]

    label_length = tf.reduce_sum(
        tf.cast(tf.not_equal(y_true, 0), tf.int64), axis=-1, keepdims=True
    )
    input_length = tf.cast(tf.fill([batch_size, 1], time_steps), tf.int64)

    loss = tf.keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
    return tf.reduce_mean(loss)
