"""Conv3D + Bidirectional LSTM + CTC model architecture."""
import tensorflow as tf

from . import config


def build_model():
    """Build the lip-reading model.

    Input:  [batch, TARGET_FRAMES, VIDEO_HEIGHT, VIDEO_WIDTH, 1]
    Output: [batch, TARGET_FRAMES, BLANK_TOKEN + 1] softmax over character ids
            (1..VOCAB_SIZE), the reserved padding id 0, and the CTC blank.

    Pooling is spatial only (pool_size (1, 2, 2)) so the temporal dimension is
    preserved end-to-end, giving one output timestep per input frame for CTC.
    """
    inputs = tf.keras.Input(
        shape=(config.TARGET_FRAMES, config.VIDEO_HEIGHT, config.VIDEO_WIDTH, 1)
    )

    x = inputs
    for filters in config.CONV3D_FILTERS:
        x = tf.keras.layers.Conv3D(filters, 3, padding="same", activation="relu")(x)
        x = tf.keras.layers.MaxPool3D(pool_size=(1, 2, 2))(x)

    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())(x)

    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(config.LSTM_UNITS, return_sequences=True)
    )(x)
    x = tf.keras.layers.Dropout(config.DROPOUT_RATE)(x)

    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(config.LSTM_UNITS, return_sequences=True)
    )(x)
    x = tf.keras.layers.Dropout(config.DROPOUT_RATE)(x)

    outputs = tf.keras.layers.Dense(config.BLANK_TOKEN + 1, activation="softmax")(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name="lipnet")


def print_model_summary(model):
    """Print the Keras summary plus the total parameter count."""
    model.summary()
    print(f"Total parameters: {model.count_params():,}")
