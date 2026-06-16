"""Training script for the lip-reading model."""
import argparse

import tensorflow as tf

from .callbacks import LearningRateSchedule, ModelCheckpoint, ProduceExample
from .config import (
    EPOCHS,
    INITIAL_LEARNING_RATE,
    MODEL_SAVE_DIR,
    TRAIN_SIZE,
    VAL_SIZE,
)
from .dataset import build_vocab_lookup, create_dataset, prepare_dataset
from .losses import ctc_loss_fn
from .model import build_model, print_model_summary


def setup_gpu():
    """Enable GPU memory growth, or report that the CPU is being used."""
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPU configured: {len(gpus)} GPU(s) available")
        except RuntimeError as exc:
            print(f"GPU configuration error: {exc}")
    else:
        print("No GPU found, using CPU")


def main(video_pattern="data/S1/*.mpg", epochs=EPOCHS):
    """Train the model on clips matching video_pattern.

    The first TRAIN_SIZE clips form the training split and the remainder form
    the validation split. The split order is stable, so the two are disjoint.
    """
    setup_gpu()

    char_to_num, num_to_char = build_vocab_lookup()

    dataset = create_dataset(video_pattern, char_to_num, shuffle=True)
    # cache() keeps decoded/padded clips in memory so epochs after the first
    # don't re-read and re-decode every video; the training split is reshuffled
    # each epoch. Validation is capped at VAL_SIZE clips to avoid spending most
    # of each epoch validating.
    train_dataset = (
        prepare_dataset(dataset.take(TRAIN_SIZE))
        .cache()
        .shuffle(64, reshuffle_each_iteration=True)
    )
    val_dataset = prepare_dataset(dataset.skip(TRAIN_SIZE).take(VAL_SIZE)).cache()

    model = build_model()
    print_model_summary(model)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=INITIAL_LEARNING_RATE),
        loss=ctc_loss_fn,
    )

    callbacks = [
        LearningRateSchedule(),
        ModelCheckpoint(save_dir=MODEL_SAVE_DIR),
        ProduceExample(val_dataset=val_dataset, num_to_char=num_to_char),
    ]

    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=callbacks,
    )

    print("Training completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the lip-reading model.")
    parser.add_argument(
        "--video_pattern",
        default="data/S1/*.mpg",
        help="Glob pattern for video files.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
        help="Number of training epochs.",
    )
    args = parser.parse_args()
    main(video_pattern=args.video_pattern, epochs=args.epochs)
