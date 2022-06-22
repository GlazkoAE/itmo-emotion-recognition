"""
Train our RNN on extracted features or images.
"""
import argparse
import os.path
import time

from keras.callbacks import (CSVLogger, EarlyStopping, ModelCheckpoint,
                             TensorBoard)

import wandb
from model.data import DataSet
from model.models import ResearchModels


def train(seq_length, saved_model=None, batch_size=32, nb_epoch=100, early_stop=10):

    model = "lstm"
    data_type = "features"

    checkpointer = ModelCheckpoint(
        filepath=os.path.join(
            "data",
            "checkpoints",
            model + "-" + data_type + ".{epoch:03d}-{val_loss:.3f}.hdf5",
        ),
        verbose=1,
        save_best_only=True,
    )

    tb = TensorBoard(log_dir=os.path.join("data", "logs", model))

    wandb.init(project="Emotion-intense", sync_tensorboard=True, tensorboard=tb)

    # Helper: Stop when we stop learning.
    early_stopper = EarlyStopping(patience=early_stop)

    # Helper: Save results.
    timestamp = time.time()
    csv_logger = CSVLogger(
        os.path.join(
            "data", "logs", model + "-" + "training-" + str(timestamp) + ".log"
        )
    )

    # Get the data and process it.
    data = DataSet(
        seq_length=seq_length,
    )

    # Get samples per epoch.
    # Multiply by 0.7 to attempt to guess how much of data.data is the train set.
    steps_per_epoch = (len(data.data) * 0.7) // batch_size

    # Get the model.
    rm = ResearchModels(1, model, seq_length, saved_model)

    # Get generators.
    generator = data.frame_generator(batch_size, "train", data_type)
    val_generator = data.frame_generator(batch_size, "val", data_type)

    # Use fit generator.
    rm.model.fit_generator(
        generator=generator,
        steps_per_epoch=steps_per_epoch,
        epochs=nb_epoch,
        verbose=1,
        callbacks=[tb, early_stopper, csv_logger, checkpointer],
        validation_data=val_generator,
        validation_steps=30,
        workers=4,
    )


def main(inputs):
    seq_length = inputs.seq_length
    batch_size = inputs.batch_size
    nb_epoch = inputs.nb_epoch
    early_stop = inputs.early_stop
    saved_model = inputs.saved_model

    train(
        seq_length=seq_length,
        saved_model=saved_model,
        batch_size=batch_size,
        nb_epoch=nb_epoch,
        early_stop=early_stop,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train the lstm model for regression task with features "
        "extracted with extract_features.py"
    )
    parser.add_argument(
        "--seq_length",
        type=int,
        default=30,
        help="number of frame to use for prediction",
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument(
        "--nb_epoch", type=int, default=1000, help="maximum number of train epochs"
    )
    parser.add_argument(
        "--early_stop",
        type=int,
        default=10,
        help="stop when model not trains for N epochs",
    )
    parser.add_argument(
        "--saved_model",
        type=str,
        default=None,
        help="path to saved model for continue train it",
    )

    args = parser.parse_args()

    main(args)
