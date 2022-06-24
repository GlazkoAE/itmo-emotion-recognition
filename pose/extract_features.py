"""
This script generates extracted features for each video, which other
models make use of.

You can change you sequence length and limit to a set number of classes
below.

class_limit is an integer that denotes the first N classes you want to
extract features from. This is useful is you don't want to wait to
extract all 101 classes. For instance, set class_limit = 8 to just
extract features for the first 8 (alphabetical) classes in the dataset.
Then set the same number when training models.
"""
import argparse
import os.path

import numpy as np
from tqdm import tqdm

from model.data import DataSet
from model.extractor import Extractor


def main(inputs):
    # Set defaults.
    seq_length = inputs.seq_length
    saved_model = inputs.saved_model

    # Get the dataset.
    data = DataSet(seq_length=seq_length)

    # get the model.
    model = Extractor(weights=saved_model)

    # Loop through data.
    pbar = tqdm(total=len(data.data))
    for video in data.data:

        # Get the path to the sequence for this video.
        path = os.path.join(
            "data",
            "sequences",
            video[2] + "-" + video[3] + "-" + str(seq_length) + "-features",
        )  # numpy will auto-append .npy

        # Check if we already have it.
        if os.path.isfile(path + ".npy"):
            pbar.update(1)
            continue

        # Get the frames for this video.
        frames = data.get_frames_for_sample(video)

        # Now downsample to just the ones we need.
        frames = data.rescale_list(frames, seq_length)

        # Now loop through and extract features to build the sequence.
        sequence = []
        for image in frames:
            features = model.extract_from_path(image)
            sequence.append(features)

        # Save the sequence.
        np.save(path, sequence)

        pbar.update(1)

    pbar.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract features with InceptionV3 from videoframes "
        "and save them to .npy"
    )
    parser.add_argument(
        "--seq_length",
        type=int,
        default=30,
        help="number of frame to use for prediction",
    )
    parser.add_argument(
        "--saved_model",
        type=str,
        default=None,
        help="path to saved model for continue train it",
    )

    args = parser.parse_args()

    main(args)
