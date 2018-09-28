from typing import List, Tuple
import .Log as log
import tensorflow as tf
import numpy as np
from os import listdir
from os.path import isdir, isfile, join
import sys

def make_dataset(data_path: str, batchsize: int, data_dir: str = 'data',
        labels_dir: str = 'labels') -> Tuple[tf.data.Iterator, tf.Operation]:
    """Loads data from given folder and creates a tensorflow dataset.

    Assumes that data_path is relative to LocalityNN/

    Also assumes that naming is equivalent for data/inputs, and that inputs are 
    CSVs.
    """

    if not isdir(data_path):
        log.error("Given data path does not exist:", data_path)
        sys.exit()

    # override with full path for ease of use
    data_dir: str = join(data_path, data_dir)
    labels_dir: str = join(data_path, labels_dir)

    if not isdir(data_dir):
        log.error("Data entries directory not present within given data_path:",
                data_dir)
        sys.exit()

    if not isdir(labels_dir):
        log.error("Label entries diretory not present within given data_path:",
                labels_dir)
        sys.exit()

    data_names: List[str] = [f for f in listdir(data_dir) if \
            isfile(join(data_dir, f))]

    data: List[np.ndarray] = []
    labels: List[np.ndarray] = []

    for data_name in data_names:
        # in case data is in NPZ format
        label_name: str = data_name.split('.')[0] + ".csv"
        if not isfile(join(labels_dir, label_name)):
            log.warning("Labels for data file", data_name, "not found;",
                    "skipping.")
            continue

        # TODO: support for loading NPZ
        data.append(np.loadtxt(join(data_dir, data_name), delimiter=','))
        # flatten the adjacency matrix to match (1 x n^2) dims of network out
        labels.append(np.loadtxt(join(labels_dir, label_name),
            delimiter=',').flatten())

    dataset: tf.data.Dataset = tf.data.Dataset.from_tensor_slices((data, 
        labels))
    dataset.shuffle(len(data)*2).repeat().batch(batchsize)
    iterator: tf.data.Iterator = tf.data.Iterator.from_structure(
            dataset.output_types, dataset.output_shape)

    return iterator, iterator.make_initializer(dataset)

