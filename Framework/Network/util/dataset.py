from typing import List, Tuple, Union
from ..util import Log as log
import tensorflow as tf
import numpy as np
import os
from os import listdir, getcwd
from os.path import isdir, isfile, join
import sys
from random import shuffle

def parse_tfrecord(example):
    """Parses tfrecord Examples into... something.
    """
    #TODO: parameterize dtype
    example_fmt: Dict[str, any] = {
            "train/entry": tf.FixedLenFeature([50],
                #tf.float32, allow_missing = True),
                tf.float32),
            "train/label": tf.FixedLenFeature([25],
                #tf.float32, allow_missing = True)
                tf.float32)
            }
    parsed = tf.parse_single_example(example, example_fmt)
    # TODO: reshape data to appropriate dimensionality
    # or maybe that needs to happen in model_fn?
    return parsed['train/entry'], parsed['train/label']

def load_dataset(data_dir: str, name: str, buffer_size: int, batchsize: int,
        prefetch_buffer: int = 1, num_parallel: int = 2):
    """Loads a TFRecord encoded dataset from data_dir, where the files in the 
    dataset start with name, e.g. name=train -> train-001.tfrecord

    prefetch_buffer defines how many BATCHES to prefetch

    """
    
    files = tf.data.Dataset.list_files(join(data_dir, name) + "*.tfrecords")
    dataset = files.interleave(tf.data.TFRecordDataset, 2)
    # shuffle dataset and repeat infinitely
    dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(buffer_size))
    # map parse function onto dataset and batch
    dataset = dataset.apply(tf.contrib.data.map_and_batch(
        map_func = parse_tfrecord, batch_size = batchsize))
    #dataset = dataset.map(map_func = parse_tfrecord)

    # prefetch prefetch_buffer batches
    dataset = dataset.prefetch(buffer_size = prefetch_buffer)
    return dataset

def make_tfrecords(data: List[np.ndarray],
        labels: Union[List[np.ndarray], np.ndarray], save_dir: str,
        valid_split: float = .2, test_split: float = .2,
        do_shuffle: bool = True) -> None:
    """Takes either paired lists of data and labels or list of data and single 
    label and creates a TFRecords file at specified location.

    save_dir is relative to the cwd of the process.
    """
    def _float_feature(value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))

    def _write_range(save_dir, name, data, labels, paired, idxrange):
        fname = join(save_dir, name + ".tfrecords")
        log.info("Creating", fname)
        if not paired: label = labels.flatten()

        writer: tf.python_io.TFRecordWriter = tf.python_io.TFRecordWriter(fname)
        for i in idxrange:
            # TODO: nice pretty arrow
            if not i % 100:
                log.info(name, "data: {}".format(i))

            entry: np.ndarray = data[i].flatten()
            if paired: label = labels[i].flatten()

            feature: Dict[str, any] = {
                    name + '/entry': _float_feature(entry),
                    name + '/label': _float_feature(label)
                }

            example = tf.train.Example(
                    features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())

        writer.close()
        sys.stdout.flush()
        log.info(fname, "complete")

    # actual logic begins here

    if (valid_split < 0 or valid_split > 1 or test_split < 0 or test_split > 1
            or valid_split + test_split > 1):
        log.critical("Improper values for valid_split or test_split:", 
                valid_split, test_split)
        sys.exit()

    if not isdir(save_dir):
        log.info("Output directory", join(getcwd(), save_dir),
                "does not exist; creating.")
        os.makedirs(save_dir)
    else:
        log.warning("Output directory", join(getcwd(), save_dir),
                "already exists; overwriting!")
    
    paired: bool
    if isinstance(labels, list):
        # one label per entry
        paired = True
        log.info("Using paired data and label lists")
        if not len(data) == len(labels):
            log.critical("Data and label lists do not match length; aborting.")
            sys.exit()
    else:
        # one label for all
        log.info("Using one label for all entries")
        paired = False

    train_max: int = int(len(data) * (1 - (test_split + valid_split)))
    valid_max: int = int(len(data) * (1 - test_split))
    log.info("Out of", len(data), "total entries, using", train_max,
            "for training and", valid_max - train_max, "for validation.")

    if do_shuffle:
        log.info("Shuffling dataset")
        # TODO: type
        if paired:
            tmp = list(zip(data, labels))
            shuffle(tmp)
            data, labels = zip(*tmp)
        else:
            shuffle(data)

    # write train examples
    _write_range(save_dir, "train", data, labels, paired, range(train_max))

    # write validation examples
    _write_range(save_dir, "validation", data, labels, paired,
            range(train_max, valid_max))

    # write testing examples
    _write_range(save_dir, "testing", data, labels, paired,
            range(valid_max, len(data)))
    
