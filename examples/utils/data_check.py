import tensorflow as tf
from google import protobuf
import os


def decode_record(record):
    """Decodes a record to a TensorFlow example."""
    seq_length = 512
    lm_length = 76
    name_to_features = {
        "input_ids": tf.io.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.io.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.io.FixedLenFeature([seq_length], tf.int64),
        "masked_lm_positions": tf.io.FixedLenFeature([lm_length], tf.int64),
        "masked_lm_ids": tf.io.FixedLenFeature([lm_length], tf.int64),
        "masked_lm_weights": tf.io.FixedLenFeature([lm_length], tf.float32),
        "next_sentence_labels": tf.io.FixedLenFeature([1], tf.int64),
    }
    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    example = tf.io.parse_single_example(record, name_to_features)
    for name in list(example.keys()):
        value = example[name]
        if value.dtype == tf.int64:
            value = tf.cast(value, tf.int32)
        example[name] = value
    return example

def preview_tfrecord(filepath):
    """Pretty prints a single record from a tfrecord file."""
    # dataset = tf.data.TFRecordDataset(os.path.expanduser(filepath))
    dataset = tf.data.TFRecordDataset(filepath)
    num_total = 0
    num_dense = 0
    early_end = 1000
    print(f"GEEZ TOTAL NUMBER OF RECORDS: {num_total}")
    print(f"GEEZ NUMBER OF DENSE RECORDS: {num_dense}")
    for item in dataset:
        if num_total >= early_end:
            break
        num_total += 1
        decoded = decode_record(item)
        print(decoded)
        break
        input_mask = decoded["input_mask"]
        num_valid = tf.reduce_sum(input_mask)
        if num_valid > 480:
            num_dense += 1
    print(f"GEEZ TOTAL NUMBER OF RECORDS: {num_total}")
    print(f"GEEZ NUMBER OF DENSE RECORDS: {num_dense}")

preview_tfrecord('gs://chenmoney-testing-east/bert-pretraining-data/shard_1.tfrecord')