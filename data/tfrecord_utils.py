import tensorflow as tf
import contextlib2

def count_examples_in_tfrecord(record_path):
    dataset = tf.data.TFRecordDataset(record_path)
    # count the examples by reduce
    return dataset.reduce(np.int64(0), lambda x, _: x + 1)

def count_examples_in_tfrecords_list(records_list):
    num_of_examples = 0
    for record_path in records_list:
        num_of_examples += count_examples_in_tfrecord(record_path)
    return num_of_examples

def open_sharded_output_tfrecords(exit_stack, base_path, num_shards):
  """Opens all TFRecord shards for writing and adds them to an exit stack.

  Args:
    exit_stack: A context2.ExitStack used to automatically closed the TFRecords
      opened in this function.
    base_path: The base path for all shards
    num_shards: The number of shards

  Returns:
    The list of opened TFRecords. Position k in the list corresponds to shard k.
  """
  tf_record_output_filenames = [
      '{}-{:05d}-of-{:05d}'.format(base_path, idx, num_shards)
      for idx in range(num_shards)
  ]

  tfrecords = [
      exit_stack.enter_context(tf.python_io.TFRecordWriter(file_name))
      for file_name in tf_record_output_filenames
  ]

  return tfrecords

def write_examples_to_sharded_records(examples, output_filebase, num_shards):
    with contextlib2.ExitStack() as tf_record_close_stack:
      output_tfrecords = open_sharded_output_tfrecords(
          tf_record_close_stack, output_filebase, num_shards)
      example_idx = 0
      for example in examples:
        output_shard_index = example_idx % num_shards
        output_tfrecords[output_shard_index].write(example.SerializeToString())
        example_idx += 1
    return example_idx

def write_examples_to_tfrecord(examples, record_path):
    random.shuffle(examples)
    with tf.io.TFRecordWriter(record_path) as writer:
        for example in examples:
            writer.write(example.SerializeToString())

def int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def int64_list_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def bytes_list_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def float_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def float_list_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))