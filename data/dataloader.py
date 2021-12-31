import tensorflow as tf
from .tf_example_decoder import TfExampleDecoder
from .transformations import create_input

class GenericDataLoader:

    def __init__(self, configs):
        self.configs = configs
        self._parser_fn = TfExampleDecoder().decode
        self._create_input_fn = create_input

    def get_dataset(self, dataset_type):
        record_names_dataset = tf.data.Dataset.from_tensor_slices(self.configs['tf_records'])
        record_names_dataset.shuffle(len(self.configs['tf_records']), reshuffle_each_iteration=True)
        dataset = record_names_dataset.interleave(
            map_func=tf.data.TFRecordDataset,
            cycle_length=8, # the number of input elements that will be processed concurrently
            num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.map(self._parser_fn)
        dataset = dataset.map(self._create_input_fn)
        dataset = dataset.shuffle(5000, reshuffle_each_iteration=True)
        dataset = dataset.batch(self.configs['batch_size'], drop_remainder=True)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return dataset