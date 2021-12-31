import tensorflow as tf

class TfExampleDecoder:
    """Tensorflow Example proto decoder."""

    def __init__(self):
        self._keys_to_features = {
            'image/encoded':
                tf.io.FixedLenFeature((), tf.string),
            'image/filename':
                tf.io.FixedLenFeature((), tf.string),
        }

    def _decode_image(self, content, channels):
      return tf.image.decode_jpeg(content, channels)

    def decode(self, serialized_example):
        parsed_tensors = tf.io.parse_single_example(
            serialized=serialized_example, features=self._keys_to_features)
        image = self._decode_image(parsed_tensors[f'image/encoded'], 3)
        return image