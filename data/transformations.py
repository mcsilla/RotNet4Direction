import tensorflow as tf


def random_brightness(image):
    seed = tf.random.uniform(
        [2], minval=0, maxval=10, dtype=tf.dtypes.int32, seed=None, name=None
    )
    return tf.image.stateless_random_brightness(image, max_delta=0.1, seed=seed)

def random_contrast(image):
    seed = tf.random.uniform(
        [2], minval=0, maxval=10, dtype=tf.dtypes.int32, seed=None, name=None
    )
    return tf.image.stateless_random_contrast(image, lower=0.8, upper=2, seed=seed)

def random_hue(image):
    seed = tf.random.uniform(
        [2], minval=0, maxval=10, dtype=tf.dtypes.int32, seed=None, name=None
    )
    return tf.image.stateless_random_hue(image, max_delta=0.2, seed=seed)

def random_saturation(image):
    seed = tf.random.uniform(
        [2], minval=0, maxval=10, dtype=tf.dtypes.int32, seed=None, name=None
    )
    return tf.image.stateless_random_saturation(image, lower=0.2, upper=2.5, seed=seed)


def make_random_transformation(image):
    rand = tf.random.uniform(
        [], minval=0, maxval=1, dtype=tf.dtypes.float32, seed=None, name=None
    )

    borders = tf.constant([0.25, 0.5, 0.75, 1])

    hue_fn = lambda: random_hue(image)
    saturation_fn = lambda: random_saturation(image)
    brightness_fn = lambda: random_brightness(image)
    contrast_fn = lambda: random_contrast(image)

    image = tf.case([(tf.less(rand, borders[0]), hue_fn),
                            (tf.less(rand, borders[1]), saturation_fn),
                            (tf.less(rand, borders[2]), brightness_fn),
                            (tf.less(rand, borders[3]), contrast_fn)], exclusive=False)
    return image

def crop_random_square_and_resize(img, size):
    s = tf.shape(img)
    h, w = s[0], s[1]
    c = tf.minimum(w, h)
    w_start = tf.random.uniform(shape=(), minval=0, maxval=tf.math.maximum(0, w - h - 1) + 1, dtype=tf.int32)
    h_start = tf.random.uniform(shape=(), minval=0, maxval=tf.math.maximum(0, h - w - 1) + 1, dtype=tf.int32)
    square = img[h_start:h_start + c, w_start:w_start + c]
    return tf.image.resize(square, [size, size], method='lanczos3')


def crop_center_and_resize(img, size):
    s = tf.shape(img)
    h, w = s[0], s[1]
    c = tf.minimum(w, h)
    w_start = (w - c) // 2
    h_start = (h - c) // 2
    center = img[h_start:h_start + c, w_start:w_start + c]
    return tf.image.resize(center, [size, size], method='lanczos3')

def create_input_from_square_image(image, image_size):
    # create input from square image
    image = tf.image.resize(image, [image_size, image_size], method='lanczos3')
    label = tf.random.uniform(shape=(), maxval=4, dtype=tf.int32)
    image = tf.image.rot90(image, k=label)
    image = tf.cast(image, tf.float32) / 127.5 - 1
    return image, label

def create_input(image, image_size):
    # create input from arbitrary image
    image = crop_random_square_and_resize(image, image_size)
    image = make_random_transformation(image)
    label = tf.random.uniform(shape=(), maxval=4, dtype=tf.int32)
    image = tf.image.rot90(image, k=label)
    image = tf.cast(image, tf.float32) / 127.5 - 1
    return image, label
