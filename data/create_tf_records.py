import random
import tqdm
import glob
import cv2
import numpy as np
import sys
from pathlib import Path
import io
from contextlib import contextmanager
import logging
import argparse
import tensorflow as tf

from train.tfrecord_utils import bytes_feature, int64_feature, write_examples_to_sharded_records


class CorruptImageError(Exception):
    """Exception raised if image data is corrupted.

    Attributes:
        file_path -- input image path which caused the error
        message -- explanation of the error
    """

    def __init__(self, file_path, message):
        self.file_path = file_path
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f'{self.message}: {self.file_path}'


class ImageIsNone(CorruptImageError):
    """Exception raised if read image is None.

    Attributes:
        file_path -- input image path which caused the error
        message -- explanation of the error
    """

    def __init__(self, file_path, message):
        super().__init__(file_path, message)


@contextmanager
def catch_stderr(callback_on_stderr):
    orig_stderr = sys.stderr
    temp_stderr = io.StringIO()
    sys.stderr = temp_stderr
    try:
        yield
    finally:
        sys.stderr = orig_stderr
        print(temp_stderr.getvalue(), file=sys.stderr, end='')
        callback_on_stderr(temp_stderr.getvalue())

def check_stderr(content, file_path):
    if('corrupt' in content.lower() or 'premature' in content.lower()):
        raise CorruptImageError(file_path, content)

def crop_image_center(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    size = min(height, width)
    y_min = (height - size) // 2
    x_min = (width - size) // 2
    image_out = image[y_min: y_min + size, x_min:x_min + size]
    return image_out

def make_image_grayscale(image: np.ndarray, mode = 'rgb') -> np.ndarray:
    image_out = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    if mode == 'rgb':
        image_out = np.repeat(np.expand_dims(image_out, axis=-1), repeats=3, axis=-1)
    return image_out

def resize_image(image: np.ndarray, size) -> np.ndarray:
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_LANCZOS4)

def create_tf_example(image_path, size):
    image = cv2.imread(image_path)
    if image is None:
        raise ImageIsNone(image_path, 'Image is None')
    with catch_stderr(lambda content: check_stderr(content, image_path)):
        image_center = crop_image_center(image)
        image_center = make_image_grayscale(image_center)

    image = resize_image(image_center, size=size)

    encoded_image = cv2.imencode('.jpg', image)[1].tobytes()

    feature_dict = {
        'image/filename':
            bytes_feature(str(image_path).encode('utf8')),
        'image/encoded':
            bytes_feature(encoded_image),

    }
    return tf.train.Example(features=tf.train.Features(feature=feature_dict))

def yield_tf_example(image_paths, image_size):
    num_of_examples = 0
    for image_path in tqdm.tqdm(image_paths):
        logging.info(f'Processing file: {image_path}')
        try:
            yield create_tf_example(image_path, image_size)
            num_of_examples += 1
        except (CorruptImageError, ImageIsNone) as e:
            logging.error(f'Error processing {image_path}: {str(e)}')
            continue
    logging.info(f'Number of image paths: {len(image_paths)}')
    logging.info(f'Number of examples: {num_of_examples}')

def select_actual_paths(all_paths, num_of_processes, actual_process):
    divided_paths = [[] for _ in range(num_of_processes)]
    for i, path in enumerate(all_paths):
        divided_paths[i % num_of_processes].append(path)
    return divided_paths[actual_process]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate tfrecords for train correct orientation model.')

    parser.add_argument('--mode', type=str, choices=('train', 'val'), default='train',
                        help='Generate records for train or validation.')
    parser.add_argument('--actual_process', type=int, default=0,
                        help='Serial number of the actual process, must be between 0 and the number of processes - 1')
    parser.add_argument('--num_of_processes', type=int, default=1,
                        help='The number of processes.')
    parser.add_argument('--num_of_shards', type=int, required=True,
                        help='The number of tfrecord shards in the actual process.')
    parser.add_argument('--validation_ratio', type=float, default=0,
                        help='The ratio of validation images. It is between 0 and 1.')
    parser.add_argument('--logdir', type=Path, required=True,
                        help='Path of log directory. Log file will be created with name "tfrecord_generation_log_{mode}_{actual_process}.log".')
    parser.add_argument('--root_dirs', nargs="+", type=Path, required=True,
                        help='Path(s) of root dirs. From these dirs all jpg files will be processed')
    parser.add_argument('--tfrecords_dir', type=Path, required=True,
                        help='Path of directory of output tf records')
    parser.add_argument('--image_size', type=int, default=512,
                        help='Size of the images in the record')
    args = parser.parse_args()

    logging.basicConfig(filename=args.logdir / f'log_{args.mode}_{args.actual_process}.log', filemode='w', level=logging.INFO)

    file_patterns = [dir_path / "**/*.jpg" for dir_path in args.root_dirs]

    tfrec_prefix = args.tfrecords_dir


    img_files_all = sorted([file_path for file_pattern in file_patterns for file_path in glob.glob(str(file_pattern), recursive=True)])
    random.seed(42)
    random.shuffle(img_files_all)

    validation_ratio = args.validation_ratio
    val_image_paths = img_files_all[:int(len(img_files_all) * validation_ratio)]
    train_image_paths = img_files_all[int(len(img_files_all) * validation_ratio):]

    actual_paths_all = train_image_paths if args.mode == 'train' else val_image_paths

    actual_paths = select_actual_paths(sorted(actual_paths_all), args.num_of_processes, args.actual_process)
    random.shuffle(actual_paths)
    write_examples_to_sharded_records(examples=yield_tf_example(actual_paths, args.image_size),
                                      output_filebase=str(tfrec_prefix / f'{args.mode}_{args.actual_process}_{args.num_of_processes}'),
                                      num_shards=args.num_of_shards)
