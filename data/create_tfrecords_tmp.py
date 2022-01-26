import tqdm
import cv2
import numpy as np
import sys
from pathlib import Path
import io
from contextlib import contextmanager
import logging
import argparse
import tensorflow as tf
import json
import time

from utils import lock_dir, divede_files
from tfrecord_utils import bytes_feature, int64_feature, write_examples_to_sharded_records

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

def resize_image_fixed_shorter_side(image, min_side):
    if image.shape[0] <= image.shape[1]:
        height = min_side
        width = int(round(image.shape[1] * (min_side / image.shape[0])))
    if image.shape[0] > image.shape[1]:
        width = min_side
        height = int(round(image.shape[0] * (min_side / image.shape[1])))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_LANCZOS4)

def resize_image(image: np.ndarray, size) -> np.ndarray:
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_LANCZOS4)

def create_tf_example(image_path, size):
    image_path = str(image_path)
    image = cv2.imread(image_path)
    if image is None:
        raise ImageIsNone(image_path, 'Image is None')
    with catch_stderr(lambda content: check_stderr(content, image_path)):
        image_resized = resize_image_fixed_shorter_side(image, min_side=size)
        image_resized = make_image_grayscale(image_resized)

    encoded_image = cv2.imencode('.jpg', image_resized)[1].tobytes()

    feature_dict = {
        'image/filename':
            bytes_feature(image_path.encode('utf8')),
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate tfrecords for train correct orientation model.')
    parser.add_argument('--config', type=str, required=True)

    args, _ = parser.parse_known_args()
    with open(args.config) as f:
        config = json.load(f)
        input_roots = config["roots_list"]
        output_dir = Path(config["output_root"])
        validation_ratio = config["validation_ratio"]
        image_size = config["image_size"]
        logdir = Path(config["logdir"])

    output_dir.mkdir(parents=True, exist_ok=True)
    logdir.mkdir(parents=True, exist_ok=True)

    parent_folders = set()
    for root in input_roots:
        image_paths = Path(root).glob("**/*.jpg")
        parent_folders.update([image_path.parent for image_path in image_paths])

    logindex = max([0] + [int(logfile.name[8:-4]) for logfile in logdir.iterdir()]) + 1

    filehandler = logging.FileHandler(logdir / f"process_{logindex}.log", 'w')
    stream_handler = logging.StreamHandler()
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s',
                        handlers=[filehandler, stream_handler])

    start_time = time.time()
    num_of_all_train_examples = 0
    num_of_all_val_examples = 0
    for dir_path in parent_folders:
        last_time_checkpoint = time.time()
        lock_dir_name = "--".join(dir_path.parts[1:])
        lock_dir_path = output_dir / f"{lock_dir_name}--lock"
        ready_dir_path = output_dir / f"{lock_dir_name}--ready"
        if ready_dir_path.exists():
            logging.info(f"Skipping {dir_path}: directory is processed...")
            continue
        with lock_dir(lock_dir_path) as locking_succeeded:
            if locking_succeeded:
                logging.info(f"Locking direrctory: {lock_dir_path}")
                logger = logging.getLogger()
                logger.removeHandler(stream_handler)
                train_files, val_files = divede_files(dir_path, validation_ratio)
                num_of_shards_train = len(train_files) // 1000 + 1
                num_of_shards_val = len(val_files) // 1000 + 1
                logging.info("Generating train records...")
                num_of_train_examples = write_examples_to_sharded_records(
                    examples=yield_tf_example(train_files, image_size),
                    output_filebase=f"{str(output_dir)}/train_{lock_dir_name}",
                    num_shards=num_of_shards_train
                )
                num_of_all_train_examples += num_of_train_examples
                logging.info(f"{num_of_train_examples} train examples from {dir_path}")
                logging.info("Generating validation records...")
                num_of_val_examples = write_examples_to_sharded_records(
                    examples=yield_tf_example(val_files, image_size),
                    output_filebase=f"{str(output_dir)}/val_{lock_dir_name}",
                    num_shards=num_of_shards_val
                )
                num_of_all_val_examples += num_of_val_examples
                logging.info(f"{num_of_val_examples} val examples from {dir_path}")
                ready_dir_path.mkdir()
                logger.addHandler(stream_handler)
            else:
                logging.info(f"Skipping {dir_path}: lock dir exists...")

        logging.info(f"Process {dir_path} in {time.time() - last_time_checkpoint} seconds")

    logging.info(f"Runtime: {time.time() - start_time} seconds")
    logging.info(f"{num_of_all_train_examples} train examples has generated")
    logging.info(f"{num_of_all_val_examples} validation examples has generated")
