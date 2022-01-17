from contextlib import contextmanager
import random

@contextmanager
def lock_dir(dir_path):
    try:
        dir_path.mkdir(parents=True)
    except FileExistsError:
        yield False
        return
    try:
        yield True
    finally:
        dir_path.rmdir()


def divede_files(dir_path, validation_ratio):
    """
        Group the files of the directory to train and validation according to the validation_ratio.
    """
    train_files = []
    val_files = []
    for file_path in dir_path.glob("*.jpg"):
        if random.random() < validation_ratio:
            val_files.append(file_path)
        else:
            train_files.append(file_path)
    return train_files, val_files