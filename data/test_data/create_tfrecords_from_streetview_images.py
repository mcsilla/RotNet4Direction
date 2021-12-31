import logging
import os
import wget
import zipfile
import argparse
import concurrent.futures
import subprocess
from pathlib import Path

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import CONFIG_MAP

def download(output_dir, part_index):
    filename = f'part{part_index}.zip'
    url = f"http://www.cs.ucf.edu/~aroshan/index_files/Dataset_PitOrlManh/zipped images/{filename}"
    logging.info(f'Downloading: {url} to {output_dir}')
    filepath = wget.download(url, out=output_dir)

    logging.info(f'Extracting: {filename}')
    with zipfile.ZipFile(filepath, 'r') as z:
        z.extractall(output_dir)
    os.remove(filepath)

def delete_upward_view(images_dir):
    logging.info('Deleting upward view files...')
    for filepath in Path(images_dir).iterdir():
        if str(filepath)[-5] == str(5):
            os.remove(filepath)

def process_collection(config, mode, actual_process, num_of_processes):
    root_dirs = " ".join(config["root_dirs"])

    logging.info(f'running command:\n' 
                 f'python {str(Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "create_tf_records.py")}'
                 f'--mode={mode} --actual_process={actual_process} --num_of_processes={num_of_processes}' 
                 f'--num_of_shards={config["num_of_shards"]} --logdir={config["logdir"]} --root_dirs={root_dirs}' 
                 f'--tfrecords_dir={config["tfrecords_dir"]} --validation_ratio={config["validation_ratio"]}' 
                 f'--image_size={config["image_size"]}')

    subprocess.check_call(["python", Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "create_tf_records.py", f'--mode={mode}',
                           f'--actual_process={actual_process}', f'--num_of_processes={num_of_processes}', f'--num_of_shards={config["num_of_shards"]}',
                           f'--logdir={config["logdir"]}', f'--root_dirs={root_dirs}', f'--tfrecords_dir={config["tfrecords_dir"]}',
                           f'--validation_ratio={config["validation_ratio"]}', f'--image_size={config["image_size"]}'])


def check_valid_part_index(value):
    ivalue = int(value)
    if ivalue < 1 or ivalue > 10:
        raise argparse.ArgumentTypeError("Value of --part_index should be between 1 and 10")
    return ivalue


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='\n\r%(levelname)s\n\r%(message)s')

    REGISTERED_CONFIG_KEYS = "".join(map(lambda s: f"  {s}\n", CONFIG_MAP.keys()))

    parser = argparse.ArgumentParser(
        description=f"""
    Download a part from google streetview database and create tfrecords according to the given config settings.

    Registered config_key values:
    {REGISTERED_CONFIG_KEYS}""",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--config_key', required=True, help="Key to use while looking up "
                                           "configuration from the CONFIG_MAP dictionary.")
    parser.add_argument('--part_index', default=1, type=check_valid_part_index, help="Index of google streetview zipped images pack. "
                                                        "It has to be betweeen 1 and 10")
    args = parser.parse_args()

    CONFIG = CONFIG_MAP[args.config_key]

    assert len(CONFIG['root_dirs']) == 1

    if len(list(Path(CONFIG['root_dirs'][0]).iterdir())) == 0:
        download(CONFIG['root_dirs'][0], args.part_index)
        delete_upward_view(CONFIG['root_dirs'][0])

    divide_collection = {
        'train': [(actual_process, CONFIG["num_of_processes_train"]) for actual_process in
                  range(CONFIG["num_of_processes_train"])],
        'val': [(actual_process, CONFIG["num_of_processes_val"]) for actual_process in
                range(CONFIG["num_of_processes_val"])],
    }

    # We can use a with statement to ensure threads are cleaned up promptly
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        # Start the load operations and mark each future with its collection
        future_to_collection = {executor.submit(process_collection, CONFIG, mode, actual_process, num_of_processes):
                                    f'{mode}_{actual_process}_{num_of_processes}' for mode in divide_collection
                                for actual_process, num_of_processes in divide_collection[mode]}
        for future in concurrent.futures.as_completed(future_to_collection):
            process_id = future_to_collection[future]
            logging.debug(process_id)
            try:
                future.result()
            except Exception as exc:
                logging.warning('%r generated an exception: %s' % (process_id, exc))
            else:
                logging.info('%r done!' % (process_id))

