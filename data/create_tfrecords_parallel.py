import concurrent.futures
import subprocess
import argparse
from pathlib import Path
import logging
from config import CONFIG_MAP
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def process_collection(config, mode, actual_process, num_of_processes):
    root_dirs = " ".join(config["root_dirs"])

    logging.info("python", Path(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "create_tf_records.py", f'--mode={mode}',
    f'--actual_process={actual_process}', f'--num_of_processes={num_of_processes}', f'--num_of_shards={config["num_of_shards"]}',
    f'--logdir={config["logdir"]}', f'--root_dirs={root_dirs}', f'--tfrecords_dir={config["tfrecords_dir"]}',
    f'--validation_ratio={config["validation_ratio"]}', f'--image_size={config["image_size"]}')

    subprocess.check_call(["python", Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "create_tf_records.py", f'--mode={mode}',
                           f'--actual_process={actual_process}', f'--num_of_processes={num_of_processes}', f'--num_of_shards={config["num_of_shards"]}',
                           f'--logdir={config["logdir"]}', f'--root_dirs={root_dirs}', f'--tfrecords_dir={config["tfrecords_dir"]}',
                           f'--validation_ratio={config["validation_ratio"]}', f'--image_size={config["image_size"]}'])

if __name__ == '__main__':

    REGISTERED_CONFIG_KEYS = "".join(map(lambda s: f"  {s}\n", CONFIG_MAP.keys()))

    parser = argparse.ArgumentParser(
        description=f"""
    Create tfrecords according to the given config settings.

    Registered config_key values:
    {REGISTERED_CONFIG_KEYS}""",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--config_key', required=True, help="Key to use while looking up "
                                           "configuration from the CONFIG_MAP dictionary.")
    parser.add_argument('--max_workers', required=True, help="Number of processes running in parallel.")
    args = parser.parse_args()

    CONFIG = CONFIG_MAP[args.config_key]

    divide_collection = {
        'train': [(actual_process, CONFIG["num_of_processes_train"]) for actual_process in
                  range(CONFIG["num_of_processes_train"])],
        'val': [(actual_process, CONFIG["num_of_processes_val"]) for actual_process in
                range(CONFIG["num_of_processes_val"])],
    }

    # We can use a with statement to ensure threads are cleaned up promptly
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        # Start the load operations and mark each future with its collection
        future_to_collection = {executor.submit(process_collection, CONFIG, mode, actual_process, num_of_processes):
                                    f'{mode}_{actual_process}_{num_of_processes}' for mode in divide_collection
                                for actual_process, num_of_processes in divide_collection[mode]}
        for future in concurrent.futures.as_completed(future_to_collection):
            process_id = future_to_collection[future]
            logging.info(process_id)
            try:
                future.result()
            except Exception as exc:
                logging.info('%r generated an exception: %s' % (process_id, exc))
            else:
                logging.info('%r done!' % (process_id))