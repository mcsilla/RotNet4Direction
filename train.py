import argparse
from argparse import RawTextHelpFormatter

print("[-] Importing tensorflow...")
import tensorflow as tf
print(f"[+] Done! Tensorflow version: {tf.version.VERSION}")

print("[-] Importing RotNet4 Trainer class...")
from train.trainer import Trainer

print("[-] Importing config files...")
from config.train import CONFIG_MAP


if __name__ == "__main__":
    REGISTERED_CONFIG_KEYS = "".join(map(lambda s: f"  {s}\n", CONFIG_MAP.keys()))

    PARSER = argparse.ArgumentParser(
        description=f"""
            Runs trainer with the given config setting.
            
            Registered config_key values:
            {REGISTERED_CONFIG_KEYS}""",
        formatter_class=RawTextHelpFormatter
    )
    PARSER.add_argument('--config_key', help="Key to use while looking up "
                        "configuration from the CONFIG_MAP dictionary.")
    ARGS = PARSER.parse_args()

    CONFIG = CONFIG_MAP[ARGS.config_key]

    tf.config.set_visible_devices(tf.config.list_physical_devices("GPU")[0:7], "GPU")

    print('GPU Devices: {}'.format([device.name for device in tf.config.list_physical_devices("GPU")]))
    print('Phisical Devices: {}'.format([device.name for device in tf.config.list_physical_devices()]))
    print('Logical Devices: {}'.format([device.name for device in tf.config.list_logical_devices()]))
    strategy = None
    if CONFIG['strategy'] == "onedevice":
        CONFIG['strategy'] = tf.distribute.OneDeviceStrategy(device="/gpu:0")
    if CONFIG['strategy'] == "mirrored":
        CONFIG['strategy'] = tf.distribute.MirroredStrategy()
        print('Number of devices: {}'.format(CONFIG['strategy'].num_replicas_in_sync))
    if CONFIG['strategy'] == "tpu":
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=CONFIG['tpu_name'])
        print(f"Connecting to tpu {CONFIG['tpu_name']}...")
        tf.config.experimental_connect_to_cluster(resolver)
        print(f"Initializing tpu {CONFIG['tpu_name']}...")
        tf.tpu.experimental.initialize_tpu_system(resolver)
        print("All TPU devices: ", tf.config.list_logical_devices('TPU'))
        CONFIG['strategy'] = tf.distribute.TPUStrategy(resolver)
    if CONFIG['strategy'] == "colab_tpu":
        try:
            tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection
            print('Running on TPU ', tpu.cluster_spec().as_dict()['worker'])
        except ValueError:
            raise BaseException(
                'ERROR: Not connected to a TPU runtime; please see the previous cell in this notebook for instructions!')
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        CONFIG['strategy'] = tf.distribute.TPUStrategy(tpu)
    TRAINER = Trainer(CONFIG)
    HISTORY = TRAINER.train()
