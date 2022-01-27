import tensorflow as tf

# a colab tpu csak GCS-ról tudja olvasni a tfrecordokat és oda tudja menteni a checkpointokat

tfrec_train_pattern = 'gs://arcanum-ml/cv/correct_orientation/tpu/tfrecords_new/train*'
tfrec_val_pattern = 'gs://arcanum-ml/cv/correct_orientation/tpu/tfrecords_new/val*'
model_dir = 'gs://arcanum-ml/cv/correct_orientation/tpu/train_2048_256_60_80_0-7'
log_dir = 'gs://arcanum-ml/cv/correct_orientation/tpu/train_2048_256_60_80_0-7'

CONFIG = {
    'project_name': 'Correct orientation of images',
    'experiment_name': 'Resnet-50-backbone, image_size=256',
    'train_dataset_config': {
        'tf_records': tf.io.gfile.glob(tfrec_train_pattern),
        'batch_size': 2048
    },
    'val_dataset_config': {
        'tf_records': tf.io.gfile.glob(tfrec_val_pattern),
        'batch_size': 2048
    },
    'image_size': 256,
    'strategy': 'tpu',
    'tpu_name': 'rotation',
    'initial_learning_rate': 5e-3,
    'end_learning_rate': 5e-5,
    'checkpoint_dir': model_dir,
    'checkpoint_file_prefix': "ckpt_",
    'log_dir': log_dir,
    'epochs': 80,
    'power': 0.7,
    'num_of_train_examples': 136910,
    'num_of_val_examples': 15101
}

steps_per_epoch = CONFIG['num_of_train_examples'] // CONFIG['train_dataset_config']['batch_size']
CONFIG['decay_steps'] = steps_per_epoch * 60
# validation_steps: 7429 // CONFIG['val_dataset_config']['batch_size']