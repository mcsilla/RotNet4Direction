import tensorflow as tf

# a colab tpu csak GCS-ról tudja olvasni a tfrecordokat és oda tudja menteni a checkpointokat

tfrec_train_pattern = 'gs://arcanum-ml/cv/correct_orientation/tpu/tfrecords/train*'
tfrec_val_pattern = 'gs://arcanum-ml/cv/correct_orientation/tpu/tfrecords/val*'
model_dir = 'gs://arcanum-ml/cv/correct_orientation/tpu/model'
log_dir = 'gs://arcanum-ml/cv/correct_orientation/tpu/logs'

CONFIG = {
    'project_name': 'Correct orientation of images',
    'experiment_name': 'Resnet-50-backbone, image_size=256',
    'train_dataset_config': {
        'tf_records': tf.io.gfile.glob(tfrec_train_pattern),
        'batch_size': 1024
    },
    'val_dataset_config': {
        'tf_records': tf.io.gfile.glob(tfrec_val_pattern),
        'batch_size': 1024
    },
    'image_size': 256,
    'strategy': 'tpu',
    'tpu_name': 'rotation',
    'initial_learning_rate': 5e-3,
    'end_learning_rate': 5e-5,
    'checkpoint_dir': model_dir,
    'checkpoint_file_prefix': "ckpt_",
    'log_dir': log_dir,
    'epochs': 50,
    'power': 0.9,
    'num_of_train_examples': 136937,
    'num_of_val_examples': 15111
}

steps_per_epoch = CONFIG['num_of_train_examples'] // CONFIG['train_dataset_config']['batch_size']
CONFIG['decay_steps'] = steps_per_epoch * 30
# validation_steps: 7429 // CONFIG['val_dataset_config']['batch_size']