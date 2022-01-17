import tensorflow as tf

tfrec_train_pattern = '/content/train*'
tfrec_val_pattern = '/content/val'
model_dir = '/content/model'
log_dir = '/content/log'

CONFIG = {
    'project_name': 'Correct orientation of images',
    'experiment_name': 'Resnet-50-backbone, image_size=256',
    'train_dataset_config': {
        'tf_records': tf.io.gfile.glob(tfrec_train_pattern),
        'batch_size': 512
    },
    'val_dataset_config': {
        'tf_records': tf.io.gfile.glob(tfrec_val_pattern),
        'batch_size': 512
    },
    'image_size': 256,
    'strategy': 'colab_tpu',
    'initial_learning_rate': 5e-3,
    'end_learning_rate': 5e-5,
    'checkpoint_dir': model_dir,
    'checkpoint_file_prefix': "ckpt_",
    'log_dir': log_dir,
    'epochs': 10,
    'power': 0.9,
    'num_of_train_examples': 2974,
    'num_of_val_examples': 314
}

steps_per_epoch = CONFIG['num_of_train_examples'] // CONFIG['train_dataset_config']['batch_size']
CONFIG['decay_steps'] = steps_per_epoch * 10
# validation_steps: 7429 // CONFIG['val_dataset_config']['batch_size']