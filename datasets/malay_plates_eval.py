import datasets.fsns as fsns

DEFAULT_DATASET_DIR = '/home/qisens/tensorflow/models/research/attention_ocr/malay/datasets/data/malay_plates_eval'

DEFAULT_CONFIG = {
    'name':
        'number_plates', # you can change the name if you want.
    'splits': {
        'train': {
            'size': 900, # change according to your own train-test split
            'pattern': 'train.tfrecord'
        },
        'test': {
            'size': 98, # change according to your own train-test split
            'pattern': 'test.tfrecord'
        }
    },
    'charset_filename':
        'charset_size=134.txt',
    'image_shape': (150,600,3),#(max_width, max_height, 3),
    'num_of_views':
        4,
    'max_sequence_length':
        37, # TO BE CONFIGURED
    'null_code':
        133,
    'items_to_descriptions': {
        'image':
            'A 3 channel color image.',
        'label':
            'Characters codes.',
        'text':
            'A unicode string.',
        'length':
            'A length of the encoded text.',
        'num_of_views':
            'A number of different views stored within the image.'
    }
}


def get_split(split_name, dataset_dir=None, config=None):
  if not dataset_dir:
    dataset_dir = DEFAULT_DATASET_DIR
  if not config:
    config = DEFAULT_CONFIG

  return fsns.get_split(split_name, dataset_dir, config)
