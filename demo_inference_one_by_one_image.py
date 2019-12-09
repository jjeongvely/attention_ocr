"""A script to run inference on a set of image files.

NOTE #1: The Attention OCR model was trained only using FSNS train dataset and
it will work only for images which look more or less similar to french street
names. In order to apply it to images from a different distribution you need
to retrain (or at least fine-tune) it using images from that distribution.

NOTE #2: This script exists for demo purposes only. It is highly recommended
to use tools and mechanisms provided by the TensorFlow Serving system to run
inference on TensorFlow models in production:
https://www.tensorflow.org/serving/serving_basic

Usage:
python demo_inference.py --batch_size=32 \
  --checkpoint=model.ckpt-399731\
  --image_path_pattern=./datasets/data/fsns/temp/fsns_train_%02d.png
"""

import numpy as np
import PIL.Image

import tensorflow as tf
from tensorflow.python.platform import flags
from tensorflow.python.training import monitored_session
import datasets
import data_provider
import time
import os

#You shoud change this!!
import common_flags_600by150 as common_flags


FLAGS = flags.FLAGS
common_flags.define()

# e.g. ./datasets/data/fsns/temp/fsns_train_%02d.png
flags.DEFINE_string('image_path_pattern', '',
                    'A file pattern with a placeholder for the image index.')

flags.DEFINE_string('dir_path', '',
                    'A file pattern with a placeholder for the image index.')


def get_dataset_image_size(dataset_name):
  # Ideally this info should be exposed through the dataset interface itself.
  # But currently it is not available by other means.
  ds_module = getattr(datasets, dataset_name)
  height, width, _ = ds_module.DEFAULT_CONFIG['image_shape']
  return width, height


def load_images(file_path, batch_size, dataset_name):
  width, height = get_dataset_image_size(dataset_name)
  images_actual_data = np.ndarray(shape=(batch_size, height, width, 3),
                                  dtype='uint8')

  # print("Reading %s" % file_path)
  pil_image = PIL.Image.open(file_path)
  images_actual_data[0, ...] = np.asarray(pil_image)
  return images_actual_data


def create_model(batch_size, dataset_name):
  width, height = get_dataset_image_size(dataset_name)
  dataset = common_flags.create_dataset(split_name=FLAGS.split_name)
  model = common_flags.create_model(
    num_char_classes=dataset.num_char_classes,
    seq_length=dataset.max_sequence_length,
    num_views=dataset.num_of_views,
    null_code=dataset.null_code,
    charset=dataset.charset)
  raw_images = tf.placeholder(tf.uint8, shape=[batch_size, height, width, 3])
  images = tf.map_fn(data_provider.preprocess_image, raw_images,
                     dtype=tf.float32)
  endpoints = model.create_base(images, labels_one_hot=None)
  return raw_images, endpoints

def main(_):

  #You need to give arguments below
  #FLAGS.dataset_name, FLAGS.dir_path(inference image directory), FLAGS.checkpoint
  #DO NOT FORGET TO CHANGE PATH TO "Common_flags"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  test = FLAGS.dir_path
  print("Predicted strings:")
  images_placeholder, endpoints = create_model(1, FLAGS.dataset_name)
  session_creator = monitored_session.ChiefSessionCreator(
    checkpoint_filename_with_path=FLAGS.checkpoint)

  with monitored_session.MonitoredSession(
          session_creator=session_creator) as sess:

    #active a dummy image
    images_data = load_images("/home/qisens/tensorflow/models/research/attention_ocr/malay/dummy/number_plates_00.png", 1, FLAGS.dataset_name)
    prediction = sess.run(endpoints.predicted_text, feed_dict={images_placeholder: images_data})

    correct = 0
    incorrect = 0

    for path, dirs, files in os.walk(FLAGS.dir_path):
      for file in files:
        file_path = os.path.join(path, file)
        gt = file.split('.')[0]

        recog_startime = time.time()
        images_data = load_images(file_path, 1, FLAGS.dataset_name)
        prediction = sess.run(endpoints.predicted_text, feed_dict={images_placeholder: images_data})
        recog_endtime = time.time()
        print('\033[0m'+"[time] {:<10.3f}   ".format(max(recog_endtime - recog_startime, 0)), end="", flush=True)

        prediction = prediction.tolist()
        predicted_gt = ''
        for chr in prediction[0].decode('utf-8'):
          if (chr == '░'):
            break
          predicted_gt += chr

        #print('[image] {:<13}   [prediction] {:<12}'.format(gt, predicted_gt))
        if (gt == predicted_gt):
          print('\033[0m'+'[image] {:<13}   [prediction] {:<12}'.format(gt, predicted_gt))
          correct += 1
        else:
          incorrect += 1
          print('\033[91m'+'[image] {:<13}   [prediction] {:<12}'.format(gt, predicted_gt))

    accuracy = (correct / (correct + incorrect)) * 100
    print('\033[0m'+"Accuracy : {0:.2f}%".format(accuracy))

if __name__ == '__main__':
  tf.app.run()
