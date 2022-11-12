import os
import shutil
import logging
import warnings
import csv
import json
from json import JSONEncoder

import numpy as np
from PIL import Image

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# In details,
# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed

from keras.utils.vis_utils import plot_model    # a modell vizualizációjához
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from tensorflow.keras.optimizers import Adam

import matplotlib.pyplot as plt
# %matplotlib inline # for jupyter-notebook

from timeit import default_timer as timer   # for verbose output

from helpers import *


INFO_LEVEL = 1


def extract_features(A):
    return {
        'orig': A,
        'proj': {
            'ver': A.sum(axis=0),
            'hor': A.sum(axis=1),
        },
        'slbp': slbp(A),
        'ryser': ryser_algorithm(A),
    }


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


def main(features_path, image_path, image_size, desired_samples, without_duplicates=False):
    image = Image.open(image_path)
    image.load()
    raw_image = np.asarray(image, dtype='int32')

    if without_duplicates:
        helper = []
        with open(features_path, 'a') as f:
            cnt = 0
            while cnt < desired_samples:
                sub_image = get_random_sub_image(raw_image, image_size)
                sub_image_json = json.dumps(sub_image, separators=(',', ':'), cls=NumpyArrayEncoder)
                if sub_image_json in helper:
                    print('this is a duplicate!')
                    continue
                helper.append(sub_image_json)
                json_obj = json.dumps(extract_features(sub_image), separators=(',', ':'), cls=NumpyArrayEncoder)
                f.write(f'{json_obj}\n')
                cnt += 1
    else:
        with open(features_path, 'a') as f:
            for sub_image in get_random_sub_images(raw_image, image_size, desired_samples):
                json_obj = json.dumps(extract_features(sub_image), separators=(',', ':'), cls=NumpyArrayEncoder)
                f.write(f'{json_obj}\n')


if __name__ == '__main__':
    # tf.logging.set_verbosity(tf.logging.ERROR)
    logging.getLogger('matplotlib').setLevel(level=logging.CRITICAL)
    logging.getLogger('tensorflow').setLevel(level=logging.CRITICAL)
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    images = {
        'phan': 'images/phantom_class_02.png',
        'brod': 'images/Brodatz_resized_04.png',
        'real': 'images/real_9.png',
    }
    size_params = [
        (16, 16000),
        (32, 4000),
        (64, 1000),
    ]
    features_folder_prefix = 'features'
    os.makedirs(features_folder_prefix, exist_ok=True)

    for size_param in size_params:
        (size, desired_samples) = size_param
        print(f'{size:>02}x{size:>02}\t{desired_samples} pcs')
        start_per_size_param = timer()
        for image_name in images:
            print(f'\t{image_name}')
            start_per_image = timer()

            # Fixing random state for reproducibility
            # tf.keras.utils.set_random_seed(1)
            # if True:
            features_path = f'{features_folder_prefix}/{image_name}-{size}x{size}.txt'

            # try:
            if True:
                start = timer()
                main(
                    features_path=features_path,
                    image_path=images[image_name],
                    image_size=size,
                    desired_samples=desired_samples,
                )
                end = timer()
            # except Exception:
            #     print('some error')
            end_per_image = timer()
            print(f'\t\t\t[PER IMAGE]\n\t\t\tTime: {end_per_image-start_per_image:.2f} s\n\n')
        end_per_size_param = timer()
        print(f'\t\t\t[PER SIZE-PARAM]\n\t\t\tTime: {end_per_size_param-start_per_size_param:.2f} s\n\n')
