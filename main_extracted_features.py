import os
import shutil
import logging
import warnings
import csv
import json

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

import matplotlib.pyplot as plt
# %matplotlib inline # for jupyter-notebook

from timeit import default_timer as timer   # for verbose output

from helpers import *


INFO_LEVEL = 0


class Layer:
    def __init__(self, name, layer_ctor, nodes, activation, input_nodes=0, dropout_rate=None):
        self.layer_ctor = layer_ctor
        self.nodes = nodes
        self.activation = activation
        self.name = name
        self.input_nodes = input_nodes
        self.dropout_rate = dropout_rate

    def create(self):
        if self.input_nodes:
            return self.layer_ctor(
                self.nodes,
                input_shape=(self.input_nodes,),
                activation=self.activation,
                name=self.name
            )
        else:
            return self.layer_ctor(
                self.nodes,
                activation=self.activation,
                name=self.name
            )

    def need_dropout(self):
        return self.dropout_rate is not None

    def create_dropout(self, postfix):
        return Dropout(rate=self.dropout_rate, name=f'{self.name}-dropout-{postfix}')


def create_model(layers, model_path, optimizer):
    model = Sequential()

    for layer in layers:
        if layer.need_dropout():
            model.add(layer.create_dropout(1))
        model.add(layer.create())
        if layer.need_dropout():
            model.add(layer.create_dropout(2))

    if INFO_LEVEL >= 1:
        model.summary()
    plot_model(
        model,
        show_shapes=True,
        show_layer_names=True,
        rankdir='TB',
        # rankdir='LR',
        show_layer_activations=True,
        to_file=model_path
    )

    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['mae'])
    # model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])

    return model, 'loss: binary_crossentropy'


def print_info(predicts, Y_test, result_folder, desired_samples, image_size, input_sblp, input_ryser, additional=''):
    max_picture = 10
    for i in range(len(predicts)):
        if max_picture == 0:
            break
        max_picture -= 1

        original = Y_test[i].reshape(image_size, image_size)
        predicted_nn = predicts[i].round().astype('int').reshape(image_size, image_size)
        predicted_ryser = ryser_algorithm(Y_test[i].reshape(image_size, image_size))

        fig, axs = plt.subplots(1, 3)
        axs[0].set_title('original\n\nrme:\n pe:\n')
        axs[0].imshow(original, interpolation='none', cmap=plt.cm.binary)

        axs[1].set_title(f'NN\n\n{relative_mean_error(original, predicted_nn):.2f}\n{pixel_error(original, predicted_nn):.2f}\n')
        axs[1].imshow(predicted_nn, interpolation='none', cmap=plt.cm.binary)

        axs[2].set_title(f'Ryser\n\n{relative_mean_error(original, predicted_ryser):.2f}\n{pixel_error(original, predicted_ryser):.2f}\n')
        axs[2].imshow(predicted_ryser, interpolation='none', cmap=plt.cm.binary)

        plt.savefig(f'{result_folder}/{i}.png')
        plt.close(fig)

    results = {
        'ryser': {
            'rme': [],
            'pe': [],
        },
        'nn': {
            'rme': [],
            'pe': [],
        },
    }

    for i in range(len(predicts)):
        predicted_nn = predicts[i].round().astype('int').reshape(image_size, image_size)
        predicted_ryser = ryser_algorithm(Y_test[i].reshape(image_size, image_size))
        original = Y_test[i].reshape(image_size, image_size)

        if original.sum() == 0:
            continue
        results['ryser']['rme'].append(relative_mean_error(original, predicted_ryser))
        results['ryser']['pe'].append(pixel_error(original, predicted_ryser))
        results['nn']['rme'].append(relative_mean_error(original, predicted_nn))
        results['nn']['pe'].append(pixel_error(original, predicted_nn))

    with open(f'{result_folder}/info.txt', 'w') as f:
        f.write(f'Image size: {image_size}x{image_size}\n')
        f.write(f'Samples: {desired_samples}\n')
        f.write(f'Added inputs:\n')
        f.write(f'    * SBLP: {input_sblp}\n')
        f.write(f'    * Ryser: {input_ryser}\n')
        f.write('\n')
        f.write(f'              Ryser          NN\n')
        f.write(f'rme\n')
        f.write(f'    * min      {np.min(results["ryser"]["rme"]):>7.2f} %    {np.min(results["nn"]["rme"]):>7.2f} %\n')
        f.write(f'    * max      {np.max(results["ryser"]["rme"]):>7.2f} %    {np.max(results["nn"]["rme"]):>7.2f} %\n')
        f.write(f'    * avg      {np.mean(results["ryser"]["rme"]):>7.2f} %    {np.mean(results["nn"]["rme"]):>7.2f} %\n')
        f.write(f'    * std      {np.std(results["ryser"]["rme"]):>7.2f} %    {np.std(results["nn"]["rme"]):>7.2f} %\n')
        f.write(f'    * median   {np.median(results["ryser"]["rme"]):>7.2f} %    {np.median(results["nn"]["rme"]):>7.2f} %\n')
        f.write(f' pe\n')
        f.write(f'    * min      {np.min(results["ryser"]["pe"]):>7.2f} %    {np.min(results["nn"]["pe"]):>7.2f} %\n')
        f.write(f'    * max      {np.max(results["ryser"]["pe"]):>7.2f} %    {np.max(results["nn"]["pe"]):>7.2f} %\n')
        f.write(f'    * avg      {np.mean(results["ryser"]["pe"]):>7.2f} %    {np.mean(results["nn"]["pe"]):>7.2f} %\n')
        f.write(f'    * std      {np.std(results["ryser"]["pe"]):>7.2f} %    {np.std(results["nn"]["pe"]):>7.2f} %\n')
        f.write(f'    * median   {np.median(results["ryser"]["pe"]):>7.2f} %    {np.median(results["nn"]["pe"]):>7.2f} %\n')
        f.write(f'\n')
        f.write(f'additional: {additional}\n')
        f.write(f'\n')
    return {
        'ryser': {
            'min': round(float(np.min(results["ryser"]["rme"])), 2),
            'max': round(float(np.max(results["ryser"]["rme"])), 2),
            'avg': round(float(np.mean(results["ryser"]["rme"])), 2),
            'std': round(float(np.std(results["ryser"]["rme"])), 2),
        },
        'nn': {
            'min': round(float(np.min(results["nn"]["rme"])), 2),
            'max': round(float(np.max(results["nn"]["rme"])), 2),
            'avg': round(float(np.mean(results["nn"]["rme"])), 2),
            'std': round(float(np.std(results["nn"]["rme"])), 2),
        }
    }


def main(result_folder, images, size, samples_per_image, features, features_folder, optimizer):
    os.makedirs(result_folder)

    raw_extracted_features = []
    for image in images:
        with open(f'{features_folder}/{image}-{size}x{size}.txt', 'r') as f:
            for x in range(samples_per_image):
                raw_extracted_features.append(json.loads(next(f).strip()))
    # print(len(origins))
    # print(len(extracted_features))
    # print(extracted_features[1])

    raw_extracted_features = np.array(raw_extracted_features)
    np.random.shuffle(raw_extracted_features)
    raw_train, raw_test = np.split(raw_extracted_features, [int(raw_extracted_features.shape[0] * 0.9)])
    if INFO_LEVEL >= 1:
        print(f'Train set: {raw_train.shape}')
        print(f'Test set: {raw_test.shape}')

    train_extracted_features = []
    train_origins = []
    for raw in raw_test:
        extracted_features_helper = []
        if 'proj' in features:
            proj_a = raw['proj']['ver']
            proj_a_max = np.amax(proj_a)
            if proj_a_max > 0:
                proj_a /= proj_a_max

            proj_b = raw['proj']['hor']
            proj_b_max = np.amax(proj_b)
            if proj_b_max > 0:
                proj_b /= proj_b_max

            extracted_features_helper.append(proj_a)
            extracted_features_helper.append(proj_b)
        if 'slbp' in features:
            extracted_features_helper.append(np.array(raw['slbp']))
        if 'ryser' in features:
            extracted_features_helper.append(np.array(raw['ryser']).reshape(size * size))
        train_extracted_features.append(np.concatenate(extracted_features_helper))
        train_origins.append(np.array(raw['orig']).reshape(size*size))
    train_extracted_features = np.array(train_extracted_features)
    train_origins = np.array(train_origins)

    test_extracted_features = []
    test_origins = []
    for raw in raw_test:
        extracted_features_helper = []
        if 'proj' in features:
            proj_a = raw['proj']['ver']
            proj_a_max = np.amax(proj_a)
            if proj_a_max > 0:
                proj_a /= proj_a_max

            proj_b = raw['proj']['hor']
            proj_b_max = np.amax(proj_b)
            if proj_b_max > 0:
                proj_b /= proj_b_max

            extracted_features_helper.append(proj_a)
            extracted_features_helper.append(proj_b)
        if 'slbp' in features:
            extracted_features_helper.append(np.array(raw['slbp']))
        if 'ryser' in features:
            extracted_features_helper.append(np.array(raw['ryser']).reshape(size * size))
        test_extracted_features.append(np.concatenate(extracted_features_helper))
        test_origins.append(np.array(raw['orig']).reshape(size*size))
    test_extracted_features = np.array(test_extracted_features)
    test_origins = np.array(test_origins)

    feature_nodes = len(train_extracted_features[0])
    result_nodes = len(train_origins[0])

    lh = int((result_nodes-feature_nodes)/2)
    l1 = int(feature_nodes+lh)
    l2 = int(l1+lh)
    if INFO_LEVEL >= 1:
        print(f'{feature_nodes} > {l1} > {l2} > {result_nodes}\n{lh}')
    model, additional_info = create_model(
        layers=[
            # 64x64
            Layer('input', Dense, l1, 'relu', input_nodes=feature_nodes),
            Layer('hidden-1', Dense, 3*l2, 'relu', dropout_rate=None),
            Layer('hidden-2', Dense, 3*l2, 'relu', dropout_rate=None),
            Layer('hidden-3', Dense, 3*l2, 'relu', dropout_rate=None),

            # # 128x128
            # Layer('input', Dense, 8448, 'relu', input_nodes=feature_nodes),
            # Layer('hidden-1', Dense, 16384, 'relu'),

            # Layer('input', Dense, 672, 'relu', input_nodes=feature_nodes),
            # Layer('hidden-1', Dense, 1024, 'relu'),

            # Layer('input', Dense, 512, 'relu', input_nodes=feature_nodes),
            # Layer('hidden-1', Dense, 2048, 'relu'),
            Layer('output', Dense, result_nodes, 'sigmoid'),
        ],
        model_path=f'{result_folder}/model.png',
        optimizer=optimizer,
    )

    # X_train = []
    # Y_train = []
    # train_cnt = 0
    # train_sub_images_len = len(train_sub_images)
    # if INFO_LEVEL >= 1:
    #     print(f'train set calculation [{train_sub_images_len}]')
    # last_percentage = 1
    # start = timer()
    # for X_train_one in train_sub_images:
    #     train_cnt += 1
    #     X, Y = extract_features(X_train_one)
    #     percentage = int(train_cnt/train_sub_images_len*100)
    #     if percentage == last_percentage and percentage != 0:
    #         last_percentage += 1
    #         end = timer()
    #         if INFO_LEVEL >= 1:
    #             print(f'\t{percentage}% \t {train_cnt} from {train_sub_images_len} in {end-start} s')
    #         start = timer()
    #     X_train.append(X)
    #     Y_train.append(Y)
    #     # print(train_cnt)
    # X_train = np.array(X_train)
    # Y_train = np.array(Y_train)
    #
    # X_test = []
    # Y_test = []
    # test_cnt = 0
    # test_sub_images_len = len(test_sub_images)
    # if INFO_LEVEL >= 1:
    #     print(f'test set calculation [{test_sub_images_len}]')
    # last_percentage = 1
    # start = timer()
    # for X_test_one in test_sub_images:
    #     test_cnt += 1
    #     X, Y = extract_features(X_test_one)
    #     percentage = int(test_cnt/test_sub_images_len*100)
    #     if percentage == last_percentage and percentage != 0:
    #         last_percentage += 1
    #         end = timer()
    #         if INFO_LEVEL >= 1:
    #             print(f'\t{percentage}% \t {test_cnt} from {test_sub_images_len} in {end-start} s')
    #         start = timer()
    #     X_test.append(X)
    #     Y_test.append(Y)
    # X_test = np.array(X_test)
    # Y_test = np.array(Y_test)
    #
    # # for r in range(int((raw_rows - image_size - 1)/image_size)):
    # #     for c in range(int((raw_cols - image_size - 1)/image_size)):
    # #         all_sub_images.append(raw_image[r:r+image_size, c:c+image_size])
    #
    # # all_sub_images = np.array(all_sub_images)
    # # print(f'Base set: {all_sub_images.shape}')
    #
    #
    # Note: lower batch_size help to the training depending on the training-set size
    # batch_size = int(desired_samples/64)
    batch_size = 128
    epochs = 50
    shuffle = True
    verbose = INFO_LEVEL >= 1

    history = model.fit(
        train_extracted_features,
        train_origins,
        batch_size=batch_size,
        epochs=epochs,
        shuffle=shuffle,
        verbose=verbose,
        validation_split=0.1
    )

    # list all data in history
    if INFO_LEVEL >= 1:
        print(history.history.keys())
    # summarize history for mae
    fig, ax = plt.subplots(nrows=1, ncols=1)  # create figure & 1 axis
    ax.plot(history.history['mae'])
    ax.plot(history.history['val_mae'])
    ax.set_title('model mae')
    ax.set_ylabel('mae')
    ax.set_xlabel('epoch')
    ax.legend(['train', 'test'], loc='upper left')
    fig.savefig(f'{result_folder}/mae.png')  # save the figure to file
    plt.close(fig)  # close the figure window

    # summarize history for loss
    fig, ax = plt.subplots(nrows=1, ncols=1)  # create figure & 1 axis
    ax.plot(history.history['loss'])
    ax.plot(history.history['val_loss'])
    ax.set_title('model loss')
    ax.set_ylabel('loss')
    ax.set_xlabel('epoch')
    ax.legend(['train', 'test'], loc='upper left')
    fig.savefig(f'{result_folder}/loss.png')  # save the figure to file
    plt.close(fig)  # close the figure window

    # # summarize history for accuracy
    # fig, ax = plt.subplots(nrows=1, ncols=1)  # create figure & 1 axis
    # ax.plot(history.history['accuracy'])
    # ax.plot(history.history['val_accuracy'])
    # ax.set_title('model accuracy')
    # ax.set_ylabel('accuracy')
    # ax.set_xlabel('epoch')
    # ax.legend(['train', 'test'], loc='upper left')
    # fig.savefig(f'{result_folder}/accuracy.png')  # save the figure to file
    # plt.close(fig)  # close the figure window

    score, mae = model.evaluate(test_extracted_features, test_origins)
    if INFO_LEVEL >= 1:
        print('Test score:', score)
        print('Test mae:', mae)


    predicts = model.predict(test_extracted_features)
    if INFO_LEVEL >= 1:
        print(test_extracted_features.shape)
        print(predicts.shape)

    return print_info(
        predicts=predicts,
        Y_test=test_origins,
        image_size=size,
        desired_samples=len(images)*samples_per_image,
        result_folder=result_folder,
        input_sblp='slbp' in features,
        input_ryser='ryser' in features,
        additional=additional_info,
    )


if __name__ == '__main__':
    # tf.logging.set_verbosity(tf.logging.ERROR)
    logging.getLogger('matplotlib').setLevel(level=logging.CRITICAL)
    logging.getLogger('tensorflow').setLevel(level=logging.CRITICAL)
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    params = [
        {
            'images': ['phan', 'brod', 'real'],
            'size': 16,
            'samples_per_image': 4000,
            'features': ['proj'],
            'features_folder': 'features',
            'optimizer': tf.keras.optimizers.Adam(learning_rate=0.01),
            'result-name-prefix': 'Adam01',
        },
        {
            'images': ['phan', 'brod', 'real'],
            'size': 16,
            'samples_per_image': 4000,
            'features': ['proj', 'slbp'],
            'features_folder': 'features',
            'optimizer': tf.keras.optimizers.Adam(learning_rate=0.01),
            'result-name-prefix': 'Adam_01',
        },
    ]

    params = []
    size_and_samples = [
        # (16, 1000),
        # (16, 2000),
        (32, 4000),
    ]
    features = [
        ['proj'],
        # ['slbp'],
        # ['ryser'],
        ['proj', 'slbp'],
        ['ryser', 'slbp']
    ]
    optimizers = [
        ('Adam_0.1000', tf.keras.optimizers.Adam(learning_rate=0.1)),
        ('Adam_0.0100', tf.keras.optimizers.Adam(learning_rate=0.01)),
        ('Adam_0.0010', tf.keras.optimizers.Adam(learning_rate=0.001)),
        ('Adam_0.0001', tf.keras.optimizers.Adam(learning_rate=0.0001)),
        # ('Adam_0.0500', tf.keras.optimizers.Adam(learning_rate=0.05)),
        # ('Adam_0.0250', tf.keras.optimizers.Adam(learning_rate=0.025)),
        # ('Adam_0.0125', tf.keras.optimizers.Adam(learning_rate=0.0125)),
        # ('Adam_0.0062', tf.keras.optimizers.Adam(learning_rate=0.0062)),
        # ('Adam_0.0031', tf.keras.optimizers.Adam(learning_rate=0.0031)),
        # ('Adam_0.0015', tf.keras.optimizers.Adam(learning_rate=0.0015)),
        # ('Adam_0.0007', tf.keras.optimizers.Adam(learning_rate=0.0007)),
        # ('Adam_0.0003', tf.keras.optimizers.Adam(learning_rate=0.0003)),
        # ('Adam_0.0001', tf.keras.optimizers.Adam(learning_rate=0.0001)),
        # ('SGD_0.1000', tf.keras.optimizers.SGD(learning_rate=0.1)),
        # ('SGD_0.0100', tf.keras.optimizers.SGD(learning_rate=0.01)),
        # ('SGD_0.0010', tf.keras.optimizers.SGD(learning_rate=0.001)),
        # ('SGD_0.0001', tf.keras.optimizers.SGD(learning_rate=0.0001)),
        # ('SGD_0.0500', tf.keras.optimizers.SGD(learning_rate=0.05)),
        # ('SGD_0.0250', tf.keras.optimizers.SGD(learning_rate=0.025)),
        # ('SGD_0.0125', tf.keras.optimizers.SGD(learning_rate=0.0125)),
        # ('SGD_0.0062', tf.keras.optimizers.SGD(learning_rate=0.0062)),
        # ('SGD_0.0031', tf.keras.optimizers.SGD(learning_rate=0.0031)),
        # ('SGD_0.0015', tf.keras.optimizers.SGD(learning_rate=0.0015)),
        # ('SGD_0.0007', tf.keras.optimizers.SGD(learning_rate=0.0007)),
        # ('SGD_0.0003', tf.keras.optimizers.SGD(learning_rate=0.0003)),
        # ('SGD_0.0001', tf.keras.optimizers.SGD(learning_rate=0.0001)),
    ]
    for (size, sample) in size_and_samples:
        for feature in features:
            for (optimizer_name, optimizer) in optimizers:
                params.append(
                    {
                        'images': ['phan', 'brod', 'real'],
                        'size': size,
                        'samples_per_image': sample,
                        'features': feature,
                        'features_folder': 'features2',
                        'optimizer': optimizer,
                        'result-name-prefix': optimizer_name,
                    }
                )
    result_folder_prefix = 'extracted-features-results/6'
    os.makedirs(result_folder_prefix, exist_ok=True)
    csv_file = open(f'{result_folder_prefix}/summary.csv', 'w')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "seed",
        "images",
        "size",
        "samples_per_image",
        "features",
        "RS_min (%)",
        "RS_max (%)",
        "RS_avg (%)",
        "RS_std (%)",
        "NN_min (%)",
        "NN_max (%)",
        "NN_avg (%)",
        "NN_std (%)",
        "time (s)",
    ])

    run_cnt = 0
    for param in params:
        # samples = int(250 * (size * size) / (8 * 8))
        size = param['size']
        samples_per_image = param['samples_per_image']
        images = param['images']
        features = param['features']
        features_folder = param['features_folder']
        optimizer = param['optimizer']
        result_name_prefix = param['result-name-prefix']

        print(f'{size:>02}x{size:>02}\t{samples_per_image}pcs/image-{"-".join(images)}')
        start_per_param = timer()

        # Fixing random state for reproducibility
        tf.keras.utils.set_random_seed(46842)

        result_folder = f'{result_folder_prefix}/{size}x{size}-{samples_per_image}pcs/'
        result_folder += f'{result_name_prefix}-{"_".join(images)}-{"_".join(features)}'
        if True:
        # try:
            start = timer()
            res = main(
                result_folder=result_folder,
                images=images,
                size=size,
                samples_per_image=samples_per_image,
                features=features,
                features_folder=features_folder,
                optimizer=optimizer,
            )
            end = timer()
            print(
                f'\t\t\tRS: '
                f'min: {res["ryser"]["min"]:>7.2f} % '
                f'max: {res["ryser"]["max"]:>7.2f} % '
                f'avg: {res["ryser"]["avg"]:>7.2f} % '
                f'std: {res["ryser"]["std"]:>7.2f} % '
            )
            print(
                f'\t\t\tNN: '
                f'min: {res["nn"]["min"]:>7.2f} % '
                f'max: {res["nn"]["max"]:>7.2f} % '
                f'avg: {res["nn"]["avg"]:>7.2f} % '
                f'std: {res["nn"]["std"]:>7.2f} % '
            )
            print(f'\t\t\tTime: {end-start:.2f} s\n\n')
            csv_writer.writerow([
                1,
                "-".join(images),
                size,
                samples_per_image,
                "-".join(features),
                res["ryser"]["min"]/100,
                res["ryser"]["max"]/100,
                res["ryser"]["avg"]/100,
                res["ryser"]["std"]/100,
                res["nn"]["min"]/100,
                res["nn"]["max"]/100,
                res["nn"]["avg"]/100,
                res["nn"]["std"]/100,
                f'{end - start:.2f}',
            ])
        # except Exception:
        #     print('some error')

        end_per_param = timer()
        print(f'\t\t\t[PER PARAM]\n\t\t\tTime: {end_per_param-start_per_param:.2f} s\n\n')
        run_cnt += 1
        print(f'{run_cnt} / {len(params)}')

    csv_file.close()
