import os
import shutil
import logging
import warnings
import csv

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
from keras.layers import Dense, Dropout, Activation, BatchNormalization

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

    # learning_rate:
    # 0.1 - 0.0001
    # amíg a test a train alatt v an addig lehet nővelni a complexitást
    # dropout -> továbbtanítás
    # linear activation function???
    # early stopping 500 epoch-ról megnézni, ahol nem történik érdemi változás
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['mae'])

    return model

# def create_model(input_nodes, hidden_nodes, activations):
#     model = Sequential()
#
    # model.add(Dense(hidden_nodes[0], input_shape=(input_nodes,), activation=activations[0], name="hidden-1"))
#     model.add(Dense(hidden_nodes[1], activation=activations[1], name="hidden-2"))
#     model.add(Dense(hidden_nodes[2], activation=activations[2], name="output"))
#
#     return model


def print_info(predicts, Y_test, result_folder, desired_samples, image_size, input_sblp, input_ryser):
    max_picture = 10
    image_invert_helper = np.full((image_size, image_size), -1, dtype=int)
    for i in range(len(predicts)):
        if max_picture == 0:
            break
        max_picture -= 1

        original = np.subtract(image_invert_helper, Y_test[i].reshape(image_size, image_size))
        predicted_nn = np.subtract(image_invert_helper, predicts[i].round().astype('int').reshape(image_size, image_size))
        predicted_ryser = np.subtract(image_invert_helper, ryser_algorithm(Y_test[i].reshape(image_size, image_size)))

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


train_sub_images = None
test_sub_images = None


def main(result_folder, image_path, image_size, desired_samples, input_sblp, input_ryser, dropout_rate, optimizer, activation_function):
    os.makedirs(result_folder)
    global train_sub_images
    global test_sub_images

    pareto_limit = 0.8
    if train_sub_images is None and test_sub_images is None:
        # image = Image.open('images/phantom_class_02.png')
        # image = Image.open('images/real_9.png')
        image = Image.open(image_path)
        image.load()
        raw_image = np.asarray(image, dtype='int32')

        # for sub_image in get_random_sub_image(raw_image, image_size, desired_samples):
        #     all_sub_images.append(sub_image)
        all_sub_images = np.array([sub_image for sub_image in get_random_sub_images(raw_image, image_size, desired_samples)])
        if INFO_LEVEL >= 1:
            print(len(all_sub_images))

        train_sub_images, test_sub_images = np.split(all_sub_images, [int(all_sub_images.shape[0] * pareto_limit)])
        if INFO_LEVEL >= 1:
            print(f'Train set: {train_sub_images.shape}')
            print(f'Test set: {test_sub_images.shape}')

    feature_nodes = image_size + image_size
    if input_sblp:
        feature_nodes += 256
    if input_ryser:
        feature_nodes += image_size * image_size

    result_nodes = image_size * image_size

    lh = int((result_nodes-feature_nodes)/2)
    l1 = int(feature_nodes+lh)
    l2 = int(l1+lh)
    if INFO_LEVEL >= 1:
        print(f'{feature_nodes} > {l1} > {l2} > {result_nodes}\n{lh}')
    model = create_model(
        layers=[
            # 64x64
            Layer('input', Dense, l1, activation_function, input_nodes=feature_nodes),
            Layer('hidden-1', Dense, l2, activation_function, dropout_rate=dropout_rate),
            Layer('hidden-2', Dense, l2, activation_function, dropout_rate=dropout_rate),
            Layer('hidden-3', Dense, l2, activation_function, dropout_rate=dropout_rate),

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

    def extract_features(A):
        proj_a = A.sum(axis=0)
        proj_b = A.sum(axis=1)

        X = [proj_a / np.amax(proj_a), proj_b / np.amax(proj_b)]

        if input_sblp:
            X.append(slbp(A))

        if input_ryser:
            X.append(ryser_algorithm(A).reshape(image_size*image_size))

        return np.concatenate(X), A.reshape(result_nodes)

    X_train = []
    Y_train = []
    train_cnt = 0
    train_sub_images_len = len(train_sub_images)
    if INFO_LEVEL >= 1:
        print(f'train set calculation [{train_sub_images_len}]')
    last_percentage = 1
    start = timer()
    for X_train_one in train_sub_images:
        train_cnt += 1
        X, Y = extract_features(X_train_one)
        percentage = int(train_cnt/train_sub_images_len*100)
        if percentage == last_percentage and percentage != 0:
            last_percentage += 1
            end = timer()
            if INFO_LEVEL >= 1:
                print(f'\t{percentage}% \t {train_cnt} from {train_sub_images_len} in {end-start} s')
            start = timer()
        X_train.append(X)
        Y_train.append(Y)
        # print(train_cnt)
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)

    X_test = []
    Y_test = []
    test_cnt = 0
    test_sub_images_len = len(test_sub_images)
    if INFO_LEVEL >= 1:
        print(f'test set calculation [{test_sub_images_len}]')
    last_percentage = 1
    start = timer()
    for X_test_one in test_sub_images:
        test_cnt += 1
        X, Y = extract_features(X_test_one)
        percentage = int(test_cnt/test_sub_images_len*100)
        if percentage == last_percentage and percentage != 0:
            last_percentage += 1
            end = timer()
            if INFO_LEVEL >= 1:
                print(f'\t{percentage}% \t {test_cnt} from {test_sub_images_len} in {end-start} s')
            start = timer()
        X_test.append(X)
        Y_test.append(Y)
    X_test = np.array(X_test)
    Y_test = np.array(Y_test)

    # for r in range(int((raw_rows - image_size - 1)/image_size)):
    #     for c in range(int((raw_cols - image_size - 1)/image_size)):
    #         all_sub_images.append(raw_image[r:r+image_size, c:c+image_size])

    # all_sub_images = np.array(all_sub_images)
    # print(f'Base set: {all_sub_images.shape}')


    # Note: lower batch_size help to the training depending on the training-set size
    # batch_size = int(desired_samples/64)
    batch_size = 64
    epochs = 500
    shuffle = True
    verbose = INFO_LEVEL >= 1
    history = model.fit(
        X_train,
        Y_train,
        batch_size=batch_size,
        epochs=epochs,
        shuffle=shuffle,
        verbose=verbose,
        validation_split=0.25,  # 20% test, 20% validation, 60% training
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

    # summarize history for accuracy
    fig, ax = plt.subplots(nrows=1, ncols=1)  # create figure & 1 axis
    ax.plot(history.history['accuracy'])
    ax.plot(history.history['val_accuracy'])
    ax.set_title('model accuracy')
    ax.set_ylabel('accuracy')
    ax.set_xlabel('epoch')
    ax.legend(['train', 'test'], loc='upper left')
    fig.savefig(f'{result_folder}/accuracy.png')  # save the figure to file
    plt.close(fig)  # close the figure window

    score, accuracy, mae = model.evaluate(X_test, Y_test)
    if INFO_LEVEL >= 1:
        print('Test categorical_crossentropy:', score)
        print('Test accuracy:', accuracy)


    predicts = model.predict(X_test)
    if INFO_LEVEL >= 1:
        print(X_test.shape)
        print(predicts.shape)

    return print_info(
        predicts=predicts,
        Y_test=Y_test,
        image_size=image_size,
        desired_samples=desired_samples,
        result_folder=result_folder,
        input_sblp=input_sblp, input_ryser=input_ryser
    )


if __name__ == '__main__':
    # tf.logging.set_verbosity(tf.logging.ERROR)
    logging.getLogger('matplotlib').setLevel(level=logging.CRITICAL)
    logging.getLogger('tensorflow').setLevel(level=logging.CRITICAL)
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    images = {
        'phan': 'images/phantom_class_02.png',
        # 'brod': 'images/Brodatz_resized_04.png',
        # 'real': 'images/real_9.png',
    }
    size_params = [
        # (32, 10000, 0.8),
        # (16, 10000, 0.1),
        # (16, 10000, 0.2),
        # (16, 10000, 0.3),
        # (16, 10000, 0.4),
        # (16, 10000, 0.5),
    ]
    for optim in ['sgd', 'adam']:
        for extent in [0.1000, 0.0100, 0.0010, 0.0001]:
            for acti in ['linear', 'relu', 'sigmoid']:
                coptim = None
                if optim == 'sgd':
                    coptim = tf.keras.optimizers.SGD(extent)
                if optim == 'adam':
                    coptim = tf.keras.optimizers.Adam(extent)
                size_params.append((16, 5000, None, coptim, optim, extent, acti))
        # (16, 1000, None, tf.keras.optimizers.SGD(0.1), 'sgd', '0.1', 'relu'),
        # (16, 1000, None, tf.keras.optimizers.SGD(0.01), 'sgd', '0.01', 'relu'),
        # (16, 1000, None, tf.keras.optimizers.SGD(0.001), 'sgd', '0.001', 'relu'),
        # (16, 1000, None, tf.keras.optimizers.SGD(0.0001), 'sgd', '0.0001', 'relu'),
        #
        # (16, 1000, None, tf.keras.optimizers.SGD(0.1), 'sgd', '0.1', 'linear'),
        # (16, 1000, None, tf.keras.optimizers.SGD(0.01), 'sgd', '0.01', 'linear'),
        # (16, 1000, None, tf.keras.optimizers.SGD(0.001), 'sgd', '0.001', 'linear'),
        # (16, 1000, None, tf.keras.optimizers.SGD(0.0001), 'sgd', '0.0001', 'linear'),
        #
        # (16, 1000, None, tf.keras.optimizers.Adam(0.1), 'adam', '0.1', 'linear'),
        # (16, 1000, None, tf.keras.optimizers.Adam(0.01), 'adam', '0.01', 'linear'),
        # (16, 1000, None, tf.keras.optimizers.Adam(0.001), 'adam', '0.001', 'linear'),
        # (16, 1000, None, tf.keras.optimizers.Adam(0.0001), 'adam', '0.0001', 'linear'),
        #
        # (16, 1000, None, tf.keras.optimizers.Adam(0.1), 'adam', '0.1', 'relu'),
        # (16, 1000, None, tf.keras.optimizers.Adam(0.0), 'adam', '0.01', 'relu'),
        # (16, 1000, None, tf.keras.optimizers.Adam(0.01), 'adam', '0.01', 'relu'),
        # (16, 1000, None, tf.keras.optimizers.Adam(0.001), 'adam', '0.001', 'relu'),
        # (16, 1000, None, tf.keras.optimizers.Adam(0.0001), 'adam', '0.0001', 'relu'),

    additional_features = [
        [],
        # ['slbp'],
        # ['ryser'],
        # ['slbp', 'ryser'],
    ]
    result_folder_prefix = 'results/optimizer-test-7'
    os.makedirs(result_folder_prefix, exist_ok=True)
    csv_file = open(f'{result_folder_prefix}/summary.csv', 'w')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "seed",
        "image_name",
        "size",
        "samples",
        "sblp_enabled",
        "ryser_enabled",
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

    best = {
        'name': None,
        'avg': None,
    }
    for size_param in size_params:
        (size, desired_samples, dropout_rate, optimizer, optimizer_name, optimiser_extent, activation_function) = size_param
        # samples = int(250 * (size * size) / (8 * 8))
        print(f'{size:>02}x{size:>02}\t{desired_samples} pcs')
        start_per_size_param = timer()
        for image_name in images:
            print(f'\t{image_name}')
            start_per_image = timer()
            for additional_feature in additional_features:
                print(f'\t\tadditional features: {" + ".join(additional_feature)}')

                enable_sblp = 'slbp' in additional_feature
                enable_ryser = 'ryser' in additional_feature
                # Fixing random state for reproducibility
                # tf.keras.utils.set_random_seed(435242)
                # if True:
                result_subfolder = f'{optimizer_name}-{activation_function}-{optimiser_extent}-{image_name}-{size}x{size}-{desired_samples}pcs'
                if enable_sblp:
                    result_subfolder += '-sblp'
                if enable_ryser:
                    result_subfolder += '-ryser'

                result_subfolder += '-'
                if dropout_rate is None:
                    result_subfolder += '0.0'
                else:
                    result_subfolder += str(dropout_rate)
                result_subfolder += 'dropout'

                result_folder = f'{result_folder_prefix}/{result_subfolder}'
                # try:
                if True:
                    start = timer()
                    res = main(
                        result_folder=result_folder,
                        image_path=images[image_name],
                        image_size=size,
                        desired_samples=desired_samples,
                        input_sblp=enable_sblp,
                        input_ryser=enable_ryser,
                        dropout_rate=dropout_rate,
                        optimizer=optimizer,
                        activation_function=activation_function,
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
                        os.path.basename(images[image_name]),
                        size,
                        desired_samples,
                        enable_sblp,
                        enable_ryser,
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
                    if best['name'] is None or best['avg'] > res['nn']['avg']:
                        best['name'] = result_subfolder
                        best['avg'] = res['nn']['avg']
                # except Exception:
                #     print('some error')
            end_per_image = timer()
            print(f'\t\t\t[PER IMAGE]\n\t\t\tTime: {end_per_image-start_per_image:.2f} s\n\n')
        end_per_size_param = timer()
        print(f'\t\t\t[PER SIZE-PARAM]\n\t\t\tTime: {end_per_size_param-start_per_size_param:.2f} s\n\n')

    csv_file.close()
    print(best)
