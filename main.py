import os
import shutil

import numpy as np
from PIL import Image

from keras.utils.vis_utils import plot_model    # a modell vizualizációjához
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from tensorflow.keras.optimizers import Adam

import matplotlib.pyplot as plt
# %matplotlib inline

from timeit import default_timer as timer   # for verbose output

from helpers import *

image_size = 16
# desired_samples = 10000
desired_samples = 1000


class Layer:
    def __init__(self, name, layer_ctor, nodes, activation, input_nodes=0):
        self.layer_ctor = layer_ctor
        self.nodes = nodes
        self.activation = activation
        self.name = name
        self.input_nodes = input_nodes

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


def create_model(layers, model_path):
    model = Sequential()

    for layer in layers:
        model.add(layer.create())

    model.summary()
    plot_model(
        model,
        show_shapes=True,
        show_layer_names=True,
        rankdir='LR',
        show_layer_activations=True,
        to_file=model_path
    )

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['mae'])

    return model

# def create_model(input_nodes, hidden_nodes, activations):
#     model = Sequential()
#
    # model.add(Dense(hidden_nodes[0], input_shape=(input_nodes,), activation=activations[0], name="hidden-1"))
#     model.add(Dense(hidden_nodes[1], activation=activations[1], name="hidden-2"))
#     model.add(Dense(hidden_nodes[2], activation=activations[2], name="output"))
#
#     return model


def print_info(predicts, Y_test, result_folder, input_sblp, input_ryser):
    # image_number = 1
    # predicted_nn = predicts[image_number].round().astype('int').reshape(image_size, image_size)
    # predicted_ryser = ryser_algorithm(Y_test[image_number].reshape(image_size, image_size))
    # original = Y_test[image_number].reshape(image_size, image_size)
    #
    # print(f'{"="*30} ORIGINAL {"="*30}')
    # print(original)
    # print(f'{"="*30}    NN {"="*30}')
    # print(predicted_nn)
    # print(f'{"="*30}   RYSER  {"="*30}')
    # print(predicted_ryser)
    #
    #
    # original_r, original_c = calculate_projections(original)
    #
    # print('NN')
    # print(f'\trme: {relative_mean_error(original, predicted_nn)}')
    # print(f'\tp e: {pixel_error(original, predicted_nn)}')
    # nn_r, nn_c = calculate_projections(predicted_nn)
    # print(f'\tprojection differences:\n\t\trows: {original_r - nn_r}\n\t\tcols: {original_c - nn_c}')
    # print(f'\teuclidean distance:\n\t\trows:{np.linalg.norm(original_r - nn_r)}\n\t\tcols:: {np.linalg.norm(original_c - nn_c)}')
    #
    # print('Ryser')
    # print(f'\trme: {relative_mean_error(original, predicted_ryser)}')
    # print(f'\tp e: {pixel_error(original, predicted_ryser)}')
    # ryser_r, ryser_c = calculate_projections(predicted_ryser)
    # print(f'\tprojection differences:\n\t\trows: {original_r - ryser_r}\n\t\tcols: {original_c - ryser_c}')
    # print(f'\teuclidean distance:\n\t\trows:{np.linalg.norm(original_r - ryser_r)}\n\t\tcols:: {np.linalg.norm(original_c - ryser_c)}')

    # fig, axs = plt.subplots(1, 3)
    # axs[0].set_title('original')
    # axs[0].imshow(original*16, interpolation='none', cmap=plt.cm.binary)
    # axs[1].set_title('NN')
    # axs[1].imshow(predicted_nn, interpolation='none', cmap=plt.cm.binary)
    # axs[2].set_title('Ryser')
    # axs[2].imshow(predicted_ryser, interpolation='none', cmap=plt.cm.binary)
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
        predicted_NN = predicts[i].round().astype('int').reshape(image_size, image_size)
        predicted_ryser = ryser_algorithm(Y_test[i].reshape(image_size, image_size))
        original = Y_test[i].reshape(image_size, image_size)

        results['ryser']['rme'].append(relative_mean_error(original, predicted_ryser))
        results['ryser']['pe'].append(pixel_error(original, predicted_ryser))
        results['nn']['rme'].append(relative_mean_error(original, predicted_NN))
        results['nn']['pe'].append(pixel_error(original, predicted_NN))

    # print(f'min\t{np.min(a)}')
    # print(f'mean\t{np.mean(a)}')
    # print(f'median\t{np.median(a)}')
    # print(f'max\t{np.max(a)}')
    # print(f'std\t{np.std(a)}')

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


def main(result_folder_prefix, input_sblp, input_ryser):
    result_folder = result_folder_prefix
    if input_sblp:
        result_folder += '-sblp'
    if input_ryser:
        result_folder += '-ryser'
    result_folder += f'-{image_size}x{image_size}'
    result_folder += f'-{desired_samples}PCS'
    os.mkdir(result_folder)

    image = Image.open('images/phantom_class_02.png')
    image.load()
    raw_image = np.asarray(image, dtype='int32')

    # for sub_image in get_random_sub_image(raw_image, image_size, desired_samples):
    #     all_sub_images.append(sub_image)
    all_sub_images = np.array([sub_image for sub_image in get_random_sub_image(raw_image, image_size, desired_samples)])
    print(len(all_sub_images))

    pareto_limit = 0.8

    train_sub_images, test_sub_images = np.split(all_sub_images, [int(all_sub_images.shape[0] * pareto_limit)])
    print(f'Train set: {train_sub_images.shape}')
    print(f'Test set: {test_sub_images.shape}')

    feature_nodes = image_size + image_size
    if input_sblp:
        feature_nodes += 256
    if input_ryser:
        feature_nodes += image_size * image_size

    result_nodes = image_size * image_size

    model = create_model(
        layers=[
            Layer('input', Dense, 512, 'relu', input_nodes=feature_nodes),
            Layer('hidden-1', Dense, 2048, 'relu'),
            Layer('output', Dense, result_nodes, 'sigmoid'),
        ],
        model_path=f'{result_folder}/model.png',
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
    batch_size = 16
    epochs = 10
    shuffle = True

    model.fit(
        X_train,
        Y_train,
        batch_size=batch_size,
        epochs=epochs,
        shuffle=shuffle,
        verbose=1
    )

    score, accuracy = model.evaluate(X_test, Y_test)
    print('Test categorical_crossentropy:', score)
    print('Test accuracy:', accuracy)


    predicts = model.predict(X_test)
    print(X_test.shape)
    print(predicts.shape)

    print_info(
        predicts=predicts,
        Y_test=Y_test,
        result_folder=result_folder,
        input_sblp=input_sblp, input_ryser=input_ryser
    )



if __name__ == '__main__':
    # Fixing random state for reproducibility
    tf.keras.utils.set_random_seed(1)
    main(
        result_folder_prefix='results',
        input_sblp=True,
        input_ryser=False
    )
