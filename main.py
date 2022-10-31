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
desired_samples = 100


def main():
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

    feature_nodes_from_lbsp = 256
    feature_nodes_from_image = image_size * image_size

    feature_nodes = feature_nodes_from_lbsp + feature_nodes_from_image
    result_nodes = image_size * image_size



    model_1 = Sequential()

    # ryser
    model_1.add(Dense(512, input_shape=(feature_nodes,), activation='relu', name="hidden-1"))
    model_1.add(Dense(384, activation='relu', name="hidden-2"))
    model_1.add(Dense(result_nodes, activation='relu', name="output"))

    model_1.summary()
    plot_model(model_1, show_shapes=True, show_layer_names=True)



    def custom_loss_function(y_true, y_pred):
    #    squared_difference = tf.square(y_true - y_pred)
    #    return tf.reduce_mean(squared_difference, axis=-1)
        mae = tf.keras.losses.MeanAbsoluteError()
        if custom_loss_function.val == 1:
            tf.print('.')
            # tf.print(y_true)
            # tf.print(y_pred)
            custom_loss_function.val = 0
        # return mae(y_true, y_pred)
        return mae(y_true, tf.convert_to_tensor(y_pred.numpy().round().astype('int').astype('float32')))
    custom_loss_function.val = 1

    # model_1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # model_1.compile(optimizer='adam', loss=tf.keras.losses.MeanAbsolutePercentageError, metrics=['accuracy'])
    # model_1.compile(optimizer='sgd', loss=tf.keras.losses.MeanAbsoluteError(reduction=tf.keras.losses.Reduction.SUM), metrics=['accuracy'])

    model_1.compile(optimizer='adam', loss=tf.keras.losses.MeanAbsoluteError(reduction=tf.keras.losses.Reduction.SUM), metrics=['accuracy'])
    # model_1.compile(optimizer='adam', loss=tf.keras.losses.MeanAbsoluteError(), metrics=['mae'])
    # model_1.compile(optimizer='sgd', loss='mae', metrics=['mae'])
    # model_1.compile(optimizer='sgd', loss=custom_loss_function, metrics=['accuracy'], run_eagerly=True)

    def extractFeatures(A):
        slbp_result = slbp(A)
        predicted_ryser = ryser_algorithm(A.reshape(image_size, image_size))
        return np.concatenate([slbp_result, predicted_ryser.reshape(image_size * image_size)]), A.reshape(result_nodes)
        # return np.concatenate([A.sum(axis=0)/normalization, A.sum(axis=1)/normalization]), A.reshape(result_nodes)

        # train_sub_images
        # test_sub_images
        # prepared_sub_images = np.array([extractFeatures(sub_image) for sub_image in all_sub_images])

    X_train = []
    Y_train = []
    train_cnt = 0
    train_sub_images_len = len(train_sub_images)
    print(f'train set calculation [{train_sub_images_len}]')
    last_percentage = 1
    start = timer()
    for X_train_one in train_sub_images:
        train_cnt += 1
        X, Y = extractFeatures(X_train_one)
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
        X, Y = extractFeatures(X_test_one)
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
    batch_size = 32
    epochs = 10
    shuffle = True

    model_1.fit(
        X_train,
        Y_train,
        batch_size=batch_size,
        epochs=epochs,
        shuffle=shuffle,
        verbose=1
    )

    score, accuracy = model_1.evaluate(X_test, Y_test)
    print('Test categorical_crossentropy:', score)
    print('Test accuracy:', accuracy)


    predicts = model_1.predict(X_test)
    print(X_test.shape)
    print(predicts.shape)
    # print(a[0].reshape(28, 28))


    image_number = 1
    # predicted_nn = a[image_number].round().astype('int').reshape(image_size, image_size)
    predicted_nn = predicts[image_number].round().astype('int').reshape(image_size, image_size)
    predicted_ryser = ryser_algorithm(Y_test[image_number].reshape(image_size, image_size))
    original = Y_test[image_number].reshape(image_size, image_size)

    print(f'{"="*30} ORIGINAL {"="*30}')
    print(original)
    print(f'{"="*30}    NN {"="*30}')
    print(predicted_nn)
    print(f'{"="*30}   RYSER  {"="*30}')
    print(predicted_ryser)


    original_r, original_c = calculate_projections(original)

    print('NN')
    print(f'\trme: {relative_mean_error(original, predicted_nn)}')
    print(f'\tp e: {pixel_error(original, predicted_nn)}')
    nn_r, nn_c = calculate_projections(predicted_nn)
    print(f'\tprojection differences:\n\t\trows: {original_r - nn_r}\n\t\tcols: {original_c - nn_c}')
    print(f'\teuclidean distance:\n\t\trows:{np.linalg.norm(original_r - nn_r)}\n\t\tcols:: {np.linalg.norm(original_c - nn_c)}')

    print('Ryser')
    print(f'\trme: {relative_mean_error(original, predicted_ryser)}')
    print(f'\tp e: {pixel_error(original, predicted_ryser)}')
    ryser_r, ryser_c = calculate_projections(predicted_ryser)
    print(f'\tprojection differences:\n\t\trows: {original_r - ryser_r}\n\t\tcols: {original_c - ryser_c}')
    print(f'\teuclidean distance:\n\t\trows:{np.linalg.norm(original_r - ryser_r)}\n\t\tcols:: {np.linalg.norm(original_c - ryser_c)}')

    # fig, axs = plt.subplots(1, 3)
    # axs[0].set_title('original')
    # axs[0].imshow(original*16, interpolation='none', cmap=plt.cm.binary)
    # axs[1].set_title('NN')
    # axs[1].imshow(predicted_nn, interpolation='none', cmap=plt.cm.binary)
    # axs[2].set_title('Ryser')
    # axs[2].imshow(predicted_ryser, interpolation='none', cmap=plt.cm.binary)
    for i in range(5):
        original = Y_test[i].reshape(image_size, image_size)
        predicted_nn = predicts[i].round().astype('int').reshape(image_size, image_size)
        predicted_ryser = ryser_algorithm(Y_test[i].reshape(image_size, image_size))

        plt.subplots_adjust(top=10)
        fig, axs = plt.subplots(1, 3)
        axs[0].set_title('original\n\nrme:\n pe:\n')
        axs[0].imshow(original, interpolation='none', cmap=plt.cm.binary)

        axs[1].set_title(f'NN\n\n{relative_mean_error(original, predicted_nn):.2f}\n{pixel_error(original, predicted_nn):.2f}\n')
        axs[1].imshow(predicted_nn, interpolation='none', cmap=plt.cm.binary)

        axs[2].set_title(f'Ryser\n\n{relative_mean_error(original, predicted_ryser):.2f}\n{pixel_error(original, predicted_ryser):.2f}\n')
        axs[2].imshow(predicted_ryser, interpolation='none', cmap=plt.cm.binary)

        # fig.figtext(0, 0, f'rme:\n pe:')

        plt.savefig(f'results/{i}.png')


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Fixing random state for reproducibility
    tf.keras.utils.set_random_seed(1)
    main()

