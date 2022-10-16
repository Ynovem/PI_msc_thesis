import numpy as np
from PIL import Image

from keras.utils.vis_utils import plot_model    # a modell vizualizációjához
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from tensorflow.keras.optimizers import Adam

from timeit import default_timer as timer   # for verbose output

from random import randint


image_size = 32
desired_samples = 10000
slbp_normalization = 'max'
slbp_normalization = 'by_step'


def relative_mean_error(original, predicted):
    if original.shape != predicted.shape:
        print(f'Error: shape\'s not similar: {original.shape} != {predicted.shape}')
    return np.abs(original - predicted).sum() / original.sum() * 100


def pixel_error(original, predicted):
    if original.shape != predicted.shape:
        print(f'Error: shape\'s not similar: {original.shape} != {predicted.shape}')
    return np.abs(original - predicted).sum() / (original.shape[0] * original.shape[1]) * 100


def calculate_projections(data):
    return (data.sum(axis=1), data.sum(axis=0))


def ryser_algorithm(data):
    data = data.copy()

    r, s = calculate_projections(data)

    tmp = [(s[i], i) for i in range(len(s))]
    tmp.sort()
    s_, permutation = zip(*tmp[::-1])
    s_ = np.asarray(s_)
    permutation = np.asarray(permutation)

    n = s_.size
    B = np.zeros((n, n), dtype='int32')
    for i in range(n):
        for j in range(r[i]):
            B[i][j] = 1

    for i in range(n - 1, 0, -1):
        sb = B.sum(axis=0)
        if sb[i] < s_[i]:
            diff = s_[i] - sb[i]
            if diff == 0:
                continue
            for j in range(i - 1, -1, -1):
                for k in range(n):
                    if B[k][i] == 0 and B[k][j] == 1:
                        diff -= 1
                        B[k][i], B[k][j] = B[k][j], B[k][i]
                    if diff == 0:
                        break
                if diff == 0:
                    break
    result = np.zeros((n, n), dtype='int32')
    for i in range(len(permutation)):
        result[:, permutation[i]] = B[:, i]

    if not np.array_equal(r, result.sum(axis=1)):
        raise Exception(f'Ryser: R is not equal in origin and result data:\n{r}\n{result.sum(axis=1)}')
    if not np.array_equal(s, result.sum(axis=0)):
        raise Exception(f'Ryser: S is not equal in origin and result data:\n{s}\n{result.sum(axis=0)}')

    return result


sample_points = np.array([
    [-1, -1],
    [-1, 0],
    [-1, 1],
    [0, 1],
    [1, 1],
    [1, 0],
    [1, -1],
    [0, -1],
])


def slbp(image, intensity_level=3):
    r, c = image.shape
    interval = np.arange(-intensity_level, intensity_level+1)
    slbp_result = np.zeros([256], dtype='int32')
    normalization_factor = 0

    for i in range(1, r-1):
        for j in range(1, c-1):
            temp_im = np.copy(image[i-1:i+2, j-1:j+2])
            center_point = temp_im[1][1]
            slbp_dict = dict()
            for k in interval:
                slbp_str = ''
                for sample_point in sample_points:
                    if temp_im[1+sample_point[0]][1+sample_point[1]] >= center_point+k:
                        slbp_str += '1'
                    else:
                        slbp_str += '0'
                slbp_int = int(slbp_str[::-1], 2)
                if slbp_int not in slbp_dict:
                    slbp_dict[slbp_int] = 1
                else:
                    slbp_dict[slbp_int] += 1
            if slbp.normalization_mode == 'by_step':
                normalization_factor += len(interval)
            for slbp_k, slbp_v in slbp_dict.items():
                slbp_result[slbp_k] += slbp_v
    if slbp.normalization_mode == 'by_step':
        return slbp_result/normalization_factor
    if slbp.normalization_mode == 'max':
        return slbp_result/np.amax(slbp_result)
    raise 'slbp-error'


slbp.normalization_mode = slbp_normalization

# sample = np.array([
#     [200,   0, 255],
#     [ 83, 132, 156],
#     [132,  10, 130],
# ])
# # 200    0  255
# #  83  132  156
# # 132   10  130
# res = slbp(sample)
# print(f'result: {res}')


def main():
    image = Image.open('images/phantom_class_02.png')
    image.load()
    raw_image = np.asarray(image, dtype='int32')

    raw_cols, raw_rows = raw_image.shape

    def get_random_sub_image(image, size, cnt):
        row_max = raw_rows - size - 1
        col_max = raw_cols - size - 1
        for i in range(cnt):
            r = randint(0, row_max)
            c = randint(0, col_max)
            s = raw_image[c:c + size, r:r + size]
            if s.shape != (size, size):
                raise (f'error [{size}]: {r}, {c}, {r + size}, {c + size}')
            yield s

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
    model_1.add(Dense(512, input_shape=(512,), activation='relu', name="hidden-1"))
    model_1.add(Dense(384, activation='relu', name="hidden-2"))
    model_1.add(Dense(256, activation='relu', name="output"))

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
        return np.concatenate([slbp_result, predicted_ryser.reshape(image_size*image_size)]), A.reshape(result_nodes)
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


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

