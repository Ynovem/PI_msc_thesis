import numpy as np
from random import randint


def relative_mean_error(original, predicted):
    if original.shape != predicted.shape:
        print(f'Error: shape\'s not similar: {original.shape} != {predicted.shape}')
    return np.abs(original - predicted).sum() / original.sum() * 100


def pixel_error(original, predicted):
    if original.shape != predicted.shape:
        print(f'Error: shape\'s not similar: {original.shape} != {predicted.shape}')
    return np.abs(original - predicted).sum() / (original.shape[0] * original.shape[1]) * 100


def calculate_projections(data):
    return data.sum(axis=1), data.sum(axis=0)


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
                for sample_point in slbp.sample_points:
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


slbp_normalization = 'max'
# slbp_normalization = 'by_step'
slbp.normalization_mode = slbp_normalization
slbp.sample_points = np.array([
    [-1, -1],
    [-1, 0],
    [-1, 1],
    [0, 1],
    [1, 1],
    [1, 0],
    [1, -1],
    [0, -1],
])

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


def get_random_sub_image(image, size, cnt):
    raw_cols, raw_rows = image.shape

    row_max = raw_rows - size - 1
    col_max = raw_cols - size - 1
    for i in range(cnt):
        r = randint(0, row_max)
        c = randint(0, col_max)
        s = image[c:c + size, r:r + size]
        if s.shape != (size, size):
            raise (f'error [{size}]: {r}, {c}, {r + size}, {c + size}')
        yield s

