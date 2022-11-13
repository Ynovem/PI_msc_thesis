import os
import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def relu(x):
    return np.maximum(np.zeros(len(x)), x)


def linear(x):
    return x


def show_function(path, name, fgv, title, values, aspect=None):
    fig, ax = plt.subplots(nrows=1, ncols=1)  # create figure & 1 axis
    ax.plot(values, fgv(values))
    # ax.set_title(title)
    ax.set_xlabel('x')
    if aspect is not None:
        ax.set_aspect(aspect)
    ax.grid(True, which='both')
    # set the x-spine (see below for more info on `set_position`)
    ax.spines['left'].set_position('zero')

    # turn off the right spine/ticks
    ax.spines['right'].set_color('none')
    ax.yaxis.tick_left()

    # set the y-spine
    ax.spines['bottom'].set_position('zero')

    # turn off the top spine/ticks
    ax.spines['top'].set_color('none')
    ax.xaxis.tick_bottom()

    fig.savefig(f'{path}/{name}.png', bbox_inches='tight')  # save the figure to file
    plt.close(fig)  # close the figure window



if __name__ == '__main__':
    base_path = 'diagrams'
    os.makedirs(base_path, exist_ok=True)
    show_function(base_path, 'sigmoid', sigmoid, 'Sigmoid Function', np.arange(-6, 6, 0.1))
    show_function(base_path, 'linear', linear, 'Linear Function', np.arange(-6, 6, 0.1), 'equal')
    show_function(base_path, 'relu', relu, 'ReLU Function', np.arange(-6, 6, 0.1), 'equal')
