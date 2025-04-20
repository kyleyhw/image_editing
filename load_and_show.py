import skimage as ski
import numpy as np
import os
import matplotlib.pyplot as plt

from matplotlib.colors import LinearSegmentedColormap



def load_image(filename):
    image_directory = './images/'
    image_dir = os.path.join(image_directory, filename)
    image = ski.io.imread(image_dir)
    if image.shape[-1] == 4:
        image = ski.color.rgba2rgb(rgba=image)
    print(image)
    return image

def show_image(image):
    plt.figure()
    plt.imshow(image)
    plt.axis(False)
    plt.show()

def separate_rgb(image):
    reordered_axes = np.moveaxis(a=image, source=-1, destination=0)
    color_intensities = tuple(reordered_axes)
    return color_intensities

def plot_separate_colors(image):
    cmap_maxes = ['#ff0000', '#00ff00', '#0000ff']
    cmap_names = [f'{color}_to_black' for color in ['red', 'green', 'blue']]

    cmaps = [LinearSegmentedColormap.from_list(name=cmap_names[i], colors=['#000000', cmap_maxes[i]])
             for i in range(len(cmap_maxes))]
    colors = separate_rgb(image)

    fig, axs = plt.subplots(len(colors)+1, 1)

    axs[0].imshow(image)
    axs[0].set_axis_off()

    for i, color in enumerate(colors):
        ax = axs[i+1]
        ax.imshow(color, cmap=cmaps[i])
        ax.set_axis_off()

    plt.show()




def show_image_and_color_histogram(image, bins=256):
    color_intensities = separate_rgb(image)

    fig, (image_ax, hist_ax) = plt.subplots(2, 1)

    image_ax.imshow(image)
    image_ax.set_axis_off()

    for i in range(len(color_intensities)):
        hist_ax.hist(color_intensities[i], bins=bins, histtype='step', label=str(i))

    hist_ax.legend()

    plt.show()

if __name__ == '__main__':
    image = load_image(filename=('monitor_test_image.jpeg'))
    # image = load_image(filename='test_image_original.jpeg')

    plot_separate_colors(image)



    # show_image_and_color_histogram(image, bins=5)