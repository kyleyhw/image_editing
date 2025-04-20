import skimage as ski
import numpy as np
import os
import matplotlib.pyplot as plt

import color_settings

colorspace = globals.rgb()


def load_image(filename, test=True):
    image_directory = './images/original/'
    if test:
        image_directory = './images/test_images/'
    image_dir = os.path.join(image_directory, filename)
    image = ski.io.imread(image_dir)
    if image.shape[-1] == 4:
        image = ski.color.rgba2rgb(rgba=image)
    return image

def show_image(image, ax):
    ax.imshow(image)
    ax.set_axis_off()

def separate_rgb(image):
    reordered_axes = np.moveaxis(a=image, source=-1, destination=0)
    color_intensities = tuple(reordered_axes)
    return color_intensities

def plot_separate_colors(image, axs):
    axs = axs.ravel()

    color_names = ['red', 'green', 'blue']


    colors = separate_rgb(image)



    axs[0].imshow(image)
    axs[0].set_axis_off()
    axs[0].set_title('original')

    for i, color in enumerate(colors):
        ax = axs[i+1]
        ax.imshow(color, cmap=colorspace.cmaps[i])
        ax.set_axis_off()
        ax.set_title(color_names[i])


def get_histogram(image, bins=256):
    counts, bins = np.histogram(image.ravel(), bins=bins)
    cdf = counts.cumsum()
    # cdf = cdf / len(counts)
    return counts, bins, cdf

def plot_image_and_color_histogram(image, axs, bins=256):
    color_intensities = separate_rgb(image)

    image_ax, hist_ax = axs
    cdf_ax = hist_ax.twinx()
    # exposure_ax = hist_ax.twinx()

    image_ax.imshow(image)
    image_ax.set_axis_off()

    for i in range(len(color_intensities)):
        color_counts, color_bins, color_cdf = get_histogram(color_intensities[i], bins=bins)
        hist_ax.stairs(color_counts, color_bins, label=colorspace.color_names[i], color=colorspace.color_hexcodes[i])
        cdf_ax.plot(color_cdf, color=colorspace.color_hexcodes[i])


    exposure_counts, exposure_bins, exposure_cdf = get_histogram(image, bins=bins)
    # exposure_ax.stairs(exposure_counts, exposure_bins, label='exposure', color='black')
    # exposure_ax.plot(exposure_cdf, color='black')


    # hist_ax.legend()
    # exposure_ax.legend()
    cdf_ax.set_axis_off()

    hist_ax.set_ylabel('pixel count')
    hist_ax.set_xlabel('pixel intensity')

def create_and_show_separate_colors(image, show=True, save=None):
    fig, axs = plt.subplots(2, 2)

    plot_separate_colors(image=image, axs=axs)

    fig.tight_layout()

    if show:
        plt.show()
    if save:
        fig.savefig(fname=f'{save}.png')

def create_and_show_color_histogram(image, show=True, save=None):
    fig, axs = plt.subplots(2, 1)

    plot_image_and_color_histogram(image=image, axs=axs, bins=256)

    fig

    if show:
        plt.show()
    if save:
        fig.savefig(fname=f'{save}.png')


if __name__ == '__main__':
    # filename = 'monitor_test_original.jpeg'
    # filename = 'test_image_original.jpeg'
    # filename = 'rgb_test_original.jpeg'
    filename = 'climbing_test_original.jpeg'
    image = load_image(filename=filename)

    create_and_show_separate_colors(image=image)
    create_and_show_color_histogram(image=image)



    # show_image_and_color_histogram(image, bins=5)