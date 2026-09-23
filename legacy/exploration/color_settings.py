from matplotlib.colors import LinearSegmentedColormap

class rgb:
    def __init__(self):
        print('init rgb')

        self.color_names = ['red', 'green', 'blue']

        self.color_hexcodes = ['#ff0000', '#00ff00', '#0000ff']

        self.cmap_names = [f'{color}_to_black' for color in self.color_names]

        self.cmaps = [LinearSegmentedColormap.from_list(name=self.cmap_names[i], colors=['#000000', self.color_hexcodes[i]])
                 for i in range(len(self.color_hexcodes))]

