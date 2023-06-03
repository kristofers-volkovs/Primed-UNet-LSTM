import matplotlib.pyplot as plt


class PlotUtils:

    @staticmethod
    def plot_regular_grid(row_count: int, x, y_prim, y, data_min: int, data_max: int, title_ax1: str, title_ax2: str,
                          title_ax3: str, map_color: str):
        """
        Plots x, y_prim and y in a grid (in this order)

        :param row_count: int
        :param x: FloatTensor
        :param y_prim: FloatTensor
        :param y: FloatTensor
        :param data_min: int
        :param data_max: int
        :param title_ax1: str
        :param title_ax2: str
        :param title_ax3: str
        :param map_color: str
        :return: figure
        """
        fig, axes = plt.subplots(row_count, 3)

        if row_count < 2:
            axes[0].set_title(title_ax1)
            axes[1].set_title(title_ax2)
            axes[2].set_title(title_ax3)

            for idx in range(row_count):
                axes[0].imshow(x[idx], cmap=map_color, vmin=data_min, vmax=data_max, interpolation=None)
                axes[1].imshow(y_prim[idx], cmap=map_color, vmin=data_min, vmax=data_max, interpolation=None)
                axes[2].imshow(y[idx], cmap=map_color, vmin=data_min, vmax=data_max, interpolation=None)
        else:
            axes[0, 0].set_title(title_ax1)
            axes[0, 1].set_title(title_ax2)
            axes[0, 2].set_title(title_ax3)

            for idx in range(row_count):
                axes[idx, 0].imshow(x[idx], cmap=map_color, vmin=data_min, vmax=data_max, interpolation=None)
                axes[idx, 1].imshow(y_prim[idx], cmap=map_color, vmin=data_min, vmax=data_max, interpolation=None)
                axes[idx, 2].imshow(y[idx], cmap=map_color, vmin=data_min, vmax=data_max, interpolation=None)

        return fig
