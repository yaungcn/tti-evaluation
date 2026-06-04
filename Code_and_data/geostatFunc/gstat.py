import numpy as np
import gstools as gs
import os
import matplotlib.pyplot as plt


# class geostat
class geostatistical:
    def __init__(self, coords, values, bins, max_dist, output_path):
        self.coords = coords
        self.values = values
        self.bins = bins
        self.max_dist = max_dist
        self.bin_center = None
        self.gamma = None
        self.bin_center_x = None
        self.gamma_x = None
        self.bin_center_y = None
        self.gamma_y = None
        self.output_path = output_path

    def fit_variogram(self, model):
        """
        拟合变异函数模型
        :param model: 变异函数模型
        :return: 拟合的变异函数模型
        """
        bin_center, gamma = gs.vario_estimate(self.coords.T, self.values, self.bins)
        self.bin_center = bin_center
        self.gamma = gamma
        model.fit_variogram(bin_center, gamma, nugget=False)
        return model

    def fit_x_direction(self, model):
        """
        拟合x方向变异函数模型
        :param model: 变异函数模型
        :return: 拟合的变异函数模型
        """
        bin_center, gamma = gs.vario_estimate(
            self.coords.T, self.values, self.bins, direction=[1, 0]
        )
        self.bin_center_x = bin_center
        self.gamma_x = gamma
        model.fit_variogram(bin_center, gamma, nugget=False)
        return model

    def fit_y_direction(self, model):
        """
        拟合y方向变异函数模型
        :param model: 变异函数模型
        :return: 拟合的变异函数模型
        """
        bin_center, gamma = gs.vario_estimate(
            self.coords.T, self.values, self.bins, direction=[0, 1]
        )
        self.bin_center_y = bin_center
        self.gamma_y = gamma
        model.fit_variogram(bin_center, gamma, nugget=False)
        return model

    def plot_variogram(self, model, title="Variogram Fitting Result"):
        ax = model.plot(x_max=self.max_dist)
        ax.scatter(self.bin_center, self.gamma, color="k", label="variogram data")
        ax.legend()
        plt.xlabel("distance")
        plt.ylabel("semivariance")
        plt.title(title)
        output_path = os.path.join(self.output_path, title + ".png")
        plt.savefig(output_path)
        plt.close()

    def plot_variogram_x(self, model, title="Variogram Fitting Result (X Direction)"):
        ax = model.plot(x_max=self.max_dist)
        ax.scatter(
            self.bin_center_x,
            self.gamma_x,
            color="k",
            label="variogram data (X Direction)",
        )
        ax.legend()
        plt.xlabel("distance")
        plt.ylabel("semivariance")
        plt.title(title)
        output_path = os.path.join(self.output_path, title + ".png")
        plt.savefig(output_path)
        plt.close()

    def plot_variogram_y(self, model, title="Variogram Fitting Result (Y Direction)"):
        ax = model.plot(x_max=self.max_dist)
        ax.scatter(
            self.bin_center_y,
            self.gamma_y,
            color="k",
            label="variogram data (Y Direction)",
        )
        ax.legend()
        plt.xlabel("distance")
        plt.ylabel("semivariance")
        plt.title(title)
        output_path = os.path.join(self.output_path, title + ".png")
        plt.savefig(output_path)
        plt.close()
