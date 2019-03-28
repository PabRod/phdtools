import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def stabil(model, y0, tstabil, *args):
    """(Try to) stabilize a dynamical system
    """
    ys = odeint(model, y0, tstabil, *args)
    ylast = ys[-1, :]

    return ylast


class Detflow:
    """ Deterministic flow """

    def __init__(self, f):
        self.f = f

    def plotPhase(self, roiX, roiY, **kwargs):
        """ Plot the phase plane """
        ax = plt.gca()
        X, Y = np.meshgrid(roiX, roiY)
        [U, V] = self.f((X, Y), 0)

        ax.streamplot(X, Y, U, V, **kwargs)
        ax.set_xlim(roiX[0], roiX[-1])
        ax.set_ylim(roiY[0], roiY[-1])
        plt.show()
