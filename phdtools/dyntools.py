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


def polarToCartesian(flow_in_polar):
    """ Convert a flow from polar to cartesian coordinates
    """

    def r(x, y):
        """ Radius coordinate """
        return np.sqrt(x**2 + y**2)

    def th(x, y):
        """ Angle coordinate """
        return np.arctan2(y, x)

    def transform(x, y):
        """ Transformation matrix """
        # TODO a more elegant way would be factorizing this
        # m = np.matrix([[x / r(x, y), -y],
        #                [y / r(x, y), x]])
        #
        # and in flow_in_cartesian use
        #
        # v = np.dot(transform(x, y), flow_in_polar([r(x, y), th(x, y)]))

        m11 = x / r(x, y)
        m12 = -y
        m21 = y / r(x, y)
        m22 = x

        return m11, m12, m21, m22

    def flow_in_cartesian(z, t=0):
        """ Transformed function handle """
        x, y = z
        m11, m12, m21, m22 = transform(x, y)
        flow_pol_eval = flow_in_polar([r(x, y), th(x, y)])

        # This is just an ugly way to represent a matrix product
        # Written like this it is easier to vectorize
        vx = m11*flow_pol_eval[0] + m12*flow_pol_eval[1]
        vy = m21*flow_pol_eval[0] + m22*flow_pol_eval[1]
        return [vx, vy]

    return flow_in_cartesian

class Detflow:
    """ Deterministic flow """

    def __init__(self, f):
        """ Constructor """
        self.f=f

    def plotPhase(self, roiX, roiY, print=False, **kwargs):
        """ Plot the phase plane """
        ax=plt.gca()
        X, Y=np.meshgrid(roiX, roiY)
        [U, V]=self.f((X, Y), 0)

        ax.streamplot(X, Y, U, V, **kwargs)
        ax.set_xlim(roiX[0], roiX[-1])
        ax.set_ylim(roiY[0], roiY[-1])

        if print:
            plt.show()
        else:
            return plt


class Trajectory(Detflow):

    def __init__(self, f, y0, ts):
        """ Constructor """

        # Invoke the __init__ of the parent class
        Detflow.__init__(self, f)

        self.y0 = y0  # Set initial conditions
        self.ts = ts  # Set time span

        self.dims = len(y0)
        self.sol = []

    def solve(self, **kwargs):
        """ Solves initial value problem """

        if (self.sol == []):
            # Solve only if not already solved
            self.sol = odeint(self.f, self.y0, self.ts, **kwargs)
        else:
            # Do nothing
            pass

    def ploty0(self, **kwargs):
        """ Plots the initial state"""

        if(self.dims == 2):
            plt.scatter(self.y0[0], self.y0[1], **kwargs)
            return plt
        else:
            # Throw exception
            pass

    def plotTimeseries(self, **kwargs):
        """ Plots the time series """

        self.solve()

        plt.plot(self.ts, self.sol, **kwargs)
        return plt

    def plotTrajectory(self, print=False, **kwargs):
        """ Plots the trajectory in the phase plane """

        self.solve()

        plt.plot(self.sol[:, 0], self.sol[:, 1], **kwargs)
        return plt
