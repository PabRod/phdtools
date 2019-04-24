import numpy as np
from scipy.integrate import odeint
import sdeint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def stabil(model, y0, tstabil, **kwargs):
    """(Try to) stabilize a dynamical system
    """
    ys = odeint(model, y0, tstabil, **kwargs)
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
        # TODO: a more elegant way would be factorizing this
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
        vx = m11 * flow_pol_eval[0] + m12 * flow_pol_eval[1]
        vy = m21 * flow_pol_eval[0] + m22 * flow_pol_eval[1]
        return [vx, vy]

    return flow_in_cartesian


class Detflow:
    """ Deterministic flow """

    def __init__(self, f, dims):
        """ Constructor """
        self.f = f
        self.dims = dims
        self.eqs = []
        self.eqs_classes = []

    def jac(self, y, step=1e-6):
        """ Numeric jacobian """

        N = self.dims
        J = np.empty((N, N))
        step_matrix = step * np.eye(N)

        for i in range(N):
            J[:, i] = (np.asarray(self.f(y + step_matrix[:, i])) - np.asarray(self.f(y)))/step

        return J

    def findeq(self, y_guess):
        """ Searches for equilibria """

        from scipy.optimize import fsolve

        eq =  fsolve(self.f, y_guess)
        eq_class = self.claseq(eq)

        if list(eq) in self.eqs:
            # Do nothing
            pass
        else:
            # Append found equilibrium to list
            self.eqs.append(list(eq))

            # Append class of equilibrium
            self.eqs_classes.append(eq_class)

        return eq, eq_class

    def claseq(self, eq, step=1e-6):
        """ Classifies the equilibrium point """

        ml = self.maxlyapunov(eq, step)
        if ml < 0.0:
            return "stable"
        elif ml > 0:
            return "unstable"
        else:
            return "center"

    def attractor(self, y0, tstabil, tattr, **kwargs):
        """ Estimates the attractor integrating after stabilization
        """
        ylast = stabil(self.f, y0, tstabil)
        yattr = odeint(self.f, ylast, tattr, **kwargs)

        return yattr

    def lyapunovs(self, y, step=1e-6):
        """ Numeric lyapunov exponents """

        J = self.jac(y, step)
        lyaps = np.linalg.eigvals(J)

        return lyaps

    def maxlyapunov(self, y, step=1e-6):
        """ Numeric maximum lyapunov """

        lyaps = self.lyapunovs(y, step)
        lyaps_real_parts = np.real(lyaps)

        return np.max(lyaps_real_parts)

    def plotPhase(self, roiX, roiY, **kwargs):
        """ Plot the phase plane """

        ax = plt.gca()
        X, Y = np.meshgrid(roiX, roiY)
        [U, V] = self.f((X, Y), 0)

        ax.streamplot(X, Y, U, V, **kwargs)
        ax.set_xlim(roiX[0], roiX[-1])
        ax.set_ylim(roiY[0], roiY[-1])

        return plt

    def plotEqs(self, **kwargs):
        """ Plot the equilibria"""

        ax = plt.gca()
        for i in range(len(self.eqs)):
            # Unpack each found equilibria
            eq = self.eqs[i]
            st = self.eqs_classes[i]

            # Color according to its type
            if (st == 'stable'):
                ax.scatter(eq[0], eq[1], color = 'black', **kwargs)
            elif (st == 'unstable'):
                ax.scatter(eq[0], eq[1], color = 'white', edgecolors = 'black', **kwargs)
            elif (st == 'center'):
                ax.scatter(eq[0], eq[1], color = 'grey', edgecolors = 'black', **kwargs)
            else:
                raise ValueError('An unexpected equilibrium type was declared')

        return plt

class Trajectory(Detflow):

    def __init__(self, f, y0, ts):
        """ Constructor """

        # Invoke the __init__ of the parent class
        if (isinstance(y0, float) | isinstance(y0, int)):
            ## Required for 1D systems, where y0 has no attribute len
            Detflow.__init__(self, f, 1)
        else:
            Detflow.__init__(self, f, len(y0))

        self.y0 = y0  # Set initial conditions
        self.ts = ts  # Set time span

        self.sol = []

    def solve(self, **kwargs):
        """ Solves initial value problem """

        if (self.sol == []):
            # Solve only if not already solved
            self.sol = odeint(self.f, self.y0, self.ts, **kwargs)
        else:
            # Do nothing
            pass

    def maxlyapunov(self, step=1e-6):
        """ Gets the maximum lyapunov for each time step"""

        self.solve()

        ntimes = len(self.ts)
        ml = np.empty(ntimes)
        i = 0
        for i in range(ntimes):
            ml[i] = Detflow.maxlyapunov(self, self.sol[i, :], step)
            i =+ 1

        return ml

    def ploty0(self, **kwargs):
        """ Plots the initial state"""

        if(self.dims == 2):
            plt.scatter(self.y0[0], self.y0[1], **kwargs)
            return plt
        else:
            # Throw exception
            raise ValueError('Only available for 2 dimensions')

    def plotTimeseries(self, **kwargs):
        """ Plots the time series """

        self.solve()

        plt.plot(self.ts, self.sol, **kwargs)
        return plt

    def plotMaxLyapunov(self, **kwargs):
        """ Plots the time series of the maximum Lyapunovs"""

        self.solve()

        plt.plot(self.ts, self.maxlyapunov(), **kwargs)
        return plt

    def plotTrajectory(self, print=False, **kwargs):
        """ Plots the trajectory in the phase plane """

        self.solve()

        if(self.dims == 2):
            plt.plot(self.sol[:, 0], self.sol[:, 1], **kwargs)
            return plt
        else:
            # Throw exception
            raise ValueError('Only available for 2 dimensions')

class stoTrajectory(Trajectory):

    def __init__(self, f, G, y0, ts):
        """ Constructor """

        # Invoke the __init__ of the parent class
        Trajectory.__init__(self, f, y0, ts)

        self.G = G # Set stochastic term
        self.y0 = y0  # Set initial conditions
        self.ts = ts  # Set time span

        self.sol = []

    def solve(self, **kwargs):
        """ Solves initial value problem """

        if (self.sol == []):
            # Solve only if not already solved
            self.sol = sdeint.itoint(self.f, self.G, self.y0, self.ts, **kwargs)
        else:
            # Do nothing
            pass
