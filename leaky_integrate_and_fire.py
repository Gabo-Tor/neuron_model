import numpy as np


class LeakyIntegrateAndFire:
    def __init__(self):
        self.urest = -70.0
        self.upeak = 35.0
        self.tau = 10

    def compute_derivatives(self, y, t0, Id):
        v = y
        # dv/dt
        dy = (self.urest - v + Id(t0)*30) / self.tau
        return dy

    def integrate(self, y0, T, Id):
        Y = np.zeros((len(T)))
        V = np.repeat(self.urest, len(T))                       
        Y[0] = y0

        for i in range(1, len(T)):
            # Compute derivatives
            if Y[i - 1] >= self.upeak:
                # Fire
                Y[i] = self.urest
                V[i] = self.upeak
            else:
                dy = self.compute_derivatives(Y[i - 1], T[i - 1], Id)
                # Euler integration
                Y[i] = Y[i - 1] + dy * (T[i] - T[i - 1])

        return V
