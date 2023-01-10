import numpy as np


class Izhikevich:
    def __init__(self):
        self.a = 0.02
        self.b = 0.2
        self.c = -65
        self.d = 8

    # Compute derivatives
    def compute_derivatives(self, y, t0, Id):
        dy = np.zeros((2,))

        v = y[0]
        u = y[1]

        # dv/dt
        dy[0] = 0.04 * v**2 + 5 * v + 140 - u + Id(t0)*2

        # du/dt
        dy[1] = self.a * (self.b * v - u)

        return dy

    def integrate(self, y0, T, Id):
        Y = np.zeros((len(T), 2))
        Y[0] = y0
        for i in range(1, len(T)):
            # Compute derivatives
            if Y[i - 1, 0] >= 30:
                Y[i, 0] = self.c
                Y[i, 1] += self.d
            else:
                dy = self.compute_derivatives(Y[i - 1], T[i - 1], Id)
                # Euler integration
                Y[i] = Y[i - 1] + dy * (T[i] - T[i - 1])
        return Y
