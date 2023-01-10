import numpy as np


class FitzHughNagumo:
    def __init__(self):
        self.a = 0.7 # Paper values, not working like hodgkin huxley?
        self.b = 0.8
        self.tau =3
    # Compute derivatives
    def compute_derivatives(self, y, t0, Id):
        dy = np.zeros((2,))

        v = y[0]
        w = y[1]

        # dv/dt
        dy[0] = v - (v**3) / 3 - w + Id(t0)/100

        # dw/dt
        dy[1] = (v + self.a - self.b * w) / self.tau

        return dy
