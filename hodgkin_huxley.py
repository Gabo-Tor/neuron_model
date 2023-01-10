import numpy as np
# Based on https://www.bonaccorso.eu/2017/08/19/hodgkin-huxley-spiking-neuron-model-python/
# https://gist.github.com/giuseppebonaccorso/60ce3eb3a829b94abf64ab2b7a56aaef

class HodgkinHuxley:
    def __init__(self):
        self.gK = 36.0  # Average potassium channel conductance per unit area (mS/cm^2)
        self.gNa = 120.0  # Average sodium channel conductance per unit area (mS/cm^2)
        self.gL = 0.3  # Average leak channel conductance per unit area (mS/cm^2)
        self.Cm = 1.0  # Membrane capacitance per unit area (uF/cm^2)
        self.VK = -12.0  # Potassium potential (mV)
        self.VNa = 115.0  # Sodium potential (mV)
        self.Vl = 10.613  # Leak potential (mV)

    # Potassium ion-channel rate functions
    def alpha_n(self, Vm):
        return (0.01 * (10.0 - Vm)) / (np.exp(1.0 - (0.1 * Vm)) - 1.0)

    def beta_n(self, Vm):
        return 0.125 * np.exp(-Vm / 80.0)

    # Sodium ion-channel rate functions
    def alpha_m(self, Vm):
        return (0.1 * (25.0 - Vm)) / (np.exp(2.5 - (0.1 * Vm)) - 1.0)

    def beta_m(self, Vm):
        return 4.0 * np.exp(-Vm / 18.0)

    def alpha_h(self, Vm):
        return 0.07 * np.exp(-Vm / 20.0)

    def beta_h(self, Vm):
        return 1.0 / (np.exp(3.0 - (0.1 * Vm)) + 1.0)

    # n, m, and h steady-state values

    def n_inf(self, Vm=0.0):
        return self.alpha_n(Vm) / (self.alpha_n(Vm) + self.beta_n(Vm))

    def m_inf(self, Vm=0.0):
        return self.alpha_m(Vm) / (self.alpha_m(Vm) + self.beta_m(Vm))

    def h_inf(self, Vm=0.0):
        return self.alpha_h(Vm) / (self.alpha_h(Vm) + self.beta_h(Vm))

    # Compute derivatives
    def compute_derivatives(self, y, t0, Id):
        dy = np.zeros((4,))

        Vm = y[0]
        n = y[1]
        m = y[2]
        h = y[3]

        # dVm/dt
        GK = (self.gK / self.Cm) * np.power(n, 4.0)
        GNa = (self.gNa / self.Cm) * np.power(m, 3.0) * h
        GL = self.gL / self.Cm

        dy[0] = (
            (Id(t0) / self.Cm)
            - (GK * (Vm - self.VK))
            - (GNa * (Vm - self.VNa))
            - (GL * (Vm - self.Vl))
        )

        # dn/dt
        dy[1] = (self.alpha_n(Vm) * (1.0 - n)) - (self.beta_n(Vm) * n)
        # dm/dt
        dy[2] = (self.alpha_m(Vm) * (1.0 - m)) - (self.beta_m(Vm) * m)
        # dh/dt
        dy[3] = (self.alpha_h(Vm) * (1.0 - h)) - (self.beta_h(Vm) * h)

        return dy
