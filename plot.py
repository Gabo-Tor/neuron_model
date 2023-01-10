from hodgkin_huxley import  HodgkinHuxley
from Izhikevich import Izhikevich
from leaky_integrate_and_fire import LeakyIntegrateAndFire
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Start and end time (in milliseconds)
tmin = 0.0
tmax = 160.0

# Time values
T = np.linspace(tmin, tmax, 10000)


# Input stimulus
def Id(t):
    if 0.0 < t < 5.0:
        return 10.0
    elif 50.0 < t <= 70.0:
        return 40.0
    elif 80.0 < t < 110.0:
        return 10.0
    elif 120.0 < t < 160.0:
        return 4.0
    return 0.0

# Hodgkin Huxley model
hh = HodgkinHuxley()

# State (Vm, n, m, h)
Y = np.array([0.0, hh.n_inf(), hh.m_inf(), hh.h_inf()])

# Solve ODE system
# Vy = (Vm[t0:tmax], n[t0:tmax], m[t0:tmax], h[t0:tmax])
hhVy = odeint(hh.compute_derivatives, Y, T, args=(Id,))

# Leaky integrate and fire model
lif = LeakyIntegrateAndFire()

# State (Vm)
Y = np.array([-70.0])

# Solve ODE system
lifVy = lif.integrate(Y, T, Id)


# # FitzHugh Nagumo model
# fn = FitzHughNagumo()

# # State (Vm, w)
# Y = np.array([0.0, 0.0])

# # Solve ODE system
# fnVy = odeint(fn.compute_derivatives, Y, T, args=(Id,))

# Izhikevich model
iz = Izhikevich()

# State (Vm, u)
y0 = np.array([-65.0, -65.0*0.2])
iVy = iz.integrate( y0, T, Id)



# Input stimulus
Idv = [Id(t) for t in T]

fig, ax = plt.subplots(2)

ax[0].plot(T, Idv, "r")
ax[0].set_xlabel("Time (ms)")
ax[0].set_ylabel(r"Densidad de corriente (uA/$cm^2$)")
ax[0].set_title("Estímulo")
ax[0].label_outer()
plt.grid()

# Neuron potential
ax[1].plot(T, hhVy[:, 0], label="Hodgkin Huxley")
ax[1].plot(T, iVy[:, 0], label="Izhikevich")
ax[1].plot(T, lifVy[:], label="Leaky Integrate and Fire")
# ax[1].plot(T, iVy[:, 1], label="Izhikevich u DEBUG")
ax[1].legend()
ax[1].set_xlabel("Tiempo (ms)")
ax[1].set_ylabel("Vm (mV)")
ax[1].set_title("Potencial de membrana")
plt.grid()

# # Trajectories with limit cycles
# fig, ax = plt.subplots(figsize=(10, 10))
# ax.plot(iVy[:, 0], iVy[:, 1], label="Vm - w")
# ax.set_title("Limit cycles")
# ax.legend()
# plt.grid()

# # Trajectories with limit cycles
# fig, ax = plt.subplots(figsize=(10, 10))
# ax.plot(hhVy[:, 0], hhVy[:, 1], label="Vm - n")
# ax.plot(hhVy[:, 0], hhVy[:, 2], label="Vm - m")
# ax.set_title("Limit cycles")
# ax.legend()
# plt.grid()

plt.show()
