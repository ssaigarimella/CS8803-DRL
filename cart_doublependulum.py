import math
import autograd.numpy as np
import matplotlib.pyplot as plt
from autograd.numpy import sin, cos, log
from autograd.numpy.random import randn, multivariate_normal
from matplotlib import animation

from pilco import Empty
from pilco.base import rollout, train, propagate, learn
from pilco.control import congp, concat
from pilco.gp import GPModel
from pilco.loss import Loss
from pilco.util import gaussian_trig, gaussian_sin, fill_mat


def dynamics(z, t, u):
    g = 9.82
    L1 = 0.6  # length of first pendulum
    L2 = 0.6  # length of second pendulum
    m1 = 0.5  # mass of first pendulum
    m2 = 0.5  # mass of second pendulum
    mc = 1.0  # mass of the cart
    b = 0.1   # damping coefficient

    # state variables
    x, x_dot, theta1, theta1_dot, theta2, theta2_dot = z

    sin_theta1, cos_theta1 = sin(theta1), cos(theta1)
    sin_theta2, cos_theta2 = sin(theta2), cos(theta2)

    # intermediate terms used in equations of motion
    d1 = mc + m1 + m2
    d2 = m1 * L1 * cos_theta1 + m2 * (L1 * cos_theta1 + L2 * cos_theta2)
    d3 = (4/3) * m1 * L1**2 + m2 * (L1**2 + L2**2 + 2 * L1 * L2 * cos(theta2))
    d4 = m2 * (L2**2 + L1 * L2 * cos(theta2))

    # compute determinant for inverse of matrix later
    detM = d1 * (d3 - d4**2 / d1) - d2**2
    invDet = 1 / detM

    # compute forces acting on system
    f1 = -m1 * L1 * sin_theta1 * theta1_dot**2 - m2 * (L1 * sin_theta1 * theta1_dot**2 + L2 * sin_theta2 * theta2_dot**2)
    f2 = (m1 + m2) * g * sin_theta1 + m2 * g * sin_theta2
    f3 = -m2 * L2 * sin_theta2 * theta2_dot**2

    # compute state derivates
    dxdt = np.zeros_like(z)
    dxdt[0] = x_dot
    dxdt[1] = invDet * (d3 * (u - b * x_dot) - d2 * f2 - d4 * f3)
    dxdt[2] = theta1_dot
    dxdt[3] = invDet * (d4 * (u - b * x_dot) - d2 * f2 - d1 * f3)
    dxdt[4] = theta2_dot
    dxdt[5] = invDet * (-d2 * (u - b * x_dot) + d1 * f2 + d4 * f3)

    return dxdt


def loss_dp(self, m, s):
    """defines the loss function used to evaluate the policy's performance"""

    D0 = np.size(s, 1)
    D1 = D0 + 2 * len(self.angle)
    M = m
    S = s

    # compute quadratic weight matrix for cost function
    ell = self.p
    Q = np.dot(np.vstack([1, ell]), np.array([[1, ell]]))
    Q = fill_mat(Q, np.zeros((D1, D1)), [0, D0], [0, D0])
    Q = fill_mat(ell**2, Q, [D0 + 1], [D0 + 1])

    # compute the target state using trigonometric transformation
    target = gaussian_trig(self.target, 0 * s, self.angle)[0]
    target = np.hstack([self.target, target])
    i = np.arange(D0)
    m, s, c = gaussian_trig(M, S, self.angle)
    q = np.dot(S[np.ix_(i, i)], c)
    M = np.hstack([M, m])
    S = np.vstack([np.hstack([S, q]), np.hstack([q.T, s])])

    # iterate through different weightings and accumulate loss
    w = self.width if hasattr(self, "width") else [1]
    L = np.array([0])
    S2 = np.array(0)
    for i in range(len(w)):
        self.z = target
        self.W = Q / w[i]**2
        r, s2, c = self.loss_sat(M, S)
        L = L + r
        S2 = S2 + s2

    return L / len(w)


def draw_rollout(latent):
    """for visualizing rollout of the double pendulum"""
    x0 = latent[:, 0]
    y0 = np.zeros_like(x0)
    x1 = x0 + 0.6 * sin(latent[:, 2])
    y1 = -0.6 * cos(latent[:, 2])
    x2 = x1 + 0.6 * sin(latent[:, 4])
    y2 = y1 - 0.6 * cos(latent[:, 4])

    fig = plt.figure()
    ax = fig.add_subplot(111, autoscale_on=False, xlim=(-2, 2), ylim=(-2, 2))
    ax.set_aspect("equal")
    ax.grid()

    line, = ax.plot([], [], 'o-', lw=2)
    time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

    def animate(i):
        linex = [x0[i], x1[i], x2[i]]
        liney = [y0[i], y1[i], y2[i]]
        line.set_data(linex, liney)
        trail = math.floor(i / (H + 1))
        time_text.set_text("trail %d, time = %.1fs" % (trail, i * dt))
        return line, time_text

    interval = math.ceil(T / dt)
    ani = animation.FuncAnimation(
        fig, animate, np.arange(len(latent)), interval=interval, blit=True)
    ani.save('double_pendulum_test.mp4', fps=20)
    plt.show()

# define state indices used in the system
odei = [0, 1, 2, 3, 4, 5] #ODE input indices
dyno = [0, 1, 2, 3, 4, 5]       #observed state variables for dynamics
angi = [2, 4]       #angular indices
dyni = [0, 1, 2, 3, 4, 5, 6, 7]     #dynamic input indices
poli = [0, 1, 2, 3, 4, 5]       #policy input indices
difi = [0, 1, 2, 3, 4, 5]       #difference indices

# simulation parameters
dt = 0.1
T = 4
H = math.ceil(T / dt)
mu0 = np.array([0, 0, 0, 0, 0, 0])
S0 = np.square(np.diag([0.1] * 6))
N=1
nc = 10

# initialize the plant model of the double pendulum on cart
plant = Empty()
plant.dynamics = dynamics
plant.prop = propagate
plant.noise = np.square(np.diag([1e-2] * 6))
plant.dt = dt
plant.odei = odei
plant.angi = angi
plant.poli = poli
plant.dyno = dyno
plant.dyni = dyni
plant.difi = difi

# initialize the policy model
m, s, c = gaussian_trig(mu0, S0, angi)
m = np.hstack([mu0, m])
c = np.dot(S0, c)
s = np.vstack([np.hstack([S0, c]), np.hstack([c.T, s])])
policy = GPModel()
policy.max_u = [10]
policy.p = {
    'inputs': multivariate_normal(m[poli], s[np.ix_(poli, poli)], nc),
    'targets': 0.1 * randn(nc, len(policy.max_u)),
    'hyp': log([1] * 8 + [1, 0.01])
}

# define cost function
Loss.fcn = loss_dp
cost = Loss()
cost.p = 0.5
cost.gamma = 1
cost.width = [0.25]
cost.angle = plant.angi
cost.target = np.array([0, 0, np.pi, 0, np.pi, 0])

# perform rollout and train the model
start = multivariate_normal(mu0, S0)
x, y, L, latent = rollout(start, policy, plant, cost, H)
policy.fcn = lambda m, s: concat(congp, gaussian_sin, policy, m, s)

for i in range(N):
    dynmodel = GPModel()
    dynmodel.fcn = dynmodel.gp0
    train(dynmodel, plant, policy, x, y)
    result = learn(mu0, S0, dynmodel, policy, plant, cost, H)

    start = multivariate_normal(mu0, S0)
    x_, y_, L, latent_ = rollout(start, policy, plant, cost, H)
    x = np.vstack([x, x_])
    y = np.vstack([y, y_])
    latent = np.vstack([latent, latent_])
    print("Test loss: %s", np.sum(L))

draw_rollout(latent)
