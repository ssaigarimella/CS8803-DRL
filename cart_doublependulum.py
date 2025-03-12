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

# ---------------------------
# Dynamics for Cart with Double Pendulum
# ---------------------------
def dynamics(z, t, u):
    # Ensure u is a scalar rather than an array:
    if np.ndim(u) > 0:
        u = u[0]

    g  = 9.82
    M  = 1.0    # cart mass
    m1 = 0.5    # mass of first pendulum
    m2 = 0.5    # mass of second pendulum
    L1 = 0.6    # length of first pendulum
    L2 = 0.6    # length of second pendulum
    b  = 0.1    # friction coefficient for cart
    
    # Unpack state
    x, x_dot, theta1_dot, theta1, theta2_dot, theta2 = z

    # Construct the mass matrix
    M11 = M + m1 + m2
    M12 = (m1 + m2) * L1 * cos(theta1)
    M13 = m2 * L2 * cos(theta1 + theta2)
    M21 = M12
    M22 = (m1 + m2) * L1**2
    M23 = m2 * L1 * L2 * cos(theta2)
    M31 = M13
    M32 = M23
    M33 = m2 * L2**2

    M_mat = np.array([[M11, M12, M13],
                      [M21, M22, M23],
                      [M31, M32, M33]])
    
    # Compute the right-hand side terms
    RHS1 = u - b*x_dot + (m1+m2)*L1*sin(theta1)*theta1_dot**2 \
           + m2*L2*sin(theta1+theta2)*(theta1_dot+theta2_dot)**2
    RHS2 = - (m1+m2)*g*L1*sin(theta1)
    RHS3 = - m2*g*L2*sin(theta1+theta2)
    RHS = np.array([RHS1, RHS2, RHS3])
    
    # Solve for accelerations
    q_ddot = np.linalg.solve(M_mat, RHS)
    
    dzdt = np.zeros(6)
    dzdt[0] = x_dot
    dzdt[1] = q_ddot[0]
    dzdt[2] = q_ddot[1]      # theta1_ddot
    dzdt[3] = theta1_dot     # derivative of theta1
    dzdt[4] = q_ddot[2]      # theta2_ddot
    dzdt[5] = theta2_dot     # derivative of theta2
    return dzdt

# ---------------------------
# Loss Function (adapted from cart-pole)
# ---------------------------
def loss_cp(self, m, s):
    D0 = np.size(s, 1)  # base state dimension (here 6)
    D1 = D0 + 2 * len(self.angle)  # after trig augmentation (here 6 + 4 = 10)
    M = m
    S = s

    ell = self.p
    Q = np.dot(np.vstack([1, ell]), np.array([[1, ell]]))
    Q = fill_mat(Q, np.zeros((D1, D1)), [0, D0], [0, D0])
    Q = fill_mat(ell**2, Q, [D0 + 1], [D0 + 1])

    # The target state (raw) is assumed to be of dimension 6.
    target_trig, _ , _ = gaussian_trig(self.target, 0 * s, self.angle)
    target = np.hstack([self.target, target_trig])
    
    i = np.arange(D0)
    m, s, c = gaussian_trig(M, S, self.angle)
    q = np.dot(S[np.ix_(i, i)], c)
    M = np.hstack([M, m])
    S = np.vstack([np.hstack([S, q]), np.hstack([q.T, s])])

    w = self.width if hasattr(self, "width") else [1]
    L_cost = np.array([0.])
    S2 = 0.
    for i in range(len(w)):
        self.z = target
        self.W = Q / w[i]**2
        r, s2, c = self.loss_sat(M, S)
        L_cost = L_cost + r
        S2 = S2 + s2

    return L_cost / len(w)

# ---------------------------
# Visualization: Animate the rollout
# ---------------------------
def draw_rollout(latent):
    # In our state:
    # latent[:,0] = x (cart position)
    # latent[:,3] = theta1 (first pendulum angle)
    # latent[:,5] = theta2 (second pendulum angle)
    # We use the same link lengths as in dynamics:
    L1 = 0.6
    L2 = 0.6

    x0 = latent[:, 0]
    y0 = np.zeros_like(x0)  # cart is on the horizontal axis

    # First pendulum bob (attached to cart):
    x1 = x0 + L1 * sin(latent[:, 3])
    y1 = -L1 * cos(latent[:, 3])
    
    # Second pendulum bob (attached to first bob):
    x2 = x1 + L2 * sin(latent[:, 3] + latent[:, 5])
    y2 = y1 - L2 * cos(latent[:, 3] + latent[:, 5])

    fig = plt.figure()
    ax = fig.add_subplot(111, autoscale_on=False, xlim=(-3, 3), ylim=(-3, 3))
    ax.set_aspect("equal")
    ax.grid()

    # We draw two lines: one from cart to bob1 and one from bob1 to bob2.
    line1, = ax.plot([], [], 'o-', lw=2)
    line2, = ax.plot([], [], 'o-', lw=2)
    time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

    dt = 0.1  # time step (must match plant.dt)
    H = math.ceil(4 / dt)  # horizon (assuming T = 4 seconds)

    def animate(i):
        # Update the first link (cart to first bob)
        linex1 = [x0[i], x1[i]]
        liney1 = [y0[i], y1[i]]
        line1.set_data(linex1, liney1)
        # Update the second link (first bob to second bob)
        linex2 = [x1[i], x2[i]]
        liney2 = [y1[i], y2[i]]
        line2.set_data(linex2, liney2)
        trail = math.floor(i / (H + 1))
        time_text.set_text("trail %d, time = %.1fs" % (trail, i * dt))
        return line1, line2, time_text

    interval = math.ceil(1000 * dt)
    ani = animation.FuncAnimation(
        fig, animate, np.arange(len(latent)), interval=interval, blit=True)
    ani.save('cart_double_pendulum_test.mp4', fps=20)
    plt.show()

# ---------------------------
# PILCO Setup (with updated state indices)
# ---------------------------
# For the base state, we now have 6 dimensions.
dt = 0.1
T  = 4
H  = math.ceil(T / dt)
mu0 = np.array([0, 0, 0, 0, 0, 0])  # initial state: cart at 0; pendulums hanging down (theta=0)
S0  = np.square(np.diag([0.1]*6))

# Augment initial state distribution with trigonometric features for the angles.
# plant.angi are the indices of the angles in the raw state: [3, 5]
m_aug, S_aug, c = gaussian_trig(mu0, S0, [3, 5])
mu0_aug = np.hstack([mu0, m_aug])
S0_aug = np.vstack([np.hstack([S0, c]), np.hstack([c.T, S_aug])])

# The number of PILCO rollouts to simulate:
N = 1
nc = 10

plant = Empty()
plant.dynamics = dynamics
plant.prop = propagate
# Noise in the dynamics (adjusted to 6 dimensions)
plant.noise = np.square(np.diag([1e-2]*6))
plant.dt = dt

# Set index arrays:
# The “raw” state is 6-dimensional. We will augment the state with trigonometric transforms of the angles.
plant.odei  = [0, 1, 2, 3, 4, 5]
plant.dyno  = [0, 1, 2, 3, 4, 5]
# indices of angles in the raw state (theta1 and theta2)
plant.angi  = [3, 5]
# When constructing the GP inputs, we drop the raw angles and use the trig features instead.
# The raw state has 6 dimensions; after gaussian_trig we append 2*len(angi)=4 extra dimensions,
# so the total becomes 10. We select indices (for example) [0,1,2,4,6,7,8,9]
plant.dyni  = [0, 1, 2, 4, 6, 7, 8, 9]
plant.poli  = [0, 1, 2, 4, 6, 7, 8, 9]
plant.difi  = [0, 1, 2, 3, 4, 5]

# Set up the policy (a GP model) using the augmented initial state
policy = GPModel()
policy.max_u = [10]
policy.p = {
    'inputs': multivariate_normal(mu0_aug[plant.poli],
                                    S0_aug[np.ix_(plant.poli, plant.poli)], nc),
    'hyp': log(np.array([[1, 1, 1, 0.7, 0.7, 1, 1, 1, 0.01]])),
    'targets': 0.1 * randn(nc, len(policy.max_u))
}


# Set up the cost. For the double pendulum, we define the target as:
#   [x, x_dot, theta1_dot, theta1, theta2_dot, theta2] = [0, 0, 0, pi, 0, 0]
Loss.fcn = loss_cp
cost = Loss()
cost.p = 0.5
cost.gamma = 1
cost.width = [0.25]
cost.angle = plant.angi
cost.target = np.array([0, 0, 0, np.pi, 0, 0])

# Run one rollout with the initial state distribution.
start = multivariate_normal(mu0, S0)
x, y, L, latent = rollout(start, policy, plant, cost, H)

# Set the policy function (using PILCO’s concat)
policy.fcn = lambda m, s: concat(congp, gaussian_sin, policy, m, s)

# (For demonstration we run one PILCO training iteration.)
for i in range(N):
    dynmodel = GPModel()
    dynmodel.fcn = dynmodel.gp0
    train(dynmodel, plant, policy, x, y)
    # result = learn(mu0, S0, dynmodel, policy, plant, cost, H)
    result = learn(mu0_aug, S0_aug, dynmodel, policy, plant, cost, H)

    start = multivariate_normal(mu0, S0)
    x_, y_, L, latent_ = rollout(start, policy, plant, cost, H)
    x = np.vstack([x, x_])
    y = np.vstack([y, y_])
    latent = np.vstack([latent, latent_])
    print("Test loss: %s" % np.sum(L))

with open('cart_double_pendulum.json', 'w') as save_file:
    save_file.write(str(result))

draw_rollout(latent)
