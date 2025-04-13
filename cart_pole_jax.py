#!/usr/bin/env python3

"""
A description of parameters and guide to tuning:-

N: Number of iterations/"outer loop iterations". Each iteration adds more data to train the GP, 
    refines the policy using a better model, and leads to compounding improvements.
    
nc: Number of Controller Basis Functions (Policy Complexity). Determines how expressive the controller's GP is;
    Low nc (e.g., 5) -> simple, smooth control law. High nc (e.g., 20–50) -> can model complex, 
    nonlinear policies (needed for swing-up).

policy.p['targets']: Initial Policy Output Variance. Controls the exploration behavior of the initial policy.
                    Higher variance (0.3 * randn(...)) -> more exploratory, covers more of state space; 
                    lower variance -> conservative, may never reach swing-up region to learn from it.

cost.width: Sharpness of the Cost Function. Determines how "tight" the Gaussian cost is around the target state.
            Small width (e.g., 0.1) -> sharply penalizes deviations from upright.
            Large width (e.g., 1.0) -> softens the penalty -> smoother gradient but less focused optimization.

plant.noise: Dynamics Noise Assumption. Covariance matrix of noise assumed during rollouts. Prevents overconfident 
            GP predictions. Introduces regularization during trajectory sampling.If learning becomes unstable 
            (GPs go haywire), try slightly increasing this noise (e.g., 1e-1), or bounding 
            the noise hyperparameter in gp.py.
            
H: Rollout Horizon. Number of timesteps per rollout; If H is too small (e.g., 10 steps), 
    the controller won't get to see the full swing-up dynamics. 
    
maxiter in learn(): Number of Policy Optimization Steps. Controls how long the L-BFGS-B 
                    optimizer is allowed to tune the policy parameters. Low values (e.g., 10) -> 
                    very coarse, may stop before converging. Higher values (50–100) -> 
                    better convergence of the policy search.

"""

import math

import autograd.numpy as np
import matplotlib.pyplot as plt
from autograd.numpy import sin, cos, log
from autograd.numpy.random import randn, multivariate_normal
from matplotlib import animation

# core PILCO functions and modules
from pilco import Empty
from pilco.base import rollout, train, propagate, learn
from pilco.control import congp, concat
from pilco.gp import GPModel
from pilco.loss import Loss
from pilco.util import gaussian_trig, gaussian_sin, fill_mat

# Define the true dynamics for the cart-pole (ODE form)
def dynamics(z, t, u):
    g = 9.82
    L = 0.6 #pendulum length
    m1 = 0.5
    m2 = 0.5
    b = 0.1 #coefficient of friction
    
    # z = [x, x_dot, theta_dot, theta]
    z1, z2, z3 = z[1], z[2], z[3]
    d = 4 * (m1 + m2) - 3 * m1 * cos(z3)**2

    dzdt = np.zeros_like(z)
    dzdt[0] = z1
    dzdt[1] = (2 * m1 * L * z2**2 * sin(z3) + 3 * m1 * g * sin(z3) * cos(z3) +
               4 * u - 4 * b * z1) / d
    dzdt[2] = (-3 * m1 * L * z2**2 * sin(z3) * cos(z3) - 6 *
               (m1 + m2) * g * sin(z3) - 6 * (u - b * z1) * cos(z3)) / (d * L)
    dzdt[3] = z2

    return dzdt

# loss function for cart-pole which penalizes deviation from upright pose
def loss_cp(self, m, s):
    D0 = np.size(s, 1)
    D1 = D0 + 2 * len(self.angle)
    M = m
    S = s
    
    # Build cost matrix Q centered on target
    ell = self.p
    Q = np.dot(np.vstack([1, ell]), np.array([[1, ell]]))
    Q = fill_mat(Q, np.zeros((D1, D1)), [0, D0], [0, D0])
    Q = fill_mat(ell**2, Q, [D0 + 1], [D0 + 1])
    
    # Compute target in trig space
    target = gaussian_trig(self.target, 0 * s, self.angle)[0]
    target = np.hstack([self.target, target])
    
    
    i = np.arange(D0)
    m, s, c = gaussian_trig(M, S, self.angle)
    q = np.dot(S[np.ix_(i, i)], c)
    M = np.hstack([M, m])
    S = np.vstack([np.hstack([S, q]), np.hstack([q.T, s])])
    
    # Evaluate loss as expected distance from target
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

# Draw and save animation of rollout
def draw_rollout(latent):
    x0 = latent[:, 0]
    y0 = np.zeros_like(x0)
    x1 = x0 + 0.6 * sin(latent[:, 3])
    y1 = -0.6 * cos(latent[:, 3])

    fig = plt.figure()
    ax = fig.add_subplot(111, autoscale_on=False, xlim=(-2, 2), ylim=(-2, 2))
    ax.set_aspect("equal")
    ax.grid()

    line, = ax.plot([], [], 'o-', lw=2)
    time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

    def animate(i):
        linex = [x0[i], x1[i]]
        liney = [y0[i], y1[i]]
        line.set_data(linex, liney)
        trial = math.floor(i / (H + 1))
        time_text.set_text("trial %d, time = %.1fs" % (trial, i * dt))
        return line, time_text

    interval = math.ceil(T / dt)
    ani = animation.FuncAnimation(
        fig, animate, np.arange(len(latent)), interval=interval, blit=True)
    ani.save('cart_pole_test1.mp4', fps=20)
    plt.show()


# ====== System Setup ======
odei = [0, 1, 2, 3]  # Indices of observable states
dyno = [0, 1, 2, 3]  # Indices of states to learn
angi = [3]           # Index of angle variable
dyni = [0, 1, 2, 4, 5]  # Inputs to dynamics model
poli = [0, 1, 2, 4, 5]  # Inputs to policy
difi = [0, 1, 2, 3]     # Indices for delta learning

dt = 0.1
T = 15
H = math.ceil(T / dt)   # Horizon
mu0 = np.array([0, 0, 0, 0])        # Initial mean
S0 = np.square(np.diag([0.1, 0.1, 0.1, 0.1]))       # Initial covariance

N =3    # Number of iterations
nc = 10  # Controller basis function count

# Create plant model
plant = Empty()
plant.dynamics = dynamics
plant.prop = propagate
plant.noise = np.square(np.diag([1e-2, 1e-2, 1e-2, 1e-2]))
plant.dt = dt
plant.odei = odei
plant.angi = angi
plant.poli = poli
plant.dyno = dyno
plant.dyni = dyni
plant.difi = difi

# Create initial controller and cost
m, s, c = gaussian_trig(mu0, S0, angi)
m = np.hstack([mu0, m])
c = np.dot(S0, c)
s = np.vstack([np.hstack([S0, c]), np.hstack([c.T, s])])

# Initialize policy model as GP
policy = GPModel()
policy.max_u = [10]
policy.p = {
    'inputs': multivariate_normal(m[poli], s[np.ix_(poli, poli)], nc),
    'targets': 0.1 * randn(nc, len(policy.max_u)),
    'hyp': log([1, 1, 1, 0.7, 0.7, 1, 0.01])
}

# Initialize cost function
Loss.fcn = loss_cp
cost = Loss()
cost.p = 0.5
cost.gamma = 1
cost.width = [0.25]
cost.angle = plant.angi
cost.target = np.array([0, 0, 0, np.pi]) # target: upright pendulum

# Generate initial rollout using random policy
start = multivariate_normal(mu0, S0)
x, y, L, latent = rollout(start, policy, plant, cost, H)

# Wrap policy with controller + saturation
policy.fcn = lambda m, s: concat(congp, gaussian_sin, policy, m, s)

# ====== PILCO Main Loop ======
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


# Save final policy and render rollout
save_file = open('cart_pole.json', 'w')
save_file.write(str(result))
save_file.close()
draw_rollout(latent)
