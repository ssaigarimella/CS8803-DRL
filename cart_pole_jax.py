#!/usr/bin/env python3

"""
JAX version of cart_pole.py for GPU acceleration.
"""

import math
import jax
import jax.numpy as jnp
from jax import random, jit
import matplotlib.pyplot as plt
from matplotlib import animation

# JAX-compatible replacements of original PILCO components
from pilco_jax import Empty
from pilco_jax.base import rollout, train, propagate, learn
from pilco_jax.control import congp, concat
from pilco_jax.gp import GPModel
from pilco_jax.loss import Loss
from pilco_jax.util import gaussian_trig, gaussian_sin, fill_mat


@jit
def dynamics(z, t, u):
    g = 9.82
    L = 0.6
    m1 = 0.5
    m2 = 0.5
    b = 0.1

    x, x_dot, theta_dot, theta = z
    u = u.reshape(())  # Fixes JAX tracer concretization error

    d = 4 * (m1 + m2) - 3 * m1 * jnp.cos(theta) ** 2

    dx = x_dot
    dx_dot = (2 * m1 * L * theta_dot**2 * jnp.sin(theta)
              + 3 * m1 * g * jnp.sin(theta) * jnp.cos(theta)
              + 4 * u - 4 * b * x_dot) / d
    dtheta_dot = (-3 * m1 * L * theta_dot**2 * jnp.sin(theta) * jnp.cos(theta)
                  - 6 * (m1 + m2) * g * jnp.sin(theta)
                  - 6 * (u - b * x_dot) * jnp.cos(theta)) / (d * L)
    dtheta = theta_dot

    return jnp.array([dx, dx_dot, dtheta_dot, dtheta])



# Loss function for cart-pole
def loss_cp(self, m, s):
    D0 = s.shape[1]
    D1 = D0 + 2 * len(self.angle)
    M = m
    S = s

    ell = self.p
    Q = jnp.dot(jnp.vstack([1, ell]), jnp.array([[1, ell]]))
    Q = fill_mat(Q, jnp.zeros((D1, D1)), [0, D0], [0, D0])
    Q = fill_mat(ell**2, Q, [D0 + 1], [D0 + 1])

    target = gaussian_trig(self.target, 0 * s, self.angle)[0]
    target = jnp.hstack([self.target, target])

    i = jnp.arange(D0)
    m, s, c = gaussian_trig(M, S, self.angle)
    q = jnp.dot(S[jnp.ix_(i, i)], c)
    M = jnp.hstack([M, m])
    S = jnp.vstack([jnp.hstack([S, q]), jnp.hstack([q.T, s])])

    w = self.width if hasattr(self, "width") else [1]
    L = jnp.array([0.0])
    S2 = jnp.array(0.0)
    for i in range(len(w)):
        self.z = target
        self.W = Q / w[i]**2
        r, s2, c = self.loss_sat(M, S)
        L = L + r
        S2 = S2 + s2

    return L / len(w)


def draw_rollout(latent):
    x0 = latent[:, 0]
    y0 = jnp.zeros_like(x0)
    x1 = x0 + 0.6 * jnp.sin(latent[:, 3])
    y1 = -0.6 * jnp.cos(latent[:, 3])

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
        fig, animate, jnp.arange(len(latent)), interval=interval, blit=True)
    ani.save('cart_pole_test1.mp4', fps=20)
    plt.show()


# ====== System Setup ======
odei = [0, 1, 2, 3]
dyno = [0, 1, 2, 3]
angi = [3]
dyni = [0, 1, 2, 4, 5]
poli = [0, 1, 2, 4, 5]
difi = [0, 1, 2, 3]

dt = 0.1
T = 15
H = math.ceil(T / dt)
mu0 = jnp.array([0.0, 0.0, 0.0, 0.0])
S0 = jnp.square(jnp.diag(jnp.array([0.1, 0.1, 0.1, 0.1])))

N = 3
nc = 10

plant = Empty()
plant.dynamics = dynamics
plant.prop = propagate
plant.noise = jnp.square(jnp.diag(jnp.array([1e-2, 1e-2, 1e-2, 1e-2])))
plant.dt = dt
plant.odei = odei
plant.angi = angi
plant.poli = poli
plant.dyno = dyno
plant.dyni = dyni
plant.difi = difi

m, s, c = gaussian_trig(mu0, S0, angi)
m = jnp.hstack([mu0, m])
c = jnp.dot(S0, c)
s = jnp.vstack([jnp.hstack([S0, c]), jnp.hstack([c.T, s])])

key = random.PRNGKey(0)
policy = GPModel()
policy.max_u = [10]
poli_idx = jnp.array(poli)

policy.p = {
    'inputs': random.multivariate_normal(key, m[poli_idx], s[jnp.ix_(poli_idx, poli_idx)], (nc,)),
    'targets': 0.1 * random.normal(key, (nc, len(policy.max_u))),
    'hyp': jnp.log(jnp.array([1, 1, 1, 0.7, 0.7, 1, 0.01]))
}


Loss.fcn = loss_cp
cost = Loss()
cost.p = 0.5
cost.gamma = 1
cost.width = [0.25]
cost.angle = plant.angi
cost.target = jnp.array([0.0, 0.0, 0.0, jnp.pi])

start = random.multivariate_normal(key, mu0, S0)
x, y, L, latent, key = rollout(start, policy, plant, cost, H, key)


policy.fcn = lambda m, s: concat(congp, gaussian_sin, policy, m, s)

for i in range(N):
    dynmodel = GPModel()
    dynmodel.fcn = dynmodel.gp0
    train(dynmodel, plant, policy, x, y)
    result = learn(mu0, S0, dynmodel, policy, plant, cost, H)

    start = random.multivariate_normal(key, mu0, S0)
    x_, y_, L, latent_ = rollout(start, policy, plant, cost, H)
    x = jnp.vstack([x, x_])
    y = jnp.vstack([y, y_])
    latent = jnp.vstack([latent, latent_])
    print("Test loss: %s" % jnp.sum(L))

with open('cart_pole.json', 'w') as f:
    f.write(str(result))

draw_rollout(latent)
