import math
import autograd.numpy as np
from autograd.numpy.random import multivariate_normal, randn
import matplotlib.pyplot as plt
from matplotlib import animation

from pilco import Empty
from pilco.base import rollout, train, propagate, learn
from pilco.control import congp, concat
from pilco.gp import GPModel
from pilco.loss import Loss
from pilco.util import gaussian_trig, gaussian_sin, fill_mat

# ------------------------------------------------------------------------------
# 1. Unicycle-specific settings
# ------------------------------------------------------------------------------
# Define state indices
odei = [0, 1, 2, 3]    # indices of the ODE states: [x, y, theta, v]
dyno = [0, 1, 2, 3]    # indices used for the dynamics model output
angi = [2]             # index for angular state (theta)
poli = [0, 1, 2, 3]    # policy input indices
difi = [0, 1, 2, 3]    # difference indices

# Simulation parameters
dt = 0.1
T = 4.0
H = int(np.ceil(T / dt))

# Initial state distribution
mu0 = np.array([0.0, 0.0, 0.0, 0.0])
S0 = np.diag([0.1, 0.1, 0.05, 0.1])**2

# Number of initial random rollouts and controlled learning trials
J = 1   # number of initial random rollouts
N = 1  # number of controlled learning iterations
nc = 10 # number of pseudo-inputs for the GP policy

# Unicycle dynamics
def dynamics(z, t, u):
    # NOTE BY SSG: Assume a simple unicycle model:
    # z = [x, y, theta, v]
    # u = [a, omega] with:
    #   a: acceleration (affecting v)
    #   omega: angular velocity (affecting theta)
    x, y, theta, v = z
    a = u[0]
    omega = u[1]
    
    dzdt = np.zeros_like(z)
    dzdt[0] = v * np.cos(theta)
    dzdt[1] = v * np.sin(theta)
    dzdt[2] = omega
    dzdt[3] = a
    return dzdt

def augment(state):
    # simply return a scalar feature (or vector) to append.
    theta = state[2]
    return np.array([np.sin(theta), np.cos(theta)])

# Define a modified rollout that optionally augments the state (saw this in the original matlab code)
def unicycle_rollout(start, policy, plant, cost, H):
    odei = plant.odei
    poli = plant.poli
    dyno = plant.dyno
    return rollout(start, policy, plant, cost, H)

# ------------------------------------------------------------------------------
# 2. Set up the plant, policy, and cost function for the unicycle
# ------------------------------------------------------------------------------
plant = Empty()
plant.dynamics = dynamics
plant.prop = propagate
plant.noise = np.diag([1e-2, 1e-2, 1e-2, 1e-2])
plant.dt = dt

# Set the indices for the plant 
plant.odei = odei
plant.dyno = dyno
plant.poli = poli
plant.dyni = list(range(len(mu0)))  
plant.difi = difi
plant.angi = angi  
plant.augment = augment

# Policy model (using a GP)
m0, s0, c0 = gaussian_trig(mu0, S0, [2])  # using theta as angular feature, for example
m_policy = np.hstack([mu0, m0])
c_policy = np.dot(S0, c0)
s_policy = np.vstack([np.hstack([S0, c_policy]), np.hstack([c_policy.T, s0])])
policy = GPModel()
policy.max_u = [1.0, 0.5]  # set maximum control limits for [a, omega]
policy.p = {
    'inputs': multivariate_normal(m_policy[poli], s_policy[np.ix_(poli, poli)], nc),
    'targets': 0.1 * randn(nc, len(policy.max_u))
}
# Set hyperparameters with correct shape: 2 outputs, each with 6 hyperparameters.
hyp_initial = np.log([1, 1, 1, 0.7, 1, 0.01])  # for D=4 --> D+2=6
policy.p['hyp'] = np.tile(hyp_initial, (len(policy.max_u), 1))

# set the control function for the policy; might need to be changed but this works for now
policy.fcn = lambda m, s: concat(congp, gaussian_sin, policy, m, s)

# for unicycle, the target is to reach a desired state (e.g., [x_target, y_target, theta_target, v_target]).
# This is like following a point that is moving in front of the unicycle.
def loss_unicycle(self, m, s):
    D0 = np.size(s, 1)
    D1 = D0 + 2  # augment with sin(theta) and cos(theta)
    M = m
    S = s
    
    ell = self.p  # weighting parameter (you may want to set this differently)
    Q = np.dot(np.vstack([1, ell]), np.array([[1, ell]]))
    Q = fill_mat(Q, np.zeros((D1, D1)), [0, D0], [0, D0])
    Q = fill_mat(ell**2, Q, [D0+1], [D0+1])
    
    # compute the target transformation (using gaussian_trig for the angular part)
    target = gaussian_trig(self.target, np.zeros((D0, D0)), [2])[0]
    target = np.hstack([self.target, target])
    i = np.arange(D0)
    m_aug, s_aug, c = gaussian_trig(M, S, [2])
    q = np.dot(S[np.ix_(i, i)], c)
    M = np.hstack([M, m_aug])
    S = np.vstack([np.hstack([S, q]), np.hstack([q.T, s_aug])])
    
    w = self.width if hasattr(self, "width") else [1]
    L_val = np.array([0])
    S2 = 0
    for i in range(len(w)):
        self.z = target
        self.W = Q / w[i]**2
        r, s2, c_val = self.loss_sat(M, S)
        L_val = L_val + r
        S2 = S2 + s2
    return L_val / len(w)

Loss.fcn = loss_unicycle  # assign custom loss function

cost = Loss()
cost.p = 0.5
cost.gamma = 1.0
cost.width = [0.25]
# Set the target state for the unicycle (for example, aim to reach x=1, y=1, theta=0, v=0)
cost.target = np.array([1.0, 1.0, 0.0, 0.0])

# ------------------------------------------------------------------------------
# 3. Initial random rollouts
# ------------------------------------------------------------------------------
# Collect initial data from J random rollouts
x_data = None
y_data = None
latents = []

for jj in range(J):
    # Use a temporary policy with reduced control authority
    temp_policy = Empty()
    temp_policy.max_u = np.array(policy.max_u) / 5.0
    # use the same rollout function, can extend it to update augmented state if needed
    x_roll, y_roll, realCost, latent_roll = rollout(multivariate_normal(mu0, S0), temp_policy, plant, cost, H)
    x_data = x_roll if x_data is None else np.vstack([x_data, x_roll])
    y_data = y_roll if y_data is None else np.vstack([y_data, y_roll])
    latents.append(latent_roll)
    
    # optional: visualize this rollout (you'll see the full rollout at the end anyways)
    # draw_rollout_unicycle(latent_roll)
    print(f"Initial rollout {jj+1} complete.")

# Compute an augmented initial state distribution if needed (mimicking the MATLAB MCMC augmentation)
z_samples = multivariate_normal(mu0, S0, 1000).T  # shape: (state_dim, 1000)
# If augmentation is defined, augment each sample (here we assume plant.augment returns a vector)
augmented = np.array([plant.augment(z_samples[:, i]) for i in range(z_samples.shape[1])]).T
# Concatenate: new state = [original; augmented]
z_aug = np.vstack([z_samples, augmented])
mu0Sim = np.mean(z_aug, axis=1)
S0Sim = np.cov(z_aug)
# Replace known indices (here odei) with mu0, S0 if needed:
mu0Sim[odei] = mu0
for i, idx in enumerate(odei):
    S0Sim[idx, idx] = S0[i, i]
# For dynamics learning, use only the observed state indices:
mu0Sim = mu0Sim[dyno]
S0Sim = S0Sim[np.ix_(dyno, dyno)]

# ------------------------------------------------------------------------------
# 4. Controlled learning loop
# ------------------------------------------------------------------------------
for j in range(N):
    # Train the dynamics model using collected data
    dynmodel = GPModel()
    dynmodel.fcn = dynmodel.gp0
    train(dynmodel, plant, policy, x_data, y_data)
    
    # Policy improvement (learnPolicy in MATLAB)
    result = learn(mu0, S0, dynmodel, policy, plant, cost, H)
    
    # Apply the updated controller to get a new rollout (applyController in MATLAB)
    x_new, y_new, L_new, latent_new = rollout(multivariate_normal(mu0, S0), policy, plant, cost, H)
    
    # Augment the collected data
    x_data = np.vstack([x_data, x_new])
    y_data = np.vstack([y_data, y_new])
    latents.append(latent_new)
    
    print(f"Controlled trial #{j+1}, test loss: {np.sum(L_new)}")
    
#save the final policy result to a JSON file
with open('unicycle_policy.json', 'w') as f:
    f.write(str(result))

#visualize the final rollout (TODO implement draw_rollout_unicycle similarly to draw_rollout in cart_pole.py)
def draw_rollout_unicycle(latent):
    x = latent[:, 0]
    y = latent[:, 1]
    theta = latent[:, 2]
    
    # Compute a representation of the unicycle (e.g., the position and a point in the direction of heading)
    x_dir = x + 0.5 * np.cos(theta)
    y_dir = y + 0.5 * np.sin(theta)
    
    fig, ax = plt.subplots()
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_aspect('equal')
    ax.grid(True)
    line, = ax.plot([], [], 'o-', lw=2)
    time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)
    
    def animate(i):
        line.set_data([x[i], x_dir[i]], [y[i], y_dir[i]])
        time_text.set_text(f"Time = {i*dt:.1f}s")
        return line, time_text
    
    ani = animation.FuncAnimation(fig, animate, frames=len(latent), interval=100, blit=True)
    plt.show()

draw_rollout_unicycle(latents[-1])
