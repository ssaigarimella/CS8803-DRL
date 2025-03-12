"""
unicycle.py

This script implements a 5-DoF unicycle riding task for PILCO.
The unicycle is 0.76 m high and consists of a 1 kg wheel, a
23.5 kg frame, and a 10 kg flywheel (mounted perpendicularly).
Two torques are applied:
  • u_w (|u_w| ≤ 10 Nm) is applied to the wheel and produces
    longitudinal and pitch (forward/backward tilt) accelerations.
  • u_t (|u_t| ≤ 50 Nm) is applied to the flywheel to generate a
    reaction torque that helps keep the unicycle laterally stable.
The dynamics are modeled by 12 coupled first-order ODEs.
The balancing task is solved using a linear controller
of the form:
    π(x,θ) = A*x_aug + b,
where the feature vector is the 12-state vector augmented by
sin(yaw) (i.e., 13 dimensions) and the total parameter count is 28.
The cost is defined to penalize deviation from the upright state.
"""

import math
import autograd.numpy as np
from autograd.numpy import sin, cos, log
from autograd.numpy.random import randn, multivariate_normal
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib import animation

from pilco import Empty
from pilco.base import rollout, propagate, learn, train
from pilco.loss import Loss
from pilco.util import gaussian_trig, fill_mat

# ---------------------------
# Unicycle Dynamics Function
# ---------------------------
def unicycle_dynamics(z, t, u):
    """
    z: 12-dim state vector
         [0]  x position
         [1]  x velocity
         [2]  y position
         [3]  y velocity
         [4]  pitch angle (forward tilt)
         [5]  pitch rate
         [6]  roll angle (side tilt)
         [7]  roll rate
         [8]  yaw angle (heading)
         [9]  yaw rate
         [10] flywheel angle
         [11] flywheel angular rate
    u: 2-dim control vector [u_w, u_t]
         u_w: torque on wheel (|u_w| ≤ 10 Nm)
         u_t: torque on flywheel (|u_t| ≤ 50 Nm)
    """
    g = 9.81         # gravitational acceleration
    M_total = 34.5   # total mass (1 + 23.5 + 10 kg)
    c1 = 0.1         # translational damping coefficient
    I_pitch = 3.4    # effective moment of inertia for pitch
    I_roll  = 3.4    # effective moment of inertia for roll
    h = 0.76         # unicycle height
    c2 = 0.1         # angular damping for pitch/roll (frame)
    c3 = 0.5         # yaw damping coefficient
    I_flywheel = 0.1 # flywheel moment of inertia (approx.)
    c4 = 0.05        # damping for flywheel
    k_r = 0.5        # coupling constant from flywheel acceleration to roll

    # Unpack state variables
    x       = z[0]
    x_dot   = z[1]
    y       = z[2]
    y_dot   = z[3]
    pitch   = z[4]
    pitch_dot = z[5]
    roll    = z[6]
    roll_dot  = z[7]
    yaw     = z[8]
    yaw_dot = z[9]
    flywheel    = z[10]
    flywheel_dot = z[11]

    # Unpack controls
    u_w = u[0]
    u_t = u[1]

    # Compute flywheel angular acceleration
    flywheel_ddot = (u_t - c4 * flywheel_dot) / I_flywheel

    # Initialize derivative vector
    dzdt = np.zeros_like(z)
    dzdt[0] = x_dot
    dzdt[1] = -g * pitch - c1 * x_dot + u_w / M_total
    dzdt[2] = y_dot
    dzdt[3] = -g * roll - c1 * y_dot
    dzdt[4] = pitch_dot
    dzdt[5] = (u_w - c2 * pitch_dot) / I_pitch - (g / h) * pitch
    dzdt[6] = roll_dot
    dzdt[7] = - (c2 / I_roll) * roll_dot - (g / h) * roll + k_r * flywheel_ddot
    dzdt[8] = yaw_dot
    dzdt[9] = -c3 * yaw_dot
    dzdt[10] = flywheel_dot
    dzdt[11] = flywheel_ddot

    return dzdt

# ---------------------------
# Linear Controller Function
# ---------------------------
def linear_policy(x, S):
    """
    Implements a linear controller:
      u = A * x_aug + b
    where the 13-dim feature vector x_aug is constructed by
    concatenating the 12-dimensional state x with sin(yaw) (yaw is x[8]).
    
    This controller has 28 parameters (A is 2x13 and b is 2).
    The outputs are clamped to the torque limits:
      |u_w| <= 10, |u_t| <= 50.
    
    The second argument S (input covariance) is not used.
    """
    # x is expected to be a 1D (or 2D) array with 12 elements.
    if x.ndim == 1:
        x_aug = np.concatenate([x, np.sin(x[8:9])])
    else:
        x_aug = np.concatenate([x, np.sin(x[:, 8:9])], axis=1)
    # theta is a 28-dim vector: first 26 for A (reshaped to 2x13), last 2 for b.
    theta = linear_policy.theta  # shape (28,)
    A = theta[:26].reshape(2, 13)
    b = theta[26:].reshape(2)
    u = A.dot(x_aug) + b
    # Clamp the control signals to the allowable torques
    u[0] = np.clip(u[0], -10, 10)
    u[1] = np.clip(u[1], -50, 50)
    return u, None, None

# Initialize controller parameters randomly
linear_policy.theta = 0.1 * randn(28)

# ---------------------------
# Simulation Parameters
# ---------------------------
dt = 0.1
T = 4
H = int(math.ceil(T / dt))
mu0 = np.zeros(12)  # initial state: assume at rest and upright
# Use a small diagonal covariance (adjust as needed)
S0 = np.square(np.diag([0.1] * 12))
N = 1    # number of rollouts
nc = 10  # (not used here but kept for consistency)

# ---------------------------
# Set Up the Unicycle Plant
# ---------------------------
plant = Empty()
plant.dynamics = unicycle_dynamics
plant.prop = propagate
plant.noise = np.square(np.diag([1e-2] * 12))
plant.dt = dt
plant.odei = list(range(12))
plant.angi = [4, 6, 8, 10]    # Use these indices for angle variables
plant.poli = list(range(12))
plant.dyno = list(range(12))
plant.dyni = list(range(12)) + list(range(12, 12 + 2))
plant.difi = list(range(12))

# ---------------------------
# Set Up the Controller (Policy)
# ---------------------------
# Here we use our linear controller rather than a GP model.
policy = Empty()
policy.fcn = linear_policy
policy.max_u = [10, 50]  # max torques for u_w and u_t

# ---------------------------
# Set Up the Cost Function
# ---------------------------
cost = Loss()
cost.p = 0.5
cost.gamma = 1
cost.width = [0.25]
cost.angle = plant.angi     # Ensure cost uses the same angle indices
cost.target = np.zeros(12)

# ---------------------------
# Run a Single Rollout
# # ---------------------------
# start = multivariate_normal(mu0, S0)
# x, y, L, latent = rollout(start, policy, plant, cost, H)

# print("Final state:", x[-1])
# print("Total cost:", L.sum())

# ---------------------------
# Main Learning Loop
# ---------------------------
from pilco.gp import GPModel
from pilco.base import train, learn

# Initialize GP model for dynamics learning
gpmodel = GPModel()

# Data storage for rollouts
data_x = []
data_y = []

# Collect a few initial random rollouts to gather initial data (e.g., 5 trials)
num_initial_trials = 5
for trial in range(num_initial_trials):
    start = multivariate_normal(mu0, S0)
    x, y, L, latent = rollout(start, policy, plant, cost, H)
    data_x.append(x)
    data_y.append(y)
    print("Initial trial", trial, "cost:", L.sum())

# Combine the initial rollout data
X_data = np.vstack(data_x)
Y_data = np.vstack(data_y)

# Main iterative learning loop
num_iterations = 20  # Set the desired number of learning iterations
for it in range(num_iterations):
    print("Learning iteration", it)
    # Train the GP dynamics model using all collected data.
    train(gpmodel, plant, policy, X_data, Y_data)
    
    # Improve the policy using the learn function (which optimizes policy parameters).
    policy = learn(mu0, S0, gpmodel, policy, plant, cost, H)
    
    # Evaluate the improved policy in a new rollout.
    start = multivariate_normal(mu0, S0)
    x, y, L, latent = rollout(start, policy, plant, cost, H)
    print("Iteration", it, "rollout cost:", L.sum())
    
    # Append the new rollout data to the dataset for further GP training.
    X_data = np.vstack([X_data, x])
    Y_data = np.vstack([Y_data, y])


# ---------------------------
# Visualization Function
# ---------------------------
def draw_rollout(latent):
    """
    A simple visualization that plots the evolution of the pitch and roll angles.
    """
    t = np.linspace(0, T, H + 1)
    pitch = latent[:, 4]  # pitch angle (forward tilt)
    roll = latent[:, 6]   # roll angle (side tilt)
    plt.figure()
    plt.plot(t, pitch, label="Pitch")
    plt.plot(t, roll, label="Roll")
    plt.xlabel("Time (s)")
    plt.ylabel("Angle (rad)")
    plt.legend()
    plt.title("Unicycle Rollout: Pitch and Roll")
    plt.show()

draw_rollout(latent)
