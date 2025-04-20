import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import torch
import matplotlib.pyplot as plt
from pilco.rewards import ExponentialReward
from pilco.controllers import RbfController, LinearController
from pilco.models import PILCO
from utils import rollout, policy
import numpy as np
import sys
import time
import os
sys.path.append("/home/alicechan/gt/cs8803drl/PILCO-gpytorch")
np.random.seed(0)

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Training on GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    device = torch.device("cpu")
    print("CUDA not available. Training on CPU.")

class myPendulum():
    def __init__(self, render_mode="rgb_array", video_folder="training_videos"):
        self.env = gym.make('InvertedPendulum-v5', render_mode=render_mode, reset_noise_scale=0.1, frame_skip=5)
        self.env = RecordVideo(self.env, video_folder=video_folder, episode_trigger=lambda x: True, name_prefix="pendulum_training", video_length=200)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.reset_called = False
        self.render_mode = render_mode

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return obs, reward, done, info

    def reset(self):
        obs, _ = self.env.reset()
        self.reset_called = True
        return obs

    def render(self):
        if not self.reset_called:
            self.reset()
        return self.env.render()

    def close(self):
        self.env.close()

    def enable_recording_mode(self):
        self.recording_mode = True
        
    def disable_recording_mode(self):
        self.recording_mode = False

os.makedirs("training_videos", exist_ok=True)
env = myPendulum(render_mode="rgb_array", video_folder="training_videos")
# env = gym.make('CartPole-v0')
# Initial random rollouts to generate a dataset
# X, Y = rollout(env=env, pilco=None, random=True, timesteps=100)
'''
increase SUBS: increase robustness, reduce aliasing and get smoother trajectories.
'''
X, Y = rollout(env=env, pilco=None, random=True, timesteps=100, render=False, SUBS=3)
'''
change the range to higher number for longer training
'''
for i in range(1, 100):
    X_, Y_ = rollout(env=env, pilco=None, random=True,  timesteps=100)
    X = np.vstack((X, X_))
    Y = np.vstack((Y, Y_))


# states: dot_posistion [-1, 1], dot_velocity[-inf,inf], sin_theta[-1,1], cos_theta[-1,1], theta_velocity[-inf,inf]
state_dim = Y.shape[1]
control_dim = X.shape[1] - state_dim
# controller1 = RbfController(state_dim=state_dim, control_dim=control_dim, num_basis_functions=5)
controller = LinearController(state_dim=state_dim, control_dim=control_dim)

# pilco = PILCO(X, Y, controller1=controller1, horizon=40)
# Example of user provided reward function, setting a custom target state
# R = ExponentialReward(state_dim=state_dim,
#                       t=np.array([0.0, 0.0, 1.0, 0.0, 0.0]))
R = ExponentialReward(
    state_dim=state_dim,
    t=np.array([0.0, 0.0, 0.0, 0.0]) # x, dx, theta, d_theta
)
m_init = np.reshape(env.reset(), (1, state_dim))
S_init = np.diag([0.01] * state_dim)
m_init = torch.from_numpy(m_init).float().cuda()
S_init = torch.from_numpy(S_init).float().cuda()

pilco = PILCO(X, Y, controller=controller, horizon=40,
              reward=R, m_init=m_init, S_init=S_init)

# Example of fixing a parameter, optional, for a linear controller1 only
# pilco.controller.b = np.array([[0.0]])
# pilco.controller.b.trainable = False
T = 30

# Lists to store data for one final plot intead
all_m_p = []
all_actual = []

# start the timer
start_time = time.time()

for rollouts in range(20):
    pilco.optimize_models()
    pilco.optimize_policy()

    X_new, Y_new = rollout(env=env, pilco=pilco, timesteps=100, render=True)

    # multi-step prediction
    m_p = np.zeros((T, state_dim))
    S_p = np.zeros((T, state_dim, state_dim))
    for h in range(T):
        m_h, S_h, _ = pilco.predict(m_init, S_init, h)
        m_p[h,:], S_p[h,:,:] = m_h[0,:].detach().cpu().numpy(), S_h[:,:].detach().cpu().numpy()

    all_m_p.append(m_p.copy())
    all_actual.append(X_new[1:T, :].copy())

    print("=======================================")
    print(f"Iteration {rollouts+1} done")
    print("=======================================")
    
    # Update dataset
    X = np.vstack((X, X_new)); Y = np.vstack((Y, Y_new))
    pilco.mgpr.set_XY(X, Y)


def record_final_model(env, pilco, num_episodes=5, timesteps=200):
    """Record videos of the final trained model, making sure it doesn't terminate early"""
    os.makedirs("final_model_videos", exist_ok=True)
    final_env = myPendulum(render_mode="rgb_array", video_folder="final_model_videos")
    
    for episode in range(num_episodes):
        print(f"Recording episode {episode+1}/{num_episodes}...")
        final_env.enable_recording_mode()  # Enable recording mode
        x = final_env.reset()
        
        for t in range(timesteps):
            u = policy(final_env, pilco, x, random=False)
            x_new, _, done, _ = final_env.step(u)
            final_env.render()
            x = x_new
        
        final_env.disable_recording_mode()  # Disable after recording
        
    final_env.close()
    print(f"Recorded {num_episodes} episodes in 'final_model_videos' folder")

# Save the trained model
model_save_path = "single_pendulum_pilco_model.pt"
torch.save(pilco.state_dict(), model_save_path)
print(f"Saved PILCO model to: {model_save_path}")

# Calculate total training time
end_time = time.time()
training_time = end_time - start_time
hours, remainder = divmod(training_time, 3600)
minutes, seconds = divmod(remainder, 60)
print(f"Total training time: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d} (HH:MM:SS)")
print(f"Total training time in seconds: {training_time:.2f}s")

# Create final comprehensive plot after all training iterations
fig, axes = plt.subplots(state_dim, 1, figsize=(10, 12), sharex=True)
state_labels = ['x position', 'x velocity', 'theta', 'angular velocity']

# Plot data from final iteration
final_m_p = all_m_p[-1]
final_actual = all_actual[-1]
final_S_p = S_p  # Uncertainty from final iteration

# Ensure we're plotting with consistent dimensions
actual_length = final_actual.shape[0]
plot_length = min(T-1, actual_length)

for i in range(state_dim):
    axes[i].plot(range(plot_length), final_m_p[0:plot_length, i], 'b-', label='Predicted')
    axes[i].plot(range(plot_length), final_actual[:plot_length, i], 'r--', label='Actual')
    axes[i].fill_between(range(plot_length),
                       final_m_p[0:plot_length, i] - 2*np.sqrt(final_S_p[0:plot_length, i, i]),
                       final_m_p[0:plot_length, i] + 2*np.sqrt(final_S_p[0:plot_length, i, i]), 
                       color='blue', alpha=0.2, label='95% Confidence')
    axes[i].set_ylabel(state_labels[i])
    axes[i].grid(True)
    if i == 0:
        axes[i].legend(loc='best')

axes[-1].set_xlabel('Time Steps')
fig.suptitle('PILCO Predictions vs Actual States (Final Iteration)', fontsize=16)
plt.tight_layout()
plt.savefig('pilco_pred_VS_actual.png', dpi=300)
# plt.show()

record_final_model(env, pilco)

env.close()