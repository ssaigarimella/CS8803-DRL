import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# Add your project path - modify this path as needed
sys.path.append("/home/alicechan/gt/cs8803drl/PILCO-gpytorch")

# Import your existing modules
from pilco.rewards import ExponentialReward
from pilco.controllers import RbfController, LinearController
from pilco.models import PILCO
from utils import policy

# Set random seed for reproducibility
np.random.seed(0)

# Create video directory
video_dir = "pendulum_videos"
os.makedirs(video_dir, exist_ok=True)

class RecordingPendulum():
    def __init__(self, video_folder=video_dir, video_name_prefix="pendulum_episode"):
        # Create environment with rgb_array rendering for video recording
        self.env = gym.make('InvertedPendulum-v5', render_mode="rgb_array", reset_noise_scale=0.1, frame_skip=5)
        
        # Wrap with video recorder
        self.env = RecordVideo(
            self.env,
            video_folder=video_folder,
            name_prefix=video_name_prefix,
            episode_trigger=lambda e: True  # Record every episode
        )
            
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.reset_called = False

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

def rollout(env, pilco=None, random=False, timesteps=100, SUBS=3):
    """
    Modified rollout function for video recording
    """
    X = []
    Y = []
    x = env.reset()
    
    for t in range(timesteps):
        # Always render for video recording
        env.render()
        
        if random:
            u = env.action_space.sample()
        else:
            u = pilco.compute_action(x[None, :]).cpu().numpy()[0, :]
        
        for i in range(SUBS):
            x_new, r, done, info = env.step(u)
            if done:
                break
                
        if t > 0:
            Y.append(x_new - x)
        X.append(np.hstack((x, u)))
        x = x_new
        
        if done:
            break
            
    return np.array(X), np.array(Y)

def load_trained_pilco(model_path):
    """
    Function to load a trained PILCO model
    """
    # Load model weights from saved file
    pilco_state_dict = torch.load(model_path)
    
    # Example (modify to fit your saved model):
    X_dummy = np.zeros((10, 5))  # Placeholder data
    Y_dummy = np.zeros((10, 4))  # Placeholder data
    
    state_dim = Y_dummy.shape[1]
    control_dim = X_dummy.shape[1] - state_dim
    controller = LinearController(state_dim=state_dim, control_dim=control_dim)
    
    # Create reward function
    R = ExponentialReward(state_dim=state_dim, t=np.array([0.0, 0.0, 0.0, 0.0]))
    
    # Create PILCO instance
    pilco = PILCO(X_dummy, Y_dummy, controller=controller, horizon=40, reward=R)
    
    # Load saved state
    pilco.load_state_dict(pilco_state_dict)
    
    return pilco

def main():
    # Create environment for recording 
    record_env = RecordingPendulum(video_name_prefix="inverted_pendulum_final")
    
    pilco = load_trained_pilco("single_pendulum_pilco_model.pt")
    
    print("Recording trained policy...")
    record_env = RecordingPendulum(video_name_prefix="inverted_pendulum_trained")
    X, Y = rollout(env=record_env, pilco=pilco, timesteps=200)
    record_env.close()
    
    print(f"Videos saved to {video_dir}")

if __name__ == "__main__":
    main()