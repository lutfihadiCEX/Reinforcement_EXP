import gymnasium as gym
from stable_baselines3 import PPO

# Load environment and trained model
env = gym.make("CartPole-v1", render_mode="human")  # 'human' enables visual window
model = PPO.load(r"C:\MLCourse\RLHF_Exploration\notebooks\models\cartpole_ppo.zip")  # Load your saved agent

obs = env.reset()[0]
done = False

while not done:
    action, _ = model.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    env.render()  # Shows the window with CartPole animation

env.close()
