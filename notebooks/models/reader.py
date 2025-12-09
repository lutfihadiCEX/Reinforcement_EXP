import gymnasium as gym
from stable_baselines3 import PPO


env = gym.make("CartPole-v1", render_mode="human")  
model = PPO.load(r"C:\MLCourse\RLHF_Exploration\notebooks\models\cartpole_ppo.zip") 
obs = env.reset()[0]
done = False

while not done:
    action, _ = model.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    env.render() 

env.close()

