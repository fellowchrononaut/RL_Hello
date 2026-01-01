import gymnasium as gym
from stable_baselines3 import A2C

def random_agent():
    env = gym.make("CartPole-v1", render_mode="human")
    observation, info = env.reset()

    rewards = 0

    episode_over = False

    while not episode_over:
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        rewards += reward
        episode_over = terminated or truncated

    print(f"Total baseline rewards: {rewards}")

    env.close()

def train():
    env = gym.make("CartPole-v1", render_mode=None)
    model = A2C("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=100_000)
    model.save('a2c_cartpole')

def test():
    env = gym.make("CartPole-v1", render_mode="human")
    observation, info = env.reset()

    model = A2C.load('a2c_cartpole')

    rewards = 0

    episode_over = False

    while not episode_over:
        action, _states = model.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, info = env.step(action)

        rewards += reward
        episode_over = terminated or truncated

    print(f"Total trained rewards: {rewards}")

    env.close()



test()

#train()
