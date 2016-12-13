__author__ = 'tripathi'
import gym
# env = gym.make('CartPole-v0')
# env.reset()
# for _ in range(10000):
#     env.render()
#     env.step(env.action_space.sample()) # take a random action

env = gym.make('Air-v0')
env.reset()
for _ in range(10000):
    env.render()
    a = env.action_space.sample()
    print env.action_space.sample()
    a,b,c,d = env.step(a) # take a random action
    print c
    if c:
        env.reset()
