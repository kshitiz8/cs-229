import numpy as np
import cPickle as pickle
import gym
import os, subprocess
import time
# hyperparameters
H = 600  # number of hidden layer neurons
batch_size = 10  # every how many episodes to do a param update?
learning_rate = 1e-5
gamma = 0.99  # discount factor for reward
decay_rate = 0.98  # decay factor for RMSProp leaky sum of grad^2
render = True
n_classes = 3
# model initialization
D = 80 * 80  # input dimensionality: 80x80 grid
#hyper_param_file = str(int(round(time.time() * 1000))) + ".p"
model_file = "breakout_softmax_np_2.p.38500"
model = pickle.load(open(model_file, 'rb'))

grad_buffer = {k: np.zeros_like(v) for k, v in model.iteritems()}  # update buffers that add up gradients over a batch
rmsprop_cache = {k: np.zeros_like(v) for k, v in model.iteritems()}  # rmsprop memory
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))  # sigmoid "squashing" function to interval [0,1]
def softmax(y):
    # maxy = np.amax(y)
    e = np.exp(y)
    return e / np.sum(e)
def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray.astype(np.int)

def prepro(I):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    I = I[32:192]  # crop
    I = rgb2gray(I)
    I = I[::2, ::2]  # downsample by factor of 2
    I[I != 0] = 1  # everything else (paddles, ball) just set to 1
    return I.astype(np.float).ravel()

def policy_forward(x):
    h = np.dot(model['W1'], x)
    h[h < 0] = 0  # ReLU nonlinearity
    f = np.dot(model['W2'].T, h)
    p = softmax(f)
    return p, h  # return probability of taking action 2, and hidden state


env = gym.make("Breakout-v0")
observation = env.reset()
prev_x = None  # used in computing the difference frame

running_reward = None
total_reward_sum = 0
reward_sum = 0
episode_number = 1
wins = 0
step_diff = 0
min_diff = 10000
while True:
    if render: env.render()
    # preprocess the observation, set input to network to be difference image
    cur_x = prepro(observation)
    x = cur_x - prev_x if prev_x is not None else np.zeros(D)
    prev_x = cur_x
    # forward the policy network and sample an action from the returned probability
    aprob, h = policy_forward(x)
    action = np.argmax( aprob) + 1
    # step the environment and get new measurements
    observation, reward, done, info = env.step(action)
    reward_sum += reward
    if reward == 0:
        step_diff = step_diff + 1
    else:
        if step_diff < min_diff:
            min_diff = step_diff
        print "step_diff: %f , min_diff: %f" % (step_diff,min_diff)
        step_diff = 0

    if done:
        observation = env.reset()
        total_reward_sum += reward_sum
        if reward_sum>0: wins+=1
        print "episode %f, episode reward: %f, mean: %f, wins: %f"% (episode_number, reward_sum, total_reward_sum/episode_number, wins)
        reward_sum=0
        episode_number+=1
