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
resume = True# resume from previous checkpoint?
render = True
n_classes = 3
# model initialization
D = 80 * 80  # input dimensionality: 80x80 grid
#hyper_param_file = str(int(round(time.time() * 1000))) + ".p"
model_file = "pong_softmax_np.p"
if resume and os.path.isfile(model_file):
    print("loaded model")
    model = pickle.load(open(model_file, 'rb'))
else:
    model = {}
    model['W1'] = np.random.randn(H, D) / np.sqrt(D)  # "Xavier" initialization
    model['W2'] = np.random.randn(H, n_classes) / np.sqrt(H)
grad_buffer = {k: np.zeros_like(v) for k, v in model.iteritems()}  # update buffers that add up gradients over a batch
rmsprop_cache = {k: np.zeros_like(v) for k, v in model.iteritems()}  # rmsprop memory
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))  # sigmoid "squashing" function to interval [0,1]
def softmax(y):
    # maxy = np.amax(y)
    e = np.exp(y)
    return e / np.sum(e)
def prepro(I):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    I = I[35:195]  # crop
    I = I[::2, ::2, 0]  # downsample by factor of 2
    I[I == 144] = 0  # erase background (background type 1)
    I[I == 109] = 0  # erase background (background type 2)
    I[I != 0] = 1  # everything else (paddles, ball) just set to 1
    return I.astype(np.float).ravel()
def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(xrange(0, r.size)):
        if r[t] != 0: running_add = 0  # reset the sum, since this was a game boundary (pong specific!)
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r
def policy_forward(x):
    h = np.dot(model['W1'], x)
    h[h < 0] = 0  # ReLU nonlinearity
    f = np.dot(model['W2'].T, h)
    p = softmax(f)
    return p, h  # return probability of taking action 2, and hidden state
def policy_backward(epx, eph, epdlogp):
    """ backward pass. (eph is array of intermediate hidden states) """
    dW2 = np.dot(eph.T, epdlogp)
    # print "dW2 ",dW2
    # print "shape of epdlogp ",np.shape(epdlogp)
    # print "shape of w2 ",np.shape(model['W2'])
    dh = np.dot(epdlogp, model['W2'].T)
    dh[eph <= 0] = 0  # backpro prelu
    dW1 = np.dot(dh.T, epx)
    return {'W1': dW1, 'W2': dW2}
def copyToHDFS(local_file, hdfs_dir):
    FNULL = open(os.devnull, 'w')
    subprocess.call(["hdfs", "dfs", "-rm", "-r", "-skipTrash", hdfs_dir + "/" + local_file], stdout=FNULL,
                    stderr=FNULL)
    subprocess.call(["hdfs", "dfs", "-copyFromLocal", local_file, hdfs_dir], stdout=FNULL, stderr=FNULL)
env = gym.make("Pong-v0")
observation = env.reset()
prev_x = None  # used in computing the difference frame
xs, hs, dlogps, drs = [], [], [], []
running_reward = None
total_reward_sum = 0
reward_sum = 0
episode_number = 1
wins = 0
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

    if done:
        observation = env.reset()
        total_reward_sum += reward_sum
        if reward_sum>0: wins+=1
        print "episode %f, episode reward: %f, mean: %f, wins: %f"% (episode_number, reward_sum, total_reward_sum/episode_number, wins)
        reward_sum=0
        episode_number+=1
