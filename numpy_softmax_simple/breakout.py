spark = True
if spark:
    from pyspark.context import SparkContext
    from pyspark.conf import SparkConf

    sc = SparkContext(conf=SparkConf().setAppName("breakout_tf"))


def map_fun(worker_num):
    """ Trains an agent with (stochastic) Policy Gradients on Pong. Uses OpenAI Gym. """
    import numpy as np
    import cPickle as pickle
    import gym
    import os, subprocess
    import time

    # hyperparameters


    H = 600  # number of hidden layer neurons
    batch_size = 10  # every how many episodes to do a param update?
    learning_rate = 0.5e-3
    gamma = 0.99  # discount factor for reward
    decay_rate = 0.95  # decay factor for RMSProp leaky sum of grad^2
    resume = False  # resume from previous checkpoint?
    render = False
    n_classes = 3

    # model initialization
    D = 80 * 80  # input dimensionality: 80x80 grid


    #hyper_param_file = str(int(round(time.time() * 1000))) + ".p"

    model_file = "breakout_softmax_np.p"
    if resume and os.path.isfile(model_file):
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


    def discount_rewards(r):
        """ take 1D float array of rewards and compute discounted reward """
        discounted_r = np.zeros_like(r)
        running_add = 0
        prev_reward_t = r.size + 100
        for t in reversed(xrange(0, r.size)):
            if r[t] != 0 and prev_reward_t - t > 30:
                running_add = 0  # reset the sum, since this was a game boundary (Breakout specific!)
            if r[t] != 0:
                prev_reward_t = t
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

    def copyToHDFS(episode_no, local_file, hdfs_dir):
        filename = local_file+"."+str(episode_no)
        FNULL = open(os.devnull, 'w')
        subprocess.call(["hdfs", "dfs", "-rm", "-r", "-skipTrash", hdfs_dir + "/" + filename], stdout=FNULL,
                        stderr=FNULL)
        subprocess.call(["hdfs", "dfs", "-copyFromLocal", local_file, hdfs_dir + "/" + filename], stdout=FNULL, stderr=FNULL)

    env = gym.make("Breakout-v0")
    observation = env.reset()
    prev_x = None  # used in computing the difference frame
    xs, hs, dlogps, drs = [], [], [], []
    running_reward = None
    reward_sum = 0
    episode_number = 1
    while True:
        if render: env.render()

        # preprocess the observation, set input to network to be difference image
        cur_x = prepro(observation)
        x = cur_x - prev_x if prev_x is not None else np.zeros(D)
        prev_x = cur_x

        # forward the policy network and sample an action from the returned probability
        aprob, h = policy_forward(x)

        # if np.random.uniform() > np.max(aprob):
        #     idx = np.random.randint(0, 2)
        #     y = np.zeros((3), dtype=int)
        #     y[idx] = 1
        # else:
        y = np.random.multinomial(1, aprob)



        # print np.argmax(y)+1
        # record various intermediates (needed later for backprop)
        xs.append(x)  # observation
        hs.append(h)  # hidden state
        action = np.argmax(y)+1

        dlogps.append(y - aprob)  # grad that encourages the action that was taken to be taken (see http://cs231n.github.io/neural-networks-2/#losses if confused)

        # step the environment and get new measurements
        observation, reward, done, info = env.step(action)
        reward_sum += reward

        drs.append(reward)  # record reward (has to be done after we call step() to get reward for previous action)

        if done:  # an episode finished
            episode_number += 1

            # stack together all inputs, hidden states, action gradients, and rewards for this episode
            epx = np.vstack(xs)
            eph = np.vstack(hs)
            epdlogp = np.vstack(dlogps)
            epr = np.vstack(drs)
            xs, hs, dlogps, drs = [], [], [], []  # reset array memory

            # compute the discounted reward backwards through time
            discounted_epr = discount_rewards(epr)
            # standardize the rewards to be unit normal (helps control the gradient estimator variance)
            discounted_epr -= np.mean(discounted_epr)
            std = np.std(discounted_epr)
            if std != 0:
                discounted_epr /= std

            epdlogp *= discounted_epr  # modulate the gradient with advantage (PG magic happens right here.)
            grad = policy_backward(epx, eph, epdlogp)
            for k in model: grad_buffer[k] += grad[k]  # accumulate grad over batch

            # perform rmsprop parameter update every batch_size episodes
            if episode_number % batch_size == 0:
                for k, v in model.iteritems():
                    g = grad_buffer[k]  # gradient
                    rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 - decay_rate) * g ** 2
                    model[k] += learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
                    grad_buffer[k] = np.zeros_like(v)  # reset batch gradient buffer

            # boring book-keeping
            running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
            print 'resetting env. episode %f reward total was %f. running mean: %f' % (episode_number, reward_sum, running_reward)

            if (episode_number) % 500 == 0:
                pickle.dump(model, open(model_file, 'wb'))
                copyToHDFS(episode_number, model_file, "/user/tripathi/checkpoint/")

            reward_sum = 0
            observation = env.reset()  # reset env
            prev_x = None




if spark:
    numExecutors = int(sc._conf.get("spark.executor.instances"))
    rdd = sc.parallelize(range(numExecutors), numExecutors)
    rdd.map(map_fun).collect()
else:
    map_fun(1)




        # if reward != 0:  # Pong has either +1 or -1 reward exactly when game ends.
        #     print ('ep %d: game finished, reward: %f' % (episode_number, reward)) + ('' if reward == -1 else ' !!!!!!!!')
