import tensorflow as tf
import gym
import numpy as np
import matplotlib.pyplot as plt
import random

class PolicyGradient:
    def __init__(self):
        self.D = 6400 # number of hidden layer neurons
        self.H = 200 # number of hidden layer neurons
        self.n_classes=3
        self.batch_size = 10 # every how many episodes to do a param update?
        self.learning_rate = 1e-3
        self.gamma = 0.90 # discount factor for reward
        self.decay_rate = 0.99 # decay factor for RMSProp leaky sum of grad^2
        self.resume = True # resume from previous checkpoint?
        self.render = True

        self.state_buffer  = []
        self.reward_buffer = []
        self.action_buffer = []

        self.all_rewards = []
        self.max_reward_length = 1000000

        self.states = tf.placeholder(tf.float32, [1,self.D], name="states")
        self.policy_forward, self.action_prob = PolicyGradient.policy_forward(self.states, [self.D, self.H],[self.H, self.n_classes])

        self.actions_taken = tf.placeholder(tf.int32, (None,), name="taken_actions")
        self.discounted_rewards = tf.placeholder(tf.float32, (None,), name="discounted_rewards")
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate, decay=self.decay_rate)
        self.train_op = PolicyGradient.rmsprop(policy_forward=self.policy_forward,
                                               actions_taken=self.actions_taken,
                                               discounted_rewards=self.discounted_rewards,
                                               optimizer=self.optimizer)


    @staticmethod
    def pre_process(I):
        """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
        I = I[35:195] # crop
        I = I[::2,::2,0] # downsample by factor of 2
        I[I == 144] = 0 # erase background (background type 1)
        I[I == 109] = 0 # erase background (background type 2)
        I[I != 0] = 1 # everything else (paddles, ball) just set to 1
        return I.astype(np.float).ravel()

    @staticmethod
    def rmsprop(policy_forward, actions_taken, discounted_rewards, optimizer):

        # # compute policy loss and regularization loss
        cross_entropy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(policy_forward, actions_taken)
        pg_loss            = tf.reduce_mean(cross_entropy_loss)
        # self.reg_loss           = tf.reduce_sum([tf.reduce_sum(tf.square(x)) for x in policy_network_variables])
        # self.loss               = self.pg_loss + self.reg_param * self.reg_loss
        #

        # compute gradients
        gradients = optimizer.compute_gradients(pg_loss)
        #
        # compute policy gradients
        for i, (grad, var) in enumerate(gradients):
          if grad is not None:
            gradients[i] = (grad * discounted_rewards, var)

        # training update
        # apply gradients to update policy network
        return optimizer.apply_gradients(gradients)


    @staticmethod
    def policy_forward(input, shape_layer1, shape_layer2 ):
        w1 = tf.get_variable("W1", shape_layer1 ,initializer=tf.random_normal_initializer())
        w2 = tf.get_variable("W2", shape_layer2,initializer=tf.random_normal_initializer())

        layer_1 = tf.matmul(input, w1)
        layer_1 = tf.nn.relu(layer_1) # ReLU nonlinearity
        out = tf.matmul(layer_1, w2)
        prob = tf.sub(tf.nn.softmax(out),tf.constant(1e-5))
        return out, prob

    def store(self, state, reward, action):
        self.state_buffer.append(state)
        self.reward_buffer.append(reward)
        self.action_buffer.append(action)

    def clean_up(self):
        self.state_buffer  = []
        self.reward_buffer = []
        self.action_buffer = []

    def get_discounted_rewards(self):
        N = len(self.reward_buffer)
        r = 0 # use discounted reward to approximate Q value

        # compute discounted future rewards
        discounted_rewards = np.zeros(N)
        for t in reversed(xrange(N)):
            # future discounted reward from now on
            r = self.reward_buffer[t] + self.gamma * r
            discounted_rewards[t] = r

        # reduce gradient variance by normalization
        self.all_rewards += discounted_rewards.tolist()
        self.all_rewards = self.all_rewards[:self.max_reward_length]
        discounted_rewards -= np.mean(self.all_rewards)
        discounted_rewards /= np.std(self.all_rewards)

        return discounted_rewards


if __name__ == '__main__':
    PG = PolicyGradient()

    # policy_network_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

    session = tf.Session()
    session.run(tf.initialize_all_variables())
    env = gym.make("Breakout-v0")


    for g in  xrange(100000):
        img = env.reset()
        for t in xrange(100000):
            # env.render()
            states =  PolicyGradient.pre_process(img)
            if random.random() < 0.5:
                a = random.randint(0,2)
            else:
                n,p = session.run([PG.policy_forward,PG.action_prob], feed_dict={PG.states:[states]})
                a = np.argmax(np.random.multinomial(1, p[0]))
                # print t,n,p[0],a

            img, reward, done, info = env.step(a+1)
            PG.store(states, reward-0.00001, a)
            if done:
                break

        discounted_rewards = PG.get_discounted_rewards()
        total_reward = 0;
        for t in xrange(len(PG.state_buffer)):

          # prepare inputs
          states  = PG.state_buffer[t]
          action = PG.action_buffer[t]
          reward = discounted_rewards[t]
          r = PG.reward_buffer[t]
          total_reward+=r+0.00001;

          session.run([PG.train_op], feed_dict={PG.states:[states], PG.actions_taken:[action], PG.discounted_rewards:[reward]})

        PG.clean_up()
        print "Game: " + str(g) + ", Reward: " + str(total_reward)
    exit()







