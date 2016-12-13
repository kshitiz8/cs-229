import tensorflow as tf
import gym
import numpy as np
import random
import os.path

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
        self.render = False

        self.state_buffer  = []
        self.reward_buffer = []
        self.action_buffer = []

        self.all_rewards = []
        self.max_reward_length = 1000000

        self.states = tf.placeholder(tf.float32, [1,self.D], name="states")
        self.policy_forward, self.action_prob = PolicyGradient.policy_forward(self.states, self.D, self.H, self.n_classes)

        self.actions_taken = tf.placeholder(tf.int32, (None,), name="taken_actions")
        self.discounted_rewards = tf.placeholder(tf.float32, (None,), name="discounted_rewards")
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate, decay=self.decay_rate)
        [self.cross_entropy_loss, self.pg_loss, self.gradients,self.train_op] = PolicyGradient.rmsprop(policy_forward=self.policy_forward,
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
        return cross_entropy_loss, pg_loss, gradients,optimizer.apply_gradients(gradients)


    @staticmethod
    def policy_forward(input, n_input, n_layer1, n_layer2 ):
        w1 = tf.get_variable("W1", [n_input, n_layer1] ,initializer=tf.random_normal_initializer())
        b1 = tf.get_variable("b1",[n_layer1] ,initializer=tf.constant_initializer(0))
        w2 = tf.get_variable("W2", [n_layer1, n_layer2],initializer=tf.random_normal_initializer())
        b2 = tf.get_variable("b2",[n_layer2] ,initializer=tf.constant_initializer(0))

        layer_1 = tf.nn.relu(tf.matmul(input, w1) + b1) # ReLU nonlinearity
        out = tf.matmul(layer_1, w2)+b2
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
            aa = 1
            if self.reward_buffer[t] == -1 : aa=0
            r = self.reward_buffer[t] + self.gamma * r *aa
            discounted_rewards[t] = r

        # reduce gradient variance by normalization
        self.all_rewards += discounted_rewards.tolist()
        self.all_rewards = self.all_rewards[:self.max_reward_length]
        discounted_rewards -= np.mean(self.all_rewards)
        discounted_rewards /= np.std(self.all_rewards)

        return discounted_rewards


    def run(self):
        session = tf.Session()
        session.run(tf.initialize_all_variables())
        env = gym.make("Pong-v0")
        saver = tf.train.Saver()
        checkpoint = "pong.ckpt"
        if self.resume and os.path.isfile(checkpoint): saver.restore(session, checkpoint)
        running_reward = None;
        for g in  xrange(100000):
            img = env.reset()
            total_reward = 0;
            for t in xrange(100000):
                if self.render: env.render()

                states =  PolicyGradient.pre_process(img)
                if random.random() < 0.5:
                    a = random.randint(0,2)
                else:
                    n,p = session.run([self.policy_forward,self.action_prob], feed_dict={self.states:[states]})
                    a = np.argmax(np.random.multinomial(1, p[0]))
                    # print t,n,p[0],a

                img, reward, done, info = env.step(a+1)
                self.store(states, reward, a)
                total_reward+=reward;
                if done:
                    break







            running_reward = total_reward if running_reward is None else running_reward * 0.99 + total_reward * 0.01
            print "Game: %f, Reward: %f, Running Mean: %f" % (g  , total_reward,running_reward)
            if g % 10 == 0:
                print "update"
                discounted_rewards = self.get_discounted_rewards()
                for t in xrange(len(self.state_buffer)):

                  # prepare inputs
                  states  = self.state_buffer[t]
                  action = self.action_buffer[t]
                  reward = discounted_rewards[t]
                  r = self.reward_buffer[t]
                  # print r, reward

                [xx,yy,zz,ss] = session.run([self.cross_entropy_loss, self.pg_loss, self.gradients,self.train_op]
                                            , feed_dict={self.states:[states],
                                                         self.actions_taken:[action],
                                                         self.discounted_rewards:[reward]})
                print xx, yy, len(zz[1])
                saver.save(session, checkpoint)
                self.clean_up()


if __name__ == '__main__':
    pg = PolicyGradient()
    pg.run()






