# Needed for spark
spark = False
if spark:
    from pyspark.context import SparkContext
    from pyspark.conf import SparkConf

    sc = SparkContext(conf=SparkConf().setAppName("pong_tf"))


def map_fun(worker_num):
    import tensorflow as tf
    import numpy as np
    import gym
    import random
    import subprocess
    import os
    import math

    class Env:
        def __init__(self, depth, render_on):
            self.env = gym.make("Pong-v0")
            self.depth = depth
            self.render_on = render_on

        def reset(self):
            self.env.reset()

        def rgb2gray(self,rgb):
            r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
            gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
            return gray

        def preprocess(self, img):

            img = img[35:195]  # crop
            img = self.rgb2gray(img)
            img = img[::2, ::2]  # downsample by factor of 2

            return img

        def step(self, action):
            total_reward = 0
            state = []
            done = False
            for i in range(self.depth):
                if not done:
                    if self.render_on: self.env.render()
                    img, reward, done, info = self.env.step(action)
                    total_reward += reward

                state.append(self.preprocess(img))

            return [np.transpose(state)], total_reward, done




    class Train:
        def __init__(self):
            self.dropout_on = True
            self.render_on = False
            self.depth = 3
            self.discount_factor=0.99
            self.classes = 3

            self.optimizer = tf.train.RMSPropOptimizer(1e-5, 0.99)

            # PlaceHolders
            self.x = tf.placeholder(tf.float32, shape=[None,80, 80, self.depth])
            self.y_ = tf.placeholder(tf.float32, shape=[None, self.classes])
            self.discounted_reward = tf.placeholder(tf.float32, shape=[None])
            self.keep_prob = tf.placeholder(tf.float32)

            self.y_conv = self.forward_network()



            self.x_buffer, self.y_buffer, self.reward_buffer = [], [], []  # input states, actions taken, rewards

            self.gradients = self.get_gradient()
            self.cross_entropy, self.policy_gradients, self.apply = self.apply_policy_gradient()

            #checkpointing
            self.saver = tf.train.Saver()
            self.checkpoint = "pong_tf.ckpt"
            self.resume = True



        @staticmethod
        def weight_variable(shape, name):
	    return tf.get_variable(name, shape, initializer=tf.random_normal_initializer())

        @staticmethod
        def bias_variable(shape, name):
	    return tf.get_variable(name, shape, initializer=tf.constant_initializer(0))

        @staticmethod
        def conv2d(x, W, stride, name):
            return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='SAME', name=name)

        @staticmethod
        def get_conv_layer(x_image, filter_size, input_depth, output_depth, stride, name):
            W = Train.weight_variable([filter_size, filter_size, input_depth, output_depth], name="W_" + name)
            b = Train.bias_variable([output_depth], name="B_" + name)

            return tf.nn.relu(Train.conv2d(x_image, W, stride=stride,name=name) + b)

        @staticmethod
        def get_fc_layer(x, input, output, name):
            W = Train.weight_variable([input, output], name="W_" + name)
            b = Train.bias_variable([output], name="B_" + name)
            return tf.nn.relu(tf.matmul(x, W) + b)

        def getParameterCount(self):
            total_parameters = 0
            for variable in tf.trainable_variables():
                # shape is an array of tf.Dimension
                shape = variable.get_shape()
                print(shape)
                print(len(shape))
                variable_parametes = 1
                for dim in shape:
                    print(dim)
                    variable_parametes *= dim.value
                print(variable_parametes)
                total_parameters += variable_parametes
            print(total_parameters)

        def forward_network(self):
            h_conv1 = Train.get_conv_layer(self.x,
                                           filter_size=8,
                                           input_depth=self.depth,
                                           output_depth=32,
                                           stride=4,
                                           name="conv1")

            h_conv2 = Train.get_conv_layer(h_conv1,
                                           filter_size=4,
                                           input_depth=32,
                                           output_depth=32,
                                           stride=2,
                                           name="conv2")

            h_conv3 = Train.get_conv_layer(h_conv2,
                                           filter_size=3,
                                           input_depth=32,
                                           output_depth=32,
                                           stride=1,
                                           name="conv3")

            # Fully connected layer 1
            h_pool2_flat = tf.reshape(h_conv3, [-1, 10 * 10 * 32])
            h_fc1 = Train.get_fc_layer(h_pool2_flat, input=10 * 10 * 32, output=200, name="fc1");

            # Dropout
            h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob) if self.dropout_on else h_fc1

            # Fully connected layer 2
            h_fc2 = Train.get_fc_layer(h_fc1_drop, input=200, output=3, name="fc2");

            # apply softmax to output
            return tf.sub(tf.nn.softmax(h_fc2),tf.constant(1e-6))


        def get_gradient(self):
            cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.y_ * tf.log(self.y_conv), reduction_indices=[1]))
            gradients  = tf.train.RMSPropOptimizer(1e-4, 0.99).compute_gradients(cross_entropy)
            return gradients

        def apply_policy_gradient(self):
            cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.y_ * tf.log(self.y_conv), reduction_indices=[1]))

            policy_gradients  = tf.train.RMSPropOptimizer(1e-4, 0.99).compute_gradients(cross_entropy)
            for i, (grad, var) in enumerate(policy_gradients):
                if grad is not None:
                    policy_gradients[i] = (grad * self.discounted_reward, var)

            return cross_entropy,policy_gradients, self.optimizer.apply_gradients(policy_gradients)


        def discount_rewards(self):
            """ take 1D float array of rewards and compute discounted reward """
            discounted_r = np.zeros_like(self.reward_buffer)
            running_add = 0
            for t in reversed(xrange(0,len(self.reward_buffer))):
                if self.reward_buffer[t] != 0: running_add = 0  # reset the sum, since this was a game boundary (pong specific!)
                running_add = running_add * self.discount_factor + self.reward_buffer[t]
                discounted_r[t] = running_add
            return discounted_r

        @staticmethod
        def copyToHDFS(local_file, hdfs_dir):
            FNULL = open(os.devnull, 'w')
            subprocess.call(["hdfs", "dfs", "-rm", "-r", "-skipTrash", hdfs_dir + "/" + local_file], stdout=FNULL,
                            stderr=FNULL)
            subprocess.call(["hdfs", "dfs", "-copyFromLocal", local_file, hdfs_dir], stdout=FNULL, stderr=FNULL)

        def run(self):

            sess = tf.Session()

            self.getParameterCount()


            sess.run(tf.initialize_all_variables())
            env = Env(self.depth, self.render_on)

            if self.resume and os.path.isfile(self.checkpoint):
                self.saver.restore(sess, self.checkpoint)
            running_reward = None
            for episode in range(1000000):
                env.reset()
                episode_reward = 0;
                # start with a random action
                action = random.randint(0,2)
                while(True):
                    state, reward, done = env.step(action+1)
                    episode_reward += reward
                    y_conv = sess.run([self.y_conv], feed_dict={self.x: state, self.keep_prob: 1.0})

                    y_ = [0.]*self.classes
                    y_[action]=1
                    self.x_buffer.append(state)
                    self.y_buffer.append([y_])
                    self.reward_buffer.append(reward)

                    if done:
                        break

                    action = np.argmax(np.random.multinomial(1, y_conv[0][0]))

                running_reward = episode_reward if running_reward is None else running_reward * 0.99 + episode_reward * 0.01
                print "Episode: %f Reward: %f, Mean Reward: %f" % (episode, episode_reward, running_reward)

                if (episode+1)%2 == 0:
                    print "%f updating model" % episode
                    disccounted_rewards = self.discount_rewards()
                    avg_loss = 0
                    min =  float("inf")
                    k = 0
                    for j in range(len(self.x_buffer)):
                        loss, policy_grads,_ = sess.run([self.cross_entropy, self.policy_gradients, self.apply],
                                                  feed_dict={
                                                      self.x:self.x_buffer[j],
                                                      self.y_: self.y_buffer[j],
                                                      self.discounted_reward:[disccounted_rewards[j]],
                                                      self.keep_prob: 1.0}
                                                  )
                        if not math.isnan(loss):
                            avg_loss = (k*avg_loss+loss)/(k+1)
                            min = loss if loss < min else min
                            k+=1
                    print k, avg_loss, min
                    self.x_buffer, self.y_buffer, self.reward_buffer = [], [], []
                    self.saver.save(sess, self.checkpoint)

                if (episode + 1) % 100 == 0:
                    print "Copying to HDFS"
                    Train.copyToHDFS(self.checkpoint, "/user/tripathi/checkpoint/")



            #     if i % 100 == 0:

            #     train_step.run(session=sess, feed_dict={self.x: batch[0], self.y_: batch[1], self.keep_prob: 0.5})
            #

    train = Train()
    train.run()
    # env = Env(3, False)
    # s,a,b = env.step(1)
    # print len(s)



if spark:
    numExecutors = int(sc._conf.get("spark.executor.instances"))
    rdd = sc.parallelize(range(numExecutors), numExecutors)
    rdd.map(map_fun).collect()
else:
    map_fun(1)
