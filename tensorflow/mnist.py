# Needed for spark
spark = False
if spark:
    from pyspark.context import SparkContext
    from pyspark.conf import SparkConf

    sc = SparkContext(conf=SparkConf().setAppName("mnist_tripathi"))


def map_fun(worker_num):
    import tensorflow as tf
    import numpy as np

    class Train:
        def __init__(self):
            self.dropout_on = True

            # PlaceHolders
            self.x = tf.placeholder(tf.float32, shape=[None, 784])  # None X 784
            self.y_ = tf.placeholder(tf.float32, shape=[None, 10])
            self.discounted_reward = tf.placeholder(tf.float32, shape=[None])
            self.keep_prob = tf.placeholder(tf.float32)

            self.y_conv = self.forward_network()

            self.optimizer = tf.train.RMSPropOptimizer(1e-6, 0.99)

            self.x_buffer, self.y_buffer, self.reward_buffer = [], [], []  # input states, actions taken, rewards

            self.gradients = self.get_gradient()
            self.policy_gradients, self.apply = self.apply_policy_gradient()



        @staticmethod
        def weight_variable(shape, name):
            initial = tf.truncated_normal(shape, stddev=0.1, name=name)
            return tf.Variable(initial)

        @staticmethod
        def bias_variable(shape, name):
            initial = tf.constant(0.1, shape=shape, name=name)
            return tf.Variable(initial)

        @staticmethod
        def conv2d(x, W, stride, name):
            return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='SAME', name=name)

        @staticmethod
        def get_conv_layer(x_image, filter_size, input_depth, output_depth, stride, name):
            W = Train.weight_variable([filter_size, filter_size, input_depth, output_depth], name="W_" + name)
            b = Train.bias_variable([output_depth], name="B_" + name)

            return tf.nn.relu(Train.conv2d(x_image, W, stride=stride,
                                           name=name) + b)  # stride = 4, zero-padding - calculated automatically

        @staticmethod
        def get_fc_layer(x, input, output, name):
            W = Train.weight_variable([input, output], name="W_" + name)
            b = Train.bias_variable([output], name="B_" + name)
            return tf.nn.relu(tf.matmul(x, W) + b)

        def forward_network(self):

            x_image = tf.reshape(self.x, [-1, 28, 28, 1])  # None X 28 X 28 X 1
            h_conv1 = Train.get_conv_layer(x_image,
                                           filter_size=8,
                                           input_depth=1,
                                           output_depth=32,
                                           stride=4,
                                           name="conv1")

            h_conv2 = Train.get_conv_layer(h_conv1,
                                           filter_size=4,
                                           input_depth=32,
                                           output_depth=64,
                                           stride=2,
                                           name="conv2")

            h_conv3 = Train.get_conv_layer(h_conv2,
                                           filter_size=3,
                                           input_depth=64,
                                           output_depth=64,
                                           stride=1,
                                           name="conv3")

            # Fully connected layer 1
            h_pool2_flat = tf.reshape(h_conv3, [-1, 4 * 4 * 64])
            h_fc1 = Train.get_fc_layer(h_pool2_flat, input=4 * 4 * 64, output=1024, name="fc1");

            # Dropout
            h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob) if self.dropout_on else h_fc1

            # Fully connected layer 2
            h_fc2 = Train.get_fc_layer(h_fc1_drop, input=1024, output=10, name="fc2");

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

            return policy_gradients, self.optimizer.apply_gradients(policy_gradients)


        def run(self):
            from tensorflow.examples.tutorials.mnist import input_data
            mnist = input_data.read_data_sets('mnist', one_hot=True)

            sess = tf.Session()
            sess.run(tf.initialize_all_variables())



            # train_step = tf.train.RMSPropOptimizer(1e-4, 0.99).minimize(cross_entropy)
            correct_prediction = tf.equal(tf.argmax(self.y_conv, 1), tf.argmax(self.y_, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            for i in range(20*50):

                if (i+1) % 1000 == 0:
                    batch = mnist.train.next_batch(50)
                    train_accuracy = accuracy.eval(session=sess,
                                                   feed_dict={self.x: batch[0], self.y_: batch[1], self.keep_prob: 1.0})
                    print("step %d, training accuracy %g" % (i, train_accuracy))

                else:
                    batch = mnist.train.next_batch(1)
                    y_conv = sess.run([self.y_conv], feed_dict={self.x: batch[0], self.keep_prob: 1.0})
                    y_ = [0.]*10

                    decision = np.argmax(np.random.multinomial(1, y_conv[0][0]))
                    y_[decision] = 1.0
                    reward = 5 if np.argmax((batch[1]))==decision else -1
                    self.x_buffer.append(batch[0])
                    self.y_buffer.append([y_])
                    self.reward_buffer.append([reward])

                    if (i+1)%50 == 0:
                        print "%f updating model" % i
                        for j in range(len(self.x_buffer)):
                            # grads = sess.run([self.gradients], feed_dict={self.x:self.x_buffer[j], self.y_: self.y_buffer[j], self.discounted_reward:self.reward_buffer[j], self.keep_prob: 1.0})
                            policy_grads,_ = sess.run([self.policy_gradients, self.apply], feed_dict={self.x:self.x_buffer[j], self.y_: self.y_buffer[j], self.discounted_reward:self.reward_buffer[j], self.keep_prob: 1.0})

                        self.x_buffer, self.y_buffer, self.reward_buffer = [], [], []


            #     if i % 100 == 0:

            #     train_step.run(session=sess, feed_dict={self.x: batch[0], self.y_: batch[1], self.keep_prob: 0.5})
            #
            print("test accuracy %g" % accuracy.eval(session=sess, feed_dict={self.x: mnist.test.images, self.y_: mnist.test.labels, self.keep_prob: 1.0}))

    train = Train()
    train.run()


if spark:
    numExecutors = int(sc._conf.get("spark.executor.instances"))
    rdd = sc.parallelize(range(numExecutors), numExecutors)
    rdd.map(map_fun).collect()
else:
    map_fun(1)
