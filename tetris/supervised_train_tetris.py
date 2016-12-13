# Needed for spark
spark = False
if spark:
    from pyspark.context import SparkContext
    from pyspark.conf import SparkConf

    sc = SparkContext(conf=SparkConf().setAppName("tetris_tripathi"))


def map_fun(worker_num):
    import cPickle as pickle
    from os import listdir
    from os.path import isfile, join, splitext, getsize
    import tensorflow as tf
    import numpy as np
    import time
    try:
        from tetris1 import TetrisApp
    except :
        pass

    class Train:
        def __init__(self):
            self.dropout_on = True

            # PlaceHolders
            self.x = tf.placeholder(tf.float32, shape=[None, 20, 44])  # None X 784
            self.y_ = tf.placeholder(tf.float32, shape=[None, 1])
            self.discounted_reward = tf.placeholder(tf.float32, shape=[None])
            self.keep_prob = tf.placeholder(tf.float32)

            self.y_conv = self.forward_network()

            cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.y_ * tf.log(self.y_conv), reduction_indices=[1]))
            self.train = tf.train.RMSPropOptimizer(1e-3, 0.95).minimize(cross_entropy)


            correct_prediction = tf.equal(tf.argmax(self.y_conv, 1), tf.argmax(self.y_, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            self.saver = tf.train.Saver()

            # data file list
            self.trainfile = [join('./train',f) for f in listdir('./train')
                              if isfile(join('./train',f))
                              and splitext(join('./train',f))[1] == '.p'
                              and getsize(join('./train',f))>0]
            self.testfile = [join('./test',f) for f in listdir('./test')
                             if isfile(join('./test',f))
                             and splitext(join('./test',f))[1] == '.p'
                             and getsize(join('./test',f))>0]



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
            x_image = tf.reshape(self.x, [-1, 20, 44, 1])  # None X 28 X 28 X 1
            h_conv1 = Train.get_conv_layer(x_image,
                                           filter_size=4,
                                           input_depth=1,
                                           output_depth=4,
                                           stride=1,
                                           name="conv1")

            h_conv2 = Train.get_conv_layer(h_conv1,
                                           filter_size=4,
                                           input_depth=4,
                                           output_depth=8,
                                           stride=1,
                                           name="conv2")

            h_conv3 = Train.get_conv_layer(h_conv2,
                                           filter_size=4,
                                           input_depth=8,
                                           output_depth=8,
                                           stride=1,
                                           name="conv3")

            # Fully connected layer 1
            h_pool2_flat = tf.reshape(h_conv3, [-1, 880*8])
            h_fc1 = Train.get_fc_layer(h_pool2_flat, input=880*8, output=1024, name="fc1");

            # Dropout
            h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob) if self.dropout_on else h_fc1

            # Fully connected layer 2
            h_fc2 = Train.get_fc_layer(h_fc1_drop, input=1024, output=4, name="fc2");

            # apply softmax to output
            return tf.sub(tf.nn.softmax(h_fc2),tf.constant(1e-6))

        def apply_policy_gradient(self):
            cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.y_ * tf.log(self.y_conv), reduction_indices=[1]))
            policy_gradients  = tf.train.RMSPropOptimizer(1e-4, 0.99).compute_gradients(cross_entropy)
            for i, (grad, var) in enumerate(policy_gradients):
                if grad is not None:
                    policy_gradients[i] = (grad * self.discounted_reward, var)

            return policy_gradients, self.optimizer.apply_gradients(policy_gradients)


        def get_batch(self, test = False):
            if test:
                random_file = self.testfile[np.random.randint(0, len(self.testfile))]
            else:
                random_file = self.trainfile[np.random.randint(0, len(self.trainfile))]

            x_buffer,y_buffer = [],[]
            f = open(random_file , 'r')
            while True:
                try:
                    obj = pickle.load(f)
                    for i in np.random.permutation(len(obj['obj'])):
                        # print obj['ep'],obj['obj'][i]['step']
                        x = obj['obj'][i]['obs']
                        y = obj['obj'][i]['action']
                        if 1 <= y <= 4 and (np.random.uniform() >0.9 or y != 1): #and not test) or (test and 2 <= y <= 4):

                            x_buffer.append(x)
                            y_buffer.append([y-1])
                except EOFError:
                    break
            f.close()
            return x_buffer,y_buffer


        def run(self):
            sess = tf.Session()
            sess.run(tf.initialize_all_variables())

            for i in range(500):
                x,y = self.get_batch()
                sess.run([self.train], feed_dict={self.x: x, self.y_: y, self.keep_prob: 1.0})

            save_path = self.saver.save(sess, "./model.ckpt")
            print("Model saved in file: %s" % save_path)

            # y_conv = sess.run([self.y_conv], feed_dict={self.x: x, self.keep_prob: 1.0})


            # print np.argmax(y_conv[0],1),y

            for i in range(10):
                x,y = self.get_batch(True)
                accuracy = sess.run([self.accuracy], feed_dict={self.x: x, self.y_: y, self.keep_prob: 1.0})
                print accuracy


            # print len(x), len(x[0]), len(x[0][0]), len(y)

        def play(self):
            App = TetrisApp()
            sess = tf.Session()
            self.saver.restore(sess, "./model.ckpt")
            action = 1
            while True:
                obs, reward, done, info = App.step(action)
                App.render()
                if done:
                    action = 1
                    App.reset()
                    done = False
                else:
                    x = [obs]
                    y_conv = sess.run([self.y_conv], feed_dict={self.x: x, self.keep_prob: 1.0})
                    action = np.argmax(y_conv[0],1)[0] + 1
                    # print y_conv,action

                    print action
                time.sleep(0.1)





    train = Train()
    train.play()



if spark:
    numExecutors = int(sc._conf.get("spark.executor.instances"))
    rdd = sc.parallelize(range(numExecutors), numExecutors)
    rdd.map(map_fun).collect()
else:
    map_fun(1)





