spark = True
if spark:
    from pyspark.context import SparkContext
    from pyspark.conf import SparkConf

    sc = SparkContext(conf=SparkConf().setAppName("breakout_tf"))

def map_fun(worker_num):

    from __future__ import print_function
    from collections import namedtuple
    import numpy as np
    import tensorflow as tf
    import six.moves.queue as queue
    import scipy.signal
    import threading



    #### Model

    def normalized_columns_initializer(std=1.0):
        def _initializer(shape, dtype=None, partition_info=None):
            out = np.random.randn(*shape).astype(np.float32)
            out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
            return tf.constant(out)
        return _initializer

    def flatten(x):
        return tf.reshape(x, [-1, np.prod(x.get_shape().as_list()[1:])])

    def conv2d(x, num_filters, name, filter_size=(3, 3), stride=(1, 1), pad="SAME", dtype=tf.float32, collections=None):
        with tf.variable_scope(name):
            stride_shape = [1, stride[0], stride[1], 1]
            filter_shape = [filter_size[0], filter_size[1], int(x.get_shape()[3]), num_filters]

            # there are "num input feature maps * filter height * filter width"
            # inputs to each hidden unit
            fan_in = np.prod(filter_shape[:3])
            # each unit in the lower layer receives a gradient from:
            # "num output feature maps * filter height * filter width" /
            #   pooling size
            fan_out = np.prod(filter_shape[:2]) * num_filters
            # initialize weights with random weights
            w_bound = np.sqrt(6. / (fan_in + fan_out))

            w = tf.get_variable("W", filter_shape, dtype, tf.random_uniform_initializer(-w_bound, w_bound),
                                collections=collections)
            b = tf.get_variable("b", [1, 1, 1, num_filters], initializer=tf.zeros_initializer,
                                collections=collections)
            return tf.nn.conv2d(x, w, stride_shape, pad) + b

    def linear(x, size, name, initializer=None, bias_init=0):
        w = tf.get_variable(name + "/w", [x.get_shape()[1], size], initializer=initializer)
        b = tf.get_variable(name + "/b", [size], initializer=tf.constant_initializer(bias_init))
        return tf.matmul(x, w) + b

    def categorical_sample(logits, d):
        value = tf.squeeze(tf.multinomial(logits - tf.reduce_max(logits, [1], keep_dims=True), 1), [1])
        return tf.one_hot(value, d)

    class LSTMPolicy(object):
        def __init__(self, ob_space, ac_space):
            self.x = x = tf.placeholder(tf.float32, [None] + list(ob_space))

            for i in range(4):
                x = tf.nn.elu(conv2d(x, 32, "l{}".format(i + 1), [3, 3], [2, 2]))
            # introduce a "fake" batch dimension of 1 after flatten so that we can do LSTM over time dim
            x = tf.expand_dims(flatten(x), [0])

            size = 256
            lstm = tf.nn.rnn_cell.BasicLSTMCell(size, state_is_tuple=True)
            self.state_size = lstm.state_size
            step_size = tf.shape(self.x)[:1]

            c_init = np.zeros((1, lstm.state_size.c), np.float32)
            h_init = np.zeros((1, lstm.state_size.h), np.float32)
            self.state_init = [c_init, h_init]
            c_in = tf.placeholder(tf.float32, [1, lstm.state_size.c])
            h_in = tf.placeholder(tf.float32, [1, lstm.state_size.h])
            self.state_in = [c_in, h_in]

            state_in = tf.nn.rnn_cell.LSTMStateTuple(c_in, h_in)
            lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
                lstm, x, initial_state=state_in, sequence_length=step_size,
                time_major=False)
            lstm_c, lstm_h = lstm_state
            x = tf.reshape(lstm_outputs, [-1, size])
            self.logits = linear(x, ac_space, "action", normalized_columns_initializer(0.01))
            self.vf = tf.reshape(linear(x, 1, "value", normalized_columns_initializer(1.0)), [-1])
            self.state_out = [lstm_c[:1, :], lstm_h[:1, :]]
            self.sample = categorical_sample(self.logits, ac_space)[0, :]
            self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)

        def get_initial_features(self):
            return self.state_init

        def act(self, ob, c, h):
            sess = tf.get_default_session()
            return sess.run([self.sample, self.vf] + self.state_out,
                            {self.x: [ob], self.state_in[0]: c, self.state_in[1]: h})

        def value(self, ob, c, h):
            sess = tf.get_default_session()
            return sess.run(self.vf, {self.x: [ob], self.state_in[0]: c, self.state_in[1]: h})[0]
    ####


    ### A3C ###
    def discount(x, gamma):
        return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

    def process_rollout(rollout, gamma, lambda_=1.0):
        """
    given a rollout, compute its returns and the advantage
    """
        batch_si = np.asarray(rollout.states)
        batch_a = np.asarray(rollout.actions)
        rewards = np.asarray(rollout.rewards)
        vpred_t = np.asarray(rollout.values + [rollout.r])

        rewards_plus_v = np.asarray(rollout.rewards + [rollout.r])
        batch_r = discount(rewards_plus_v, gamma)[:-1]
        delta_t = rewards + gamma * vpred_t[1:] - vpred_t[:-1]
        # this formula for the advantage comes "Generalized Advantage Estimation":
        # https://arxiv.org/abs/1506.02438
        batch_adv = discount(delta_t, gamma * lambda_)

        features = rollout.features[0]
        return Batch(batch_si, batch_a, batch_adv, batch_r, rollout.terminal, features)

    Batch = namedtuple("Batch", ["si", "a", "adv", "r", "terminal", "features"])

    class PartialRollout(object):
        """
    a piece of a complete rollout.  We run our agent, and and process its experience
    once it has processed enough steps.
    """
        def __init__(self):
            self.states = []
            self.actions = []
            self.rewards = []
            self.values = []
            self.r = 0.0
            self.terminal = False
            self.features = []

        def add(self, state, action, reward, value, terminal, features):
            self.states += [state]
            self.actions += [action]
            self.rewards += [reward]
            self.values += [value]
            self.terminal = terminal
            self.features += [features]

        def extend(self, other):
            assert not self.terminal
            self.states.extend(other.states)
            self.actions.extend(other.actions)
            self.rewards.extend(other.rewards)
            self.values.extend(other.values)
            self.r = other.r
            self.terminal = other.terminal
            self.features.extend(other.features)

    class RunnerThread(threading.Thread):
        """
    One of the key distinctions between a normal environment and a universe environment
    is that a universe environment is _real time_.  This means that there should be a thread
    that would constantly interact with the environment and tell it what to do.  This thread is here.
    """
        def __init__(self, env, policy, num_local_steps):
            threading.Thread.__init__(self)
            self.queue = queue.Queue(5)
            self.num_local_steps = num_local_steps
            self.env = env
            self.last_features = None
            self.policy = policy
            self.daemon = True
            self.sess = None
            self.summary_writer = None

        def start_runner(self, sess, summary_writer):
            self.sess = sess
            self.summary_writer = summary_writer
            self.start()

        def run(self):
            with self.sess.as_default():
                self._run()

        def _run(self):
            rollout_provider = env_runner(self.env, self.policy, self.num_local_steps, self.summary_writer)
            while True:
                # the timeout variable exists becuase apparently, if one worker dies, the other workers
                # won't die with it, unless the timeout is set to some large number.  This is an empirical
                # observation.

                self.queue.put(next(rollout_provider), timeout=600.0)




    def env_runner(env, policy, num_local_steps, summary_writer):
        """
    The logic of the thread runner.  In brief, it constantly keeps on running
    the policy, and as long as the rollout exceeds a certain length, the therad
    runner appends the policy to the queue.
    """
        last_state = env.reset()
        last_features = policy.get_initial_features()
        length = 0
        rewards = 0

        while True:
            terminal_end = False
            rollout = PartialRollout()

            for _ in range(num_local_steps):
                fetched = policy.act(last_state, *last_features)
                action, value_, features = fetched[0], fetched[1], fetched[2:]
                # argmax to convert from one-hot
                state, reward, terminal, info = env.step(action.argmax())

                # collect the experience
                rollout.add(last_state, action, reward, value_, terminal, last_features)
                length += 1
                rewards += reward

                last_state = state
                last_features = features

                if info:
                    summary = tf.Summary()
                    for k, v in info.items():
                        summary.value.add(tag=k, simple_value=float(v))
                    summary_writer.add_summary(summary, policy.global_step.eval())
                    summary_writer.flush()


                if terminal or length >= env.spec.timestep_limit:
                    terminal_end = True
                    if length >= env.spec.timestep_limit or not env.metadata.get('semantics.autoreset'):
                        last_state = env.reset()
                    last_features = policy.get_initial_features()
                    print("Episode finished. Sum of rewards: %d. Length: %d" % (rewards, length))
                    length = 0
                    rewards = 0
                    break

            if not terminal_end:
                rollout.r = policy.value(last_state, *last_features)

            # once we have enough experience, yield it, and have the TheradRunner place it on a queue
            yield rollout

    class A3C(object):
        def __init__(self, env, task):
            """
    An implementation of the A3C algorithm that is reasonably well-tuned for the VNC environments.
    Below, we will have a modest amount of complexity due to the way TensorFlow handles data parallelism.
    But overall, we'll define the model, specify its inputs, and describe how the policy graidents step
    should be computed.
    """

            self.env = env
            self.task = task
            worker_device = "/job:worker/task:{}/cpu:0".format(task)
            with tf.device(tf.train.replica_device_setter(1, worker_device=worker_device)):
                with tf.variable_scope("global"):
                    self.network = LSTMPolicy(env.observation_space.shape, env.action_space.n)
                    self.global_step = tf.get_variable("global_step", [], tf.int32, initializer=tf.zeros_initializer,
                                                       trainable=False)

            with tf.device(worker_device):
                with tf.variable_scope("local"):
                    self.local_network = pi = LSTMPolicy(env.observation_space.shape, env.action_space.n)
                    pi.global_step = self.global_step

                self.ac = tf.placeholder(tf.float32, [None, env.action_space.n], name="ac")
                self.adv = tf.placeholder(tf.float32, [None], name="adv")
                self.r = tf.placeholder(tf.float32, [None], name="r")

                log_prob_tf = tf.nn.log_softmax(pi.logits)
                prob_tf = tf.nn.softmax(pi.logits)

                # the "policy gradients" loss:  its derivative is precisely the policy gradient
                # notice that self.ac is a placeholder that is provided externally.
                # ac will contain the advantages, as calculated in process_rollout
                pi_loss = - tf.reduce_sum(tf.reduce_sum(log_prob_tf * self.ac, [1]) * self.adv)

                # loss of value function
                vf_loss = 0.5 * tf.reduce_sum(tf.square(pi.vf - self.r))
                entropy = - tf.reduce_sum(prob_tf * log_prob_tf)

                bs = tf.to_float(tf.shape(pi.x)[0])
                self.loss = pi_loss + 0.5 * vf_loss - entropy * 0.01

                # 20 represents the number of "local steps":  the number of timesteps
                # we run the policy before we update the parameters.
                # The larger local steps is, the lower is the variance in our policy gradients estimate
                # on the one hand;  but on the other hand, we get less frequent parameter updates, which
                # slows down learning.  In this code, we found that making local steps be much
                # smaller than 20 makes the algorithm more difficult to tune and to get to work.
                self.runner = RunnerThread(env, pi, 20)


                grads = tf.gradients(self.loss, pi.var_list)

                tf.scalar_summary("model/policy_loss", pi_loss / bs)
                tf.scalar_summary("model/value_loss", vf_loss / bs)
                tf.scalar_summary("model/entropy", entropy / bs)
                tf.image_summary("model/state", pi.x)
                tf.scalar_summary("model/grad_global_norm", tf.global_norm(grads))
                tf.scalar_summary("model/var_global_norm", tf.global_norm(pi.var_list))

                self.summary_op = tf.merge_all_summaries()
                grads, _ = tf.clip_by_global_norm(grads, 40.0)

                # copy weights from the parameter server to the local model
                self.sync = tf.group(*[v1.assign(v2) for v1, v2 in zip(pi.var_list, self.network.var_list)])

                grads_and_vars = list(zip(grads, self.network.var_list))
                inc_step = self.global_step.assign_add(tf.shape(pi.x)[0])

                # each worker has a different set of adam optimizer parameters
                opt = tf.train.AdamOptimizer(1e-4)
                self.train_op = tf.group(opt.apply_gradients(grads_and_vars), inc_step)
                self.summary_writer = None
                self.local_steps = 0

        def start(self, sess, summary_writer):
            self.runner.start_runner(sess, summary_writer)
            self.summary_writer = summary_writer

        def pull_batch_from_queue(self):
            """
    self explanatory:  take a rollout from the queue of the thread runner.
    """
            rollout = self.runner.queue.get(timeout=600.0)
            while not rollout.terminal:
                try:
                    rollout.extend(self.runner.queue.get_nowait())
                except queue.Empty:
                    break
            return rollout

        def process(self, sess):
            """
    process grabs a rollout that's been produced by the thread runner,
    and updates the parameters.  The update is then sent to the parameter
    server.
    """

            sess.run(self.sync)  # copy weights from shared to local
            rollout = self.pull_batch_from_queue()
            batch = process_rollout(rollout, gamma=0.99, lambda_=1.0)

            should_compute_summary = self.task == 0 and self.local_steps % 11 == 0

            if should_compute_summary:
                fetches = [self.summary_op, self.train_op, self.global_step]
            else:
                fetches = [self.train_op, self.global_step]

            feed_dict = {
                self.local_network.x: batch.si,
                self.ac: batch.a,
                self.adv: batch.adv,
                self.r: batch.r,
                self.local_network.state_in[0]: batch.features[0],
                self.local_network.state_in[1]: batch.features[1],
            }

            fetched = sess.run(fetches, feed_dict=feed_dict)

            if should_compute_summary:
                self.summary_writer.add_summary(tf.Summary.FromString(fetched[0]), fetched[-1])
                self.summary_writer.flush()
            self.local_steps += 1
    #### end A3C


    ### Worker
    class FastSaver(tf.train.Saver):
        def save(self, sess, save_path, global_step=None, latest_filename=None,
                 meta_graph_suffix="meta", write_meta_graph=True):
            super(FastSaver, self).save(sess, save_path, global_step, latest_filename,
                                        meta_graph_suffix, False)



    def run(args, server):
        env = create_env(args.env_id, client_id=str(args.task), remotes=args.remotes)
        trainer = A3C(env, args.task)

        # Variable names that start with "local" are not saved in checkpoints.
        variables_to_save = [v for v in tf.all_variables() if not v.name.startswith("local")]
        init_op = tf.initialize_variables(variables_to_save)
        init_all_op = tf.initialize_all_variables()
        saver = FastSaver(variables_to_save)

        def init_fn(ses):
            logger.info("Initializing all parameters.")
            ses.run(init_all_op)

        config = tf.ConfigProto(device_filters=["/job:ps", "/job:worker/task:{}/cpu:0".format(args.task)])
        logdir = os.path.join(args.log_dir, 'train')
        summary_writer = tf.train.SummaryWriter(logdir + "_%d" % args.task)
        logger.info("Events directory: %s_%s", logdir, args.task)
        sv = tf.train.Supervisor(is_chief=(args.task == 0),
                                 logdir=logdir,
                                 saver=saver,
                                 summary_op=None,
                                 init_op=init_op,
                                 init_fn=init_fn,
                                 summary_writer=summary_writer,
                                 ready_op=tf.report_uninitialized_variables(variables_to_save),
                                 global_step=trainer.global_step,
                                 save_model_secs=30,
                                 save_summaries_secs=30)

        num_global_steps = 100000000

        logger.info(
            "Starting session. If this hangs, we're mostly likely waiting to connect to the parameter server. " +
            "One common cause is that the parameter server DNS name isn't resolving yet, or is misspecified.")
        with sv.managed_session(server.target, config=config) as sess, sess.as_default():
            trainer.start(sess, summary_writer)
            global_step = sess.run(trainer.global_step)
            logger.info("Starting training at step=%d", global_step)
            while not sv.should_stop() and (not num_global_steps or global_step < num_global_steps):
                trainer.process(sess)
                global_step = sess.run(trainer.global_step)

        # Ask for all the services to stop.
        sv.stop()
        logger.info('reached %s steps. worker stopped.', global_step)
