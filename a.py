import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


import  os
print os.environ

import pip
installed_packages = pip.get_installed_distributions()
installed_packages_list = sorted(["%s==%s" % (i.key, i.version)
     for i in installed_packages])
print(installed_packages_list)
# plt.plot([1,2,3])
#print np.random.randn(200,80*80)

n_values = 32
x = tf.linspace(-3.0, 3.0, n_values)

# %% Construct a tf.Session to execute the graph.
sess = tf.Session()
result = sess.run(x)
print result

sigma = 1.0
mean = 0.0
z = (tf.exp(tf.neg(tf.pow(x - mean, 2.0) /
                   (2.0 * tf.pow(sigma, 2.0)))) *
     (1.0 / (sigma * tf.sqrt(2.0 * 3.1415))))

assert z.graph is tf.get_default_graph()
plt.imshow(z.eval(session=sess))
print(z.get_shape())
print tf.shape(z).eval(session=sess)
print(tf.pack([tf.shape(z), tf.shape(z), [3], [4]]).eval(session=sess))
result = sess.run(z)
print result
sess.close()