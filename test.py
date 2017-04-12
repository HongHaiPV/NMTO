from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import tensorflow as tf

arr = [[1,2,3,1], [1,,2,3], [1,2,3]]
arr = tf.reshape(arr, [-1, 1])

sess = tf.Session()
print(sess.run(arr))
