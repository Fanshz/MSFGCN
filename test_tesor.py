import tensorflow as tf
import numpy as np

input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)

output1 = tf.multiply(input1, input2)
output2 = tf.multiply(output1, input1)

with tf.Session() as sess:
  print(sess.run(output2, feed_dict={input1: 7, input2: 2.}))