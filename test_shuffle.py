import tensorflow as tf
import numpy as np
p = tf.Variable([0.2, 0.7, 0.049, 0.051])
maxi = tf.argmax(p)
d = tf.Variable(tf.random.shuffle(p))
u = tf.Variable(d)
maxi_d = tf.argmax(d)
indices = tf.Variable([[maxi], [maxi_d]])
updates = tf.Variable([p[maxi], d[maxi]])
u = tf.scatter_nd_update(u, indices, updates)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
print(sess.run(u))

P = tf.Variable([[0.2, 0.7, 0.049, 0.051],
                [0.8, 0.09, 0.058, 0.052]]);

def non_maximal_class_shuffle(p):
    p1 = tf.Variable(initial_value=lambda: tf.zeros(tf.shape(p)))
    p = p1.assign(p)
    d = tf.Variable(initial_value=lambda: tf.zeros(tf.shape(p)))
    d = d.assign(p)
    d = tf.random.shuffle(d)
    u = tf.Variable(initial_value=lambda: tf.zeros(tf.shape(p)))
    u = u.assign(d)
    maxi = tf.argmax(p)
    maxi_d = tf.argmax(d)
    indices = tf.Variable(initial_value=lambda: tf.zeros((2,1), dtype=tf.int32))
    indices = indices.assign([[maxi], [maxi_d]])
    updates = tf.Variable(initial_value=lambda: tf.zeros((2)))
    updates = updates.assign([p[maxi], d[maxi]])
    return tf.scatter_nd_update(u, indices, updates)
U = tf.map_fn(non_maximal_class_shuffle, P)
sess = tf.Session();
sess.run(tf.global_variables_initializer());
print(sess.run(P));
print(sess.run(U));
