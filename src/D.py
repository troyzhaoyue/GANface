import tensorflow as tf
import tensorflow.contrib.slim as slim

def discriminator(input, reuse=None, hidden_layer_num=1, num_of_units=2048):
    with tf.variable_scope("Discriminator",reuse=reuse) as D_scope:
        with tf.variable_scope("d_input") as s:
            net = slim.fully_connected(input, num_of_units, activation_fn=None,scope=s, reuse=reuse)

        for iter in range(hidden_layer_num):
            with tf.variable_scope("d_hidden%d"%(iter+1)) as s:
                net = slim.fully_connected(net, num_of_units, activation_fn=None, scope=s, reuse=reuse)

        with tf.variable_scope("d_output") as s:
            net = slim.fully_connected(net, 1, activation_fn=None, scope=s, reuse=reuse)

    D_variables = tf.contrib.framework.get_variables(D_scope)

    return tf.nn.sigmoid(net), net, D_variables


