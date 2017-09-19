import os.path, gym
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import roboschool

class ZooPolicyTensorflow(object):
    def __init__(self, name, ob_space, ac_space, take_weights_here=None):
        self.name = name

        with tf.variable_scope(name):
            obs_tuple = [
                tf.placeholder(tf.float32,         (None, 1), name="obs0"),
                tf.placeholder(tf.float32,        (None, 28), name="obs1"),
            ]
            self.obs_tuple = obs_tuple

            actions_input = []
            actions_input.append(obs_tuple[1])

            x = tf.concat( actions_input, axis=1 )
            dense1_w = tf.get_variable("dense1_w", [28,128])
            dense1_b = tf.get_variable("dense1_b", [128])
            x = tf.matmul(x, dense1_w) + dense1_b
            x = tf.nn.relu(x)
            dense2_w = tf.get_variable("dense2_w", [128,64])
            dense2_b = tf.get_variable("dense2_b", [64])
            x = tf.matmul(x, dense2_w) + dense2_b
            x = tf.nn.relu(x)
            final_w = tf.get_variable("final_w", [64,8])
            final_b = tf.get_variable("final_b", [8])
            x = tf.matmul(x, final_w) + final_b
            pi = x
            self.pi = pi

        if take_weights_here is None:
            take_weights_here = {}
            exec(open(os.path.splitext(__file__)[0] + ".weights").read(), take_weights_here)
        self.assigns = [
            (  dense1_w, take_weights_here["weights_dense1_w"]),
            (  dense1_b, take_weights_here["weights_dense1_b"]),
            (  dense2_w, take_weights_here["weights_dense2_w"]),
            (  dense2_b, take_weights_here["weights_dense2_b"]),
            (   final_w, take_weights_here["weights_final_w"]),
            (   final_b, take_weights_here["weights_final_b"]),
        ]

        self.weight_assignment_placeholders = []
        self.weight_assignment_nodes = []
        for var, w in self.assigns:
            ph = tf.placeholder(tf.float32, w.shape)
            self.weight_assignment_placeholders.append(ph)
            self.weight_assignment_nodes.append( tf.assign(var, ph) )

        self.load_weights()

    def load_weights(self):
        feed_dict = {}
        for (var, w), ph in zip(self.assigns, self.weight_assignment_placeholders):
            feed_dict[ph] = w
        tf.get_default_session().run(self.weight_assignment_nodes, feed_dict=feed_dict)

    def act(self, obs_data, cx):
        obs_data = [np.ones((1,)), obs_data]
        obs_data = [obs_data[0], obs_data[1]]
        # Because we need batch dimension, data[None] changes shape from [A] to [1,A]
        a = tf.get_default_session().run(self.pi, feed_dict=dict( (ph,data[None]) for ph,data in zip(self.obs_tuple, obs_data) ))
        return a[0]  # return first in batch


