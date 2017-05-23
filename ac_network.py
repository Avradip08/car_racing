import numpy as np
import tensorflow as tf
from config import ACNetworkConfig, WorldConfig

class ActorCriticNetwork(object):
    """
    Main network for the Actor Critic Algorithm
    """
    def __init__(self, device="/cpu:0", scope):
        self._device = device
        self._scope = scope
        self._build_graph()

    def _get_initializer(self, layer_type):
        #if layer_type == "conv":
        #    return tf.contrib.layers.xavier_initializer_conv2d()
        #elif layer_type == "fc":
        #    return tf.

        return tf.contrib.layers.xavier_initializer()

    def _build_graph(self):
        # Unpack config for readability
        H = W = WorldConfig.WORLD_SIZE
        C = WorldConfig.NUM_FRAMES_IN_STATE
        nA = WorldConfig.NUM_ACTIONS
        dH = ACNetworkConfig.FEATURE_DIM

        with tf.device(self._device):
            with tf.variable_scope(self._scope):

                # Define Graph entry points
               self.s = tf.placeholder(tf.float32, [None, H, W, C])
               self.a = tf.placeholder(tf.float32, [None, nA])
               self.td = tf.placeholder(tf.float32, [None]) # (R - V)
               self.r = tf.placeholder(tf.float32, [None])

                # Start Drawing Graph
                # Build CNN Feature Extraction Graph
                conv_w_init = self._get_initializer('conv')
                conv1 = tf.layers.conv2d(self.s, 16, 8, strides=4, activation=tf.nn.relu,\
                                     kernel_initializer=conv_w_init)
                conv2 = tf.layers.conv2d(self.s, 32, 3, strides=2, activation=tf.nn.relu,\
                                     kernel_initializer=conv_w_init)
                conv2_flatten = tf.contrib.layers.flatten(conv2)

                # Build Policy and Value Graph
                w_init = self._get_initializer('fc')
                features = tf.contrib.layers.fully_connected(conv2_flatten, num_outputs=dH, weights_initializer=w_init)

                pi_scores = tf.contrib.layers.fully_connected(features, num_outputs=nA, activation_fn=None, weights_initializer=w_init)
                v = tf.contrib.layers.fully_connected(features, num_outputs=1, activation_fn=None, weights_initializer=w_init)

                self.policy = tf.nn.softmax(pi_scores)
                self.v = tf.reshape(v, [-1])

                # Attach Loss and Optimizers
                log_policy = tf.log(tf.clip_by_value(self.policy, 1e-20, 1.0))
                entropy = -tf.reduce_sum(self.policy * log_policy, axis=1)

                p_loss = -tf.reduce_sum(tf.reduce_sum(tf.mul(log_policy, self.a), axis=1) * self.td + entropy * 0.01)

                v_loss = 0.5 * tf.nn.l2_loss(self.r - self.v)

                self.loss = p_loss + v_loss

    def get_vars(self):

    def get_policy(self, sess, state):
        feed_dict = {
            self.s : [state]
        }
        return sess.run(self.policy, feed_dict=feed_dict)[0]

    def get_value(self, sess, state):
        feed_dict = {
            self.s : [state]
        }
        return sess.run(self.v, feed_dict=feed_dict)[0]



