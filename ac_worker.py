import tensorflow as tf
import numpy as np
from world import World
from ac_network import ActorCriticNetwork as ACN
from config import ACWorkerConfig

class ACWorker(object):
    """
    Each worker owns its own world(env)
    """
    def __init__(self, worker_num, shared_network, device):
        # Create world and reset all states
        self.world = World("CarRacing-v0", render=False)

        self.network = ACN("worker"+str(worker_num), device=device)
        self.shared_network = shared_network
        self._device = device
        self.worker_num = worker_num

        # Prepare for copying network parameters
        self.network.set_copy_params_op(self.shared_network)

    def choose_action(self, policy):
        """
        policy : length nA list of action softmax scores
        """
        return np.random.choice(range(len(policy)), p = policy)

    #TODO: def run(self, sess, ep, score_input): ?
    def run(self, sess):
        """
        Threading Operation of this worker
        """
        transitions = []

        T = ACWorkerConfig.T

        # Sync this network with the shared network
        self.network.copy_params_from_shared_network(sess)

        for t in range(T):
            policy, v = self.network.get_policy_and_value(sess, self.world.get_state())
            action = self.choose_action(policy)

            self.world.step(action, certainty = 1.0)

            curr_transition = self.world.get_last_transition()
            curr_transition.append(v)
            transitions.append(curr_transition)

            if self.world.is_terminal():
                # Record score and stuff here
                self.world.reset()
                break

        R = self.network.get_value(sess, self.world.get_state())
        if t != T - 1: #TODO: If early termination
            R = 0.0

        # Go in reverse order
        transitions.reverse()

        batch_s, batch_a, batch_r, batch_td = [], [], [], []

        for s, a, r, sp, v in transitions:
            R = r + ACWorkerConfig.GAMMA * R
            td = R - v
            one_hot_a = np.zeros(WorldConfig.NUM_ACTIONS)
            one_hot_a[a] = 1

            batch_s.append(s)
            batch_a.append(one_hot_a)
            batch_r.append(R)
            batch_td.append(td)

        feed_dict = {self.network.s: batch_s, self.network.a: batch_a, \
            self.network.td: batch_td, self.network.r: batch_r}
        self.shared_network.apply_gradients(sess, self.network.gradients, feed_dict)
