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

    def run(self, sess, ep, score_input):
        """
        Threading Operation of this worker
        """
        transitions = []

        T = ACWorkerConfig.T

        # Sync this network with the shared network
        self.network.copy_params_from_shared_network(sess)

        for t in range(T):
            policy, v = self.network.get_policy_and_value(sess, self.world.get_state())
            action = choose_action(policy)

            self.world.step(action, certainty = 1.0)

            curr_transition = self.world.get_last_transition()
            curr_transition.append(v)
            transitions.append(curr_transition)

            if self.world.is_terminal():
                # Record score and stuff here
                self.world.reset()
                break

        R = self.network.get_value(sess, self.world.get_state())
        if t != T - 1: # If early termination
            R = 0.0

        # Go in reverse order
        transitions.reverse()

        for s, a, r, sp, v in transitions:
            R = r + ACWorkerConfig.GAMMA * R
            td = R - v

        self.shared_network.apply_gradients(sess,)





