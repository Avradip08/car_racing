import tensorflow as tf
import numpy as np
from ac_network import ActorCriticNetwork as ACN
from config import ACWorkerConfig, WorldConfig

import sys

class ACWorker(object):
    """
    Each worker owns its own world(env)
    """
    def __init__(self, worker_num, shared_network, device, filewriter, saver):
        # Create world and reset all states
        #self.world = World("CarRacing-v0", render=False)

        self.network = ACN("worker"+str(worker_num), device=device)
        self.shared_network = shared_network
        self._device = device
        self.worker_num = worker_num

        # Prepare for gradient calc
        self.network.set_gradients_op()

        # Prepare for copying network parameters
        self.network.set_copy_params_op(self.shared_network)

        #if worker_num == 0:
        self.network.add_summaries()
        self.filewriter = filewriter
        self.saver = saver

        self.guiness = 0.0

    def choose_action(self, policy):
        """
        policy : length nA list of action softmax scores
        """
        return np.random.choice(range(len(policy)), p = policy)

    #TODO: def run(self, sess, ep, score_input): ?
    def run(self, sess, iteration):
        """
        Threading Operation of this worker
        """
        transitions = []

        T = ACWorkerConfig.T

        # Sync this network with the shared network
        self.network.copy_params_from_shared_network(sess)

        #print "Thread {} coppied parameters from the shared network".format(self.worker_num)

        for t in range(T):
            policy, v = self.network.get_policy_and_value(sess, self.world.get_state())
            action = self.choose_action(policy)

            self.world.step(action, certainty = 1.0)

            curr_transition = self.world.get_last_transition()
            curr_transition.append(v)
            transitions.append(curr_transition)

            if self.world.is_terminal():
                if self.world.max_real_rewards >= self.guiness:
                    self.guiness = self.world.max_real_rewards
                    #self.saver.save(sess, "./checkpoint/"+str(int(self.guiness))+"_iter"+str(iteration))
                    self.network.save(sess, str(int(self.guiness))+"_iter"+str(iteration))
                if self.world.num_tiles >= 150:
                    self.network.save(sess, str(int(self.world.max_real_rewards))+"_iter"+str(iteration))

                self.world.print_stats()
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

        # Calculate the gradientsm loss and the global norm
        feed_dict = {self.network.s: np.stack(batch_s), self.network.a: batch_a, \
            self.network.td: batch_td, self.network.r: batch_r, self.network.ep_score:self.world.real_rewards}
        loss, grads, global_norm, summary = self.network.get_gradients_and_loss(sess, feed_dict)


        # Attach summary only for thread 0
        if self.worker_num == 0:
            self.filewriter.add_summary(summary, iteration)

        # Apply gradients to the shared network
        feed_dict = {}
        for p in self.shared_network.grads_placeholders:
            feed_dict[p] = find_grad(p, grads)
        self.shared_network.apply_gradients(sess, feed_dict)

def find_grad(p, grad_list):
    for grad in grad_list:
        if grad.shape == p.shape:
            return grad

    return None
