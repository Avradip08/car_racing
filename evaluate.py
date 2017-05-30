import numpy as np
import tensorflow as tf
from world import World
import gym
from ac_network import ActorCriticNetwork as ACN
import sys
from gym import wrappers


def choose_action(policy):
    return np.random.choice(range(len(policy)), p = policy)

def evaluate_single_episode(sess, network, world):
    #world.reset()

    while not world.is_terminal():
        policy, v = network.get_policy_and_value(sess, world.get_state())
        action = choose_action(policy)
        world.step(action, certainty=1.0, train=True)
        world.env.render()

    world.print_stats()
    #print "Total Reward : {}".format(world.real_rewards)
    return world.rewards

def evaluate(sess, network):
    test_env = gym.make("CarRacing-v0")
    test_env = wrappers.Monitor(test_env, '/tmp/car_racing-1')

    test_world = World("f", test_env)
    test_world.reset()
    test_world.env.render()
    L = 100

    total_rewards = 0.0

    print "Start Evaluating..."

    for i in range(L):
        total_rewards += evaluate_single_episode(sess, network, test_world)
        test_world.reset()

    avg_reward = total_rewards / float(L)

    return avg_reward


def load_model(graph, sess, worker_num, score, iter_num):

    path = "./checkpoint/worker"+str(worker_num)+"_"+str(score)+"_iter"+str(iter_num)

    # Build empty graph first
    with graph.as_default():

        #tf.reset_default_graph()
        network = ACN("worker"+str(worker_num))
        network.set_gradients_op()

        #print [v.name for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)]

        saver = tf.train.import_meta_graph(path+".meta")

        saver.restore(sess, path)

        #sess.run(tf.global_variables_initializer())


        #print [v.name for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)]
        #print len([v.name for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="shared")])
        #print sess.run(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="shared")[43])

        all_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="worker"+str(worker_num))

        for v in all_vars:
            print(sess.run(v))

        return network

    return None

def main(argv):
    worker_num = argv[1]
    score = argv[2]
    iter_num = argv[3]

    graph = tf.Graph()
    sess = tf.Session(graph=graph)
    #network = load_model(graph, sess, "./checkpoint/worker3_907_iter29021.meta")
    network = load_model(graph, sess, worker_num, score, iter_num)

    avg_reward = evaluate(sess, network)

    #print avg_reward

if __name__ == "__main__":
    main(sys.argv)
