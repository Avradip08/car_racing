import numpy as np
import tensorflow as tf
from world import World
import gym

def choose_action(policy):
    return np.random.choice(range(len(policy)), p = policy)

def evaluate_single_episode(sess, network, world):
    #world.reset()

    while not world.is_terminal():
        policy, v = network.get_policy_and_value(sess, world.get_state())
        action = choose_action(policy)

        world.step(action, certainty=1.0)

    print "Total Reward : {}".format(world.rewards)
    return world.rewards

def evaluate(sess, network):
    test_env = gym.make("CarRacing-v0")
    test_world = World("f", test_env)
    test_world.reset()
    L = 1

    total_rewards = 0.0

    print "Start Evaluating..."

    for i in range(L):
        total_rewards += evaluate_single_episode(sess, network, test_world)
        test_world.reset()

    avg_reward = total_rewards / float(L)

    return avg_reward

