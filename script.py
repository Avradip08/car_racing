import os
import sys
import time

import numpy as np
from config import ACNetworkConfig, WorldConfig, A3CConfig, ACWorkerConfig
import threading
from multiprocessing import Process

import tensorflow as tf
import numpy as np
from ac_network import ActorCriticNetwork as ACN
from ac_worker import ACWorker
import gym
from world import World

from evaluate import evaluate


# Create a subsequent network
worker_threads = []


graph = tf.Graph()
shared_network = None

saver = None
sess = tf.Session(graph=graph)

filewriter = tf.summary.FileWriter("./results", sess.graph)


# Create a shared network
with graph.as_default():
    shared_network= ACN("shared")
    shared_network.set_gradients_op()
    #shared_network.add_summaries()

    #saver = tf.train.Saver(shared_network.all_vars)
    saver = None

    # Create worker networks
    for i in range(A3CConfig.NUM_THREADS):
        thread = ACWorker(i, shared_network, "/cpu:0", filewriter, saver)
        worker_threads.append(thread)

    var_init = tf.global_variables_initializer()


sess.run(var_init)



def run_single_thread(worker_num, sess, env):
    num_iteration = 0
    worker = worker_threads[worker_num]

    worker.env = env
    worker.world = World("f",worker.env)
    worker.world.reset()

    while True:
        if num_iteration >= ACWorkerConfig.MAX_ITERATIONS: break
        worker.run(sess, num_iteration)
        num_iteration += 1

    # Delete worlds
    worker.world = None

def run_threads(worker_threads, sess, iteration):
    threads = []

    # Allocate Thread resources per worker
    for i in range(A3CConfig.NUM_THREADS):
        env = gym.make("CarRacing-v0")
        env._max_episode_steps = 3000
        #print "generate environgment for thread {}, id {}".format(str(i), str(id(env)))
        threads.append(threading.Thread(target=run_single_thread, args=(i, sess, env)))

    # Fire all threads
    for t in threads:
        t.start()

    return threads

iteration = 0
while True:
    if iteration >= 10: break
    processing_threads = run_threads(worker_threads, sess, iteration)

    # Finish
    for t in processing_threads:
        t.join()

    #saver.save(sess, ACNetworkConfig.SAVE_PATH+str(iteration))

    avg_reward = evaluate(sess, shared_network)
    print "Average Reward : {}".format(str(avg_reward))

    iteration += 1
