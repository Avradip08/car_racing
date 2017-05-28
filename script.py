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

# Create a shared network
with graph.as_default():
    shared_network= ACN("shared")
    shared_network.set_gradients_op()
    shared_network.add_summaries()

    # Create worker networks
    for i in range(A3CConfig.NUM_THREADS):
        thread = ACWorker(i, shared_network, "/cpu:0")
        worker_threads.append(thread)

    var_init = tf.global_variables_initializer()

sess = tf.Session(graph=graph)

filewriter = tf.summary.FileWriter("./results", sess.graph)

sess.run(var_init)



def run_single_thread(worker_num, sess, env):
    num_iteration = 0
    worker = worker_threads[worker_num]

    worker.env = env
    worker.world = World("f",worker.env)
    worker.world.reset()

    while True:
        if num_iteration >= ACWorkerConfig.MAX_ITERATIONS: break
        worker.run(sess)
        num_iteration += 1

    # Delete worlds
    worker.world = None

def run_threads(worker_threads, sess, iteration):
    threads = []

    # Allocate Thread resources per worker
    for i in range(A3CConfig.NUM_THREADS):
        print "generate environgment for thread {}".format(str(i))
        env = gym.make("CarRacing-v0")
        threads.append(threading.Thread(target=run_single_thread, args=(i, sess, env)))

    # Fire all threads
    for t in threads:
        t.start()

    return threads

iteration = 0
while True:
    if iteration >= 5: break
    print "Iteration Start"
    processing_threads = run_threads(worker_threads, sess, iteration)

    # Finish
    for t in processing_threads:
        t.join()

    print "Iteration Over"
    avg_reward = evaluate(sess, shared_network)
    print "Average Reward : {}".format(str(avg_reward))

    iteration += 1
