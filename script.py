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


# Create a subsequent network
worker_threads = []


graph = tf.Graph()

# Create a shared network
with graph.as_default():
    shared_network= ACN("shared")
    shared_network.set_gradients_op()

    for i in range(A3CConfig.NUM_THREADS):
        thread = ACWorker(i, shared_network, "/cpu:0")
        worker_threads.append(thread)

    var_init = tf.global_variables_initializer()

sess = tf.Session(graph=graph)
sess.run(var_init)

filewriter = tf.summary.FileWriter("./results", sess.graph)

def run_single_thread(worker_num, sess, env):
    num_iteration = 0
    worker = worker_threads[worker_num]
    worker.env = env
    worker.world = World("f",worker.env)
    worker.world.reset()

    while True:
        if num_iteration >= ACWorkerConfig.MAX_ITERATIONS: break
        worker.run(sess)
        print "thread loop"
        num_iteration += 1

def run_threads(worker_threads, sess):
    threads = []

    # Allocate Thread resources per worker
    for i in range(A3CConfig.NUM_THREADS):
        print "Creating processing thread of " + str(i)
        env = gym.make("CarRacing-v0")
        threads.append(threading.Thread(target=run_single_thread, args=(i, sess, env)))
        #threads.append(Process(target=run_single_thread, args=(i, sess, env)))

    # Fire all threads
    for t in threads:
        t.start()
    return threads

# Run threads
processing_threads = run_threads(worker_threads, sess)

# Finish
for t in processing_threads:
    t.join()
