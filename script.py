import os
import sys
import time

import numpy as np
from config import ACNetworkConfig, WorldConfig, A3CConfig, ACWorkerConfig
import threading
import tensorflow as tf
import numpy as np
from ac_network import ActorCriticNetwork as ACN
from ac_worker import ACWorker
import gym
from world import World


# Create a shared network
shared_network= ACN("shared")

# Create a subsequent network
worker_threads = []

class ThreadData:
    worker_threads = worker_threads


for i in range(A3CConfig.NUM_THREADS):
    thread = ACWorker(i, shared_network, "/cpu:0")
    ThreadData.worker_threads.append(thread)

sess = tf.Session()
#sess = tf.Session(config=tf.ConfigProto(log_device_placement=False,\
                                                   #allow_soft_placement=True))
sess.run(tf.global_variables_initializer())

print tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

print "Workers: ", worker_threads

def run_single_thread(td, worker_num, sess, env):
    num_iteration = 0


    worker = td.worker_threads[worker_num]
    #worker.env = gym.make("CarRacing-v0")
    worker.env = env
    worker.world = World("f",worker.env)

    #worker = td.worker_threads[worker_num]
    #worker.world = world

    #from world import World

    worker.world.reset()

    #worker.world = World("CarRacing-v0", render=False)
    print "Thread Starting : {}".format(str(worker_num))
    while True:
        if num_iteration >= ACWorkerConfig.MAX_ITERATIONS: break
        worker.run(sess)
        num_iteration += 1

def run_threads(worker_threads, sess):
    threads = []
    for i in range(A3CConfig.NUM_THREADS):
        print "Creating processing thread of " + str(i)
        #worker_threads[i].world = World("CarRacing-v0", render=False)
        #world = World("CarRacing-v0", render=False)
        world = None
        env = gym.make("CarRacing-v0")
        #env.reset()
        #env.render()
        #env.reset()
        threads.append(threading.Thread(target=run_single_thread, args=(ThreadData,i, sess, env)))
    for t in threads:
        t.start()
        #time.sleep(2)
    return threads

# Run threads
processing_threads = run_threads(worker_threads, sess)

# Finish
for t in processing_threads:
    t.join()
