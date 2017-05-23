import numpy as np
import tensorflow as tf
from config import ACNetworkConfig, WorldConfig, A3CConfig
import threading
import tensorflow as tf
import numpy as np
from world import World
from ac_network import ActorCriticNetwork as ACN
from config import ACWorkerConfig
from ac_worker import ACWorker


# Create a shared network
shared_network= ACN("shared")

# Create a subsequent network
worker_threads = []
for i in range(A3CConfig.NUM_THREADS):
    thread = ACWorker(i, shared_network, "/cpu:0")
    worker_threads.append(thread)

sess = tf.Session()
sess.run(tf.global_variables_initializer())


print "Workers: ", worker_threads

def run_single_thread(worker, sess):
    num_iteration = 0
    while True:
        if num_iteration >= ACWorkerConfig.MAX_ITERATIONS: break
        print num_iteration
        worker.run(sess)
        num_iteration += 1

def run_threads(worker_threads, sess):
    threads = []
    for i in range(A3CConfig.NUM_THREADS):
        print "Creating processing thread of " + str(i)
        threads.append(threading.Thread(target=run_single_thread, args=(worker_threads[i], sess)))
    for t in threads:
        t.start()
    return threads

# Run threads
processing_threads = run_threads(worker_threads, sess)

# Finish
for t in processing_threads:
    t.join()
