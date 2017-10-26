#!/usr/bin/env python

import os
import sys
sys.path.append(os.getcwd())

import tensorflow as tf
import util

if __name__ == "__main__":
  if len(sys.argv) > 1:
    name = sys.argv[1]
  else:
    name = os.environ["EXP"]
  util.set_gpus()
  cluster_config = util.get_config("experiments.conf")[name]["cluster"]
  cluster = tf.train.ClusterSpec(cluster_config["addresses"])
  server = tf.train.Server(cluster, job_name="ps", task_index=0)
  server.join()
