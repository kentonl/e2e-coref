#!/usr/bin/env python

import os

import tensorflow as tf
import util

if __name__ == "__main__":
  config = util.initialize_from_env()
  report_frequency = config["report_frequency"]
  cluster_config = config["cluster"]
  util.set_gpus()
  cluster = tf.train.ClusterSpec(cluster_config["addresses"])
  server = tf.train.Server(cluster, job_name="ps", task_index=0)
  server.join()
