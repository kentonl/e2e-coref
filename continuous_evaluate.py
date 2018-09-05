#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import time
import shutil

import tensorflow as tf
import coref_model as cm
import util

def copy_checkpoint(source, target):
  for ext in (".index", ".data-00000-of-00001"):
    shutil.copyfile(source + ext, target + ext)

if __name__ == "__main__":
  config = util.initialize_from_env()
  model = cm.CorefModel(config)

  saver = tf.train.Saver()
  log_dir = config["log_dir"]
  writer = tf.summary.FileWriter(log_dir, flush_secs=20)
  evaluated_checkpoints = set()
  max_f1 = 0
  checkpoint_pattern = re.compile(".*model.ckpt-([0-9]*)\Z")

  with tf.Session() as session:
    while True:
      ckpt = tf.train.get_checkpoint_state(log_dir)
      if ckpt and ckpt.model_checkpoint_path and ckpt.model_checkpoint_path not in evaluated_checkpoints:
        print("Evaluating {}".format(ckpt.model_checkpoint_path))

        # Move it to a temporary location to avoid being deleted by the training supervisor.
        tmp_checkpoint_path = os.path.join(log_dir, "model.tmp.ckpt")
        copy_checkpoint(ckpt.model_checkpoint_path, tmp_checkpoint_path)

        global_step = int(checkpoint_pattern.match(ckpt.model_checkpoint_path).group(1))
        saver.restore(session, ckpt.model_checkpoint_path)

        eval_summary, f1 = model.evaluate(session)

        if f1 > max_f1:
          max_f1 = f1
          copy_checkpoint(tmp_checkpoint_path, os.path.join(log_dir, "model.max.ckpt"))

        print("Current max F1: {:.2f}".format(max_f1))

        writer.add_summary(eval_summary, global_step)
        print("Evaluation written to {} at step {}".format(log_dir, global_step))

        evaluated_checkpoints.add(ckpt.model_checkpoint_path)
        sleep_time = 60
      else:
        sleep_time = 10
      print("Waiting for {} seconds before looking for next checkpoint.".format(sleep_time))
      time.sleep(sleep_time)
