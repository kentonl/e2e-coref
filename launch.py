#!/usr/bin/env python

import sys
import subprocess as sp

import util

def screen(py_script, name, args):
  bash = "source ~/.bashrc; python {} {}; exec bash".format(py_script, " ".join(str(a) for a in args))
  command = ["screen", "-dmS", name, "bash", "-c", bash]
  print " ".join(command)
  sp.call(command)

if __name__ == "__main__":
  exp_name = sys.argv[1]
  util.mkdirs("logs")
  cluster_config = util.get_config("experiments.conf")[exp_name]["cluster"]
  screen("parameter_server.py", "ps", [exp_name])
  screen("evaluator.py", "eval", [exp_name])
  for i, _ in enumerate(cluster_config["addresses"]["worker"]):
    screen("worker.py", "w{}".format(i), [exp_name, i])
