# End-to-end Neural Coreference Resolution

### Introduction
This repository contains the code for replicating results from

* [End-to-end Neural Coreference Resolution](https://homes.cs.washington.edu/~kentonl/pub/lhlz-emnlp.2017.pdf)
* [Kenton Lee](https://homes.cs.washington.edu/~kentonl), [Luheng He](https://homes.cs.washington.edu/~luheng), [Mike Lewis](https://research.fb.com/people/lewis-mike) and [Luke Zettlemoyer](https://www.cs.washington.edu/people/faculty/lsz)
* In Proceedings of the Conference on Empirical Methods in Natural Language Process (EMNLP), 2017

A demo of the code can be found here: http://www.kentonl.com/e2e-coref.

### Requirements
* Python 2.7
  * TensorFlow 1.0.0
  * pyhocon (for parsing the configurations)
  * NLTK (for sentence splitting and tokenization in the demo)

### Setting Up

* Download pretrained word embeddings and build custom kernels by running `setup_all.sh`.
  * There are 3 platform-dependent ways to build custom TensorFlow kernels. Please comment/uncomment the appropriate lines in the script.
* Run one of the following:
  * To use the pretrained model only, run `setup_pretrained.sh`
  * To train your own models, run `setup_training.sh`
    * This assumes access to OntoNotes 5.0. Please edit the `ontonotes_path` variable.

## Training Instructions

* Experiment configurations are found in `experiments.conf`
* Choose an experiment that you would like to run, e.g. `best`
* For a single-machine experiment, run the following two commands:
  * `python singleton.py <experiment>`
  * `python evaluator.py <experiment>`
* For a distributed multi-gpu experiment, edit the `cluster` property of the configuration and run the following commands:
  * `python parameter_server.py <experiment>`
  * `python worker.py <experiment>` (for every worker in your cluster)
  * `python evaluator.py <experiment>` (on the same machine as your first worker)
* Results are stored in the `logs` directory and can be viewed via TensorBoard.
* For final evaluation of the checkpoint with the maximum dev F1:
  * Run `python test_single.py <experiment>` for the single-model evaluation.
  * Run `python test_ensemble.py <experiment1> <experiment2> <experiment3>...` for the ensemble-model evaluation.

## Demo Instructions

* For the command-line demo with the pretrained model:
  * Run `python demo.py final`
* For the web demo with the pretrained model:
  * Run `python demo.py final 8080`
  * Edit the URL at the end of `docs/main.js` to point to the demo location, e.g. `localhost:8080`
  * Open `docs/index.html` in a web browser.
* To run the demo with other experiments, replace `final` with your configuration name.

## Batched Prediction Instructions

* Create a file where each line is in the following json format (make sure to strip the newlines so each line is well-formed json):
```
{
  "clusters": [],
  "doc_key": "nw",
  "sentences": [["This", "is", "the", "first", "sentence", "."], ["This", "is", "the", "second", "."]],
  "speakers": [["spk1", "spk1", "spk1", "spk1", "spk1", "spk1"], ["spk2", "spk2", "spk2", "spk2", "spk2"]]
}
```
  * `clusters` should be left empty and is only used for evaluation purposes.
  * `doc_key` indicates the genre, which can be one of the following: `"bc", "bn", "mz", "nw", "pt", "tc", "wb"`
  * `speakers` indicates the speaker of each word. These can be all empty strings if there is only one known speaker.
* Change the value of `eval_path` in the configuration file to the path to this new file.
* Run `python decoder.py <experiment> <output_file>`, which outputs the original file extended with annotations of the predicted clusters, the top spans, and the head attention scores.
* To visualize the predictions, place the output file in the `viz` directory and run `run.sh`. This will run a web server hosting the files in the `viz` directory. If run locally, it can be reached at `http://localhost:8080?path=<output_file>`

## Other Quirks

* It does not use GPUs by default. Instead, it looks for the `GPU` environment variable, which the code treats as shorthand for `CUDA_VISIBLE_DEVICES`.
* The evaluator should not be run on GPUs, since evaluating full documents does not fit within GPU memory constraints.
* The training runs indefinitely and needs to be terminated manually. The model generally converges at about 400k steps and within 48 hours.
* On some machines, the custom kernels seem to have compatibility issues with virtualenv. If you are using virtualenv and observe segmentation faults, trying running the experiments without virtualenv.
