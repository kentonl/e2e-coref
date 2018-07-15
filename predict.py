from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import json

import tensorflow as tf
import coref_model as cm
import util

if __name__ == "__main__":
  config = util.initialize_from_env()

  # Predictions will be written to this file in .jsonlines format.
  output_filename = sys.argv[2]

  model = cm.CorefModel(config)
  model.load_eval_data()

  with tf.Session() as session:
    model.restore(session)

    with open(output_filename, "w") as f:
      for example_num, (tensorized_example, example) in enumerate(model.eval_data):
        feed_dict = {i:t for i,t in zip(model.input_tensors, tensorized_example)}
        _, _, _, top_span_starts, top_span_ends, top_antecedents, top_antecedent_scores = session.run(model.predictions, feed_dict=feed_dict)
        predicted_antecedents = model.get_predicted_antecedents(top_antecedents, top_antecedent_scores)
        example["predicted_clusters"], _ = model.get_predicted_clusters(top_span_starts, top_span_ends, predicted_antecedents)

        f.write(json.dumps(example))
        f.write("\n")
        if example_num % 100 == 0:
          print("Decoded {} examples.".format(example_num + 1))
