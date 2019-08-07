from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import json
import io

def get_char_vocab(input_filenames, output_filename):
  vocab = set()
  for filename in input_filenames:
    with open(filename) as f:
      for line in f.readlines():
        for sentence in json.loads(line)["sentences"]:
          for word in sentence:
            vocab.update(word)
  vocab = sorted(list(vocab))
  with io.open(output_filename, mode="w", encoding="utf8") as f:
    for char in vocab:
      f.write(char)
      f.write(u"\n")
  print("Wrote {} characters to {}".format(len(vocab), output_filename))

def get_char_vocab_language(language):
  get_char_vocab(["{}.{}.jsonlines".format(partition, language) for partition in ("train", "dev", "test")], "char_vocab.{}.txt".format(language))

get_char_vocab_language("english")
get_char_vocab_language("chinese")
get_char_vocab_language("arabic")
