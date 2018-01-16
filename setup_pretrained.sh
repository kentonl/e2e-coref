#!/bin/bash

curl -O https://lil.cs.washington.edu/coref/char_vocab.english.txt

curl -O https://lil.cs.washington.edu/coref/final.tgz
mkdir -p logs
tar -xzvf final.tgz -C logs
rm final.tgz
