#!/bin/bash

curl -O http://lsz-gpu-01.cs.washington.edu/resources/coref/char_vocab.english.txt

ckpt_file=c2f_final.tgz
curl -O http://lsz-gpu-01.cs.washington.edu/resources/coref/$ckpt_file
mkdir -p logs
tar -xzvf $ckpt_file -C logs
rm $ckpt_file
