#!/bin/bash

for i in `seq 0 9`;
do
	tmux new-session -d "taskset -c 4*i,4*i+1,4*i+2,4*i+3 python train.py --algo sac --env Hopper-v3 --eval-episodes 10 --eval-freq 10000 > ~/results/sacHopv3.txt 2>&1"
done
