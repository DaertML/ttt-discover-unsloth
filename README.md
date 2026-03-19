# ttt-discover-unsloth
An implementation of the TTT Discover paper using unsloth. The ideas from this work come from this video and this paper:
- https://www.youtube.com/watch?v=JgB_ywOmGCc&t=2013s
- https://arxiv.org/abs/2601.16175

# Introduction
The ideas for TTT (Test Time Training) resemble a different paradigm to classic RL or continual learning; in which the objective is to get better models, neglecting the exploitation phase to get the best answer, instead of following on a population of average great answers.

This let's the model to get better at the task at hand, instead of looking forward getting better in a general manner. The great part of the "meta-learning" kind of behavior that is used to achieve this, is that you can plug any RL algorithm that you wish to the mix.

There are two main elements that will govern the improvement of the solution: the search and learn process; being the search function the one that explores further and the learn process the one that applies all the discovered improvements back to model weights.

How to run:

python run.py \
  --model unsloth/Qwen2.5-Coder-1.5B-Instruct-bnb-4bit \
  --steps 5 \
  --rollouts 4

