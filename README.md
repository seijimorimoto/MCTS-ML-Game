# MCTS-ML-Game

Usage of Monte Carlo Tree Search (MCTS) with a Decision Tree (DT) for the playout policy in the
Othello board game.

This repository is structured as follows:

- code/: contains the MCTS algorithm used, the code for collecting data from it and training
  decision tree models, and the code for evaluating agents by making them play against each other
  while recording their win rate.
- data/: a directory containing data collected from different executions of the MCTS algorithm.
- models/: contains the decision tree models generated based on the data collected.
- results/: includes the results of different sets of matches between different agents.
