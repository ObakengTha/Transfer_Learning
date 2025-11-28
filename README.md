This project implements Genetic Programming (GP) for regression on the 227 CPU Performance Dataset, with optional transfer learning using evolved GP trees. The system uses symbolic regression to learn mathematical expressions that predict the target CPU performance variable.

The project includes:

A full GP implementation from scratch

Tree-based symbolic regression

Custom fitness evaluation

Crossover & mutation operators

Early stopping

Saving the best evolved model (best_tree.pk5)

Pre-processing and cleaning of the numeric dataset

Project Overview

Genetic Programming (GP) is used to evolve symbolic mathematical expressions that can predict the target CPU performance using features extracted from a dataset (227Dataset.xlsx).

The GP system learns a regression model using:

Binary expression trees

Function set: +, -, *, /, sin, log, exp

Terminal set: dataset features + random constants

Tournament selection

Subtree crossover

Point mutation

Depth-bounded random tree generation

The best evolved model is saved using pickle for reuse or transfer learning.
