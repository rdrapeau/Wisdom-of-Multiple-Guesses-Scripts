# Wisdom of Multiple Guesses Scripts

This repository is a compendium to the folowing paper:

- J. Ugander, R. Drapeau, C. Guestrin, "The Wisdom of Multiple Guesses", EC 2015. [PDF](http://www.stanford.edu/~jugander/papers/ec15-multipleguesses.pdf)

The paper contains two experiments comparing different scoring rules for eliciting uncertainty in a simple "Dot Guessing Game" run on Amazon Mechanical Turk.

### Experiments

Results from the two experiments are labelled as:

- Mulitple Guesses Experiment = mge

- Interval Comparison Experiment = ice

### {experiment label}_{#}guess.csv

These CSV file contain the guesses and answers for a section of the game run in each of {experiment label}. The files contain a participant id, the index position where the image was shown, the time it took to complete the image (in milliseconds), followed by the {#} guesses the participant made, and then the correct answer.

### {experiment label}_{users}.csv

Thie file contains the scores that each participant finished the experiment with. For more details about the scoring functions used, please see the paper.

## Running Instructions

Running `python generate_figures.py` will generate the result figures in the paper.

Please use [Python 2.7](https://www.python.org/download/releases/2.7/) and [matplotlib](http://matplotlib.org/) version 1.3.1 or higher.

