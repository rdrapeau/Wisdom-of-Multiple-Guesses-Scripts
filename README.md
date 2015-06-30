# Wisdom of Multiple Guesses Scripts

See the [paper](http://www.stanford.edu/~jugander/papers/ec15-multipleguesses.pdf) for more details about the 2 experiments that were run.

## About the Data

### Exeriments

The labels for each are:

Mulitple Guesses Experiment = mge

Interval Comparison Experiment = ice

### {experiment label}_{#}guess.csv

This contains the guesses and answers for a section of the game run in  {experiment}. The data contain (in csv format) the user id, the index the image was shown, the time it took to complete the image, followed by the {#} guesses the user made, and then the correct answer.

Special: [ice_interval.csv](ice_data/ice_interval.csv) contain the data for the interval game, which is exactly the same as [mge_2guess.csv](mge_data/mge_2guess.csv).

### {experiment label}_{users}.csv

Thie file contains the data relating to the user. Namely, the user id and the score the user finished the experiment with. For more details about the scoring functions used, please see the paper.

## Running Instructions

Run `python generate_figures.py` to generate the figures in the paper.

Please use [Python 2.7](https://www.python.org/download/releases/2.7/) and [matplotlib](http://matplotlib.org/) version 1.3.1 or higher.


