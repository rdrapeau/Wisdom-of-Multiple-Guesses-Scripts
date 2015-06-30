# Wisdom of Multiple Guesses Scripts
Run `python generate_figures.py` to generate the figures in the paper.

## About the Data

### Exeriments

See the [paper](http://www.stanford.edu/~jugander/papers/ec15-multipleguesses.pdf) for more details about the 2 experiments that were run. The labels for each are:

Mulitple Guesses Experiment = mge

Interval Comparison Experiment = ice

### {experiment label}_{#}guess.csv

This contains the guesses and answers for a section of the game run in  {experiment}. The data contain (in csv format) the user id, the index the image was shown, the time it took to complete the image, followed by the {#} guesses the user made, and then the correct answer.

Special: [ice_interval.csv](ice_data/ice_interval.csv) contain the data for the interval game, which is exactly the same as [mge_2guess.csv](mge_data/mge_2guess.csv).

### {experiment label}_{users}.csv

Thie file contains the data relating to the user. Namely, the user id and the score the user finished the experiment with. For more details about the scoring functions used, please see the paper.
