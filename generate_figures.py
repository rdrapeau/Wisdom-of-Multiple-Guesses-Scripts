# drapeau@cs.washington.edu

from pylab import *
import random, scipy.stats

TRAINING_INDICES = set([1, 2, 13, 14, 25, 26])
PLOT_SIZE = (21, 12)

NUM_BOOTSTRAP_SAMPLES = 1000 # 10,000 was used in the paper


'''
Reads in a CSV and splits each line by ','. The header line is removed.
'''
def readCSV(filepath):
    with open(filepath, 'r') as f:
        return [tuple(line.split(',')) for line in f.read().splitlines()][1:]


'''
Formats the scores as a dictionary indexed by the user's id.
'''
def format_scores(raw_line):
    return {int(user_id) : int(score) for (user_id, score) in raw_line}


'''
Combines multiple csv log files into a single dictionary object.
'''
def getGame(labels_and_files):
    result = {}
    for label in labels_and_files:
        entries = readCSV(labels_and_files[label])
        result[label] = []

        for entry in entries:
            result[label].append([int(item) for item in entry])

    return result


'''
Cleans the game by removing unincentivized users (ones who did not finish
the game with a positive score).
'''
def clean_game(game, scores):
    cleaned_game = {}
    for section in game:
        entries = game[section]
        cleaned_game[section] = []

        for entry in entries:
            user_id = entry[0]
            if scores.get(user_id, 0) != 0:
                cleaned_game[section].append(entry)

    return cleaned_game


'''
Indexes the game into {game : {dot_count : [guesses]}}. Throws away training data.
'''
def index_by_dot_count(game):
    result = {section : {} for section in game}
    dot_counts = set()

    for section in game:
        for entry in game[section]:
            dot_count = entry[-1]
            index = entry[1]
            if index not in TRAINING_INDICES:
                dot_counts.add(dot_count)
                if dot_count not in result[section]:
                    result[section][dot_count] = []

                result[section][dot_count].append(tuple(entry[3:-1]))

    return result, dot_counts


'''
Indexes the game into {user id : {section : [guesses]}}. Throws away training data.
'''
def index_by_user_id(game):
    result = {}
    counts = {}

    for section in game:
        for entry in game[section]:
            if entry[1] in TRAINING_INDICES:
                continue

            user_id = entry[0]
            if user_id not in result:
                result[user_id] = {i : [] for i in game}

            result[user_id][section].append(tuple(entry[3:]))
            counts[user_id] = counts.get(user_id, 0) + 1

    for user in counts:
        if counts[user] != 20:
            print counts[user], user

    return result


def sample_with_replacement(samples, num_samples):
    return [random.choice(samples) for _ in xrange(num_samples)]


'''
Returns the weighted median in samples given the weights for each sample.
'''
def get_weighted_median(samples, weights):
    # Normalize the weights
    total = np.sum(weights)
    weights = [weight / float(total) for weight in weights]

    zipped = zip(samples, weights)
    zipped.sort(key = lambda x : x[0])

    samples = [entry[0] for entry in zipped]
    weights = np.cumsum([entry[1] for entry in zipped])

    for i, weight in enumerate(weights):
        if weight >= 0.50:
            return samples[i]


def get_mse(estimates, actual):
    result = np.sum([(estimate - actual) ** 2 for estimate in estimates]) / float(len(estimates))
    return result


def figure_3(mg_game):
    figure(figsize = PLOT_SIZE)
    estimators_1_2_guess(mg_game, 79, 50)
    tight_layout()


'''
Figure 3 Right
'''
def estimators_1_2_guess(game, dot_count, sample_size):
    data, dot_counts = index_by_dot_count(game)
    dot_counts = sorted(list(dot_counts))
    mean_mse_ratios = []
    median_mse_ratios = []

    # Data for the subfigure
    means_subfigure = {}
    medians_subfigure = {}

    for image in dot_counts:
        print 'Bootstrapping Image (' + str(image) + ') for Figure 3'

        guesses_1 = data['1'][image]
        guesses_2 = data['2'][image]
        means = {'1' : [], '2' : []}
        medians = {'1' : [], '2' : []}

        for _ in xrange(NUM_BOOTSTRAP_SAMPLES):
            guesses_1_sample = sample_with_replacement(guesses_1, sample_size)

            means['1'].append(np.mean(guesses_1_sample))
            medians['1'].append(np.median(guesses_1_sample))

            gusses_2_sample = sample_with_replacement(guesses_2, sample_size)
            average_guesses = [np.mean(guess) for guess in gusses_2_sample]
            weights = [1.0 / (np.max(guess) - np.min(guess)) for guess in gusses_2_sample]
            squared_weights = [weight ** 2 for weight in weights]

            weighted_mean = np.sum([squared_weights[i] * average_guesses[i]
                for i in xrange(sample_size)]) / np.sum(squared_weights)
            weighted_median = get_weighted_median(average_guesses, weights)

            means['2'].append(weighted_mean)
            medians['2'].append(weighted_median)

        mean_mse = {'1' : get_mse(means['1'], image), '2' : get_mse(means['2'], image)}
        median_mse = {'1' : get_mse(medians['1'], image), '2' : get_mse(medians['2'], image)}

        mean_mse_ratios.append(mean_mse['2'] / mean_mse['1'])
        median_mse_ratios.append(median_mse['2'] / median_mse['1'])

        # Save for subfigure
        if image == dot_count:
            means_subfigure['1'] = means['1']
            means_subfigure['2'] = means['2']
            medians_subfigure['1'] = medians['1']
            medians_subfigure['2'] = medians['2']

    # PLOT MEDIANS
    subplot2grid((100, 3), (0, 1), colspan = 2, rowspan = 50)
    x = range(len(dot_counts))
    images = [str(image) for image in dot_counts]
    for i, ratio in enumerate(median_mse_ratios):
        plot(i, ratio, 'o', color = 'black', ms = 8)

    axhline(y = 1, color = 'gray')
    xlim(-1, len(dot_counts))
    xticks(x, images)
    yscale('log')
    xlabel('Dot Count')
    ylabel('Median: 2 guess MSE / 1 guess MSE')
    ylim(ymin =10 ** (-2.5), ymax = 10 ** 2.5)
    axvspan(-.5, 4.5, alpha = 0.25, color = 'grey')
    tick_params(axis='x', labelsize=14)

    # PLOT MEANS
    subplot2grid((100, 3), (50, 1), colspan = 2, rowspan = 50)
    x = range(len(dot_counts))
    images = [str(image) for image in dot_counts]
    for i, ratio in enumerate(mean_mse_ratios):
        plot(i, ratio, 'o', color = 'black', ms = 8)

    axhline(y = 1, color = 'gray')
    xlim(-1, len(dot_counts))
    xticks(x, images)
    yscale('log')
    xlabel('Dot Count')
    ylabel('Mean: 2 guess MSE / 1 Guess MSE')
    ylim(ymin =10 ** (-2.5), ymax = 10 ** 2.5)
    axvspan(-.5, 4.5, alpha = 0.25, color = 'grey')
    tick_params(axis='x', labelsize=14)

    # PLOT SUBFIGURE
    mean_hist_and_median_lolipop(means_subfigure, medians_subfigure, dot_count)


'''
Figure 3 Center
'''
def mean_hist_and_median_lolipop(means, medians, dot_count):
    subplot2grid((100, 3), (0, 0), rowspan = 50)
    xlabel("Median estimate for " + str(dot_count) + " dots")
    ylabel("Frequency")

    def count(data):
        result = {}
        for entry in data:
            result[entry] = result.get(entry, 0) + 1.0 / NUM_BOOTSTRAP_SAMPLES

        return result

    # PLOTTING STUFF
    guess_1_median_counts = count(medians['1'])
    for i, x in enumerate(guess_1_median_counts):
        plot([x - 0.065] * 2, [0, guess_1_median_counts[x]], '-', color = '#FF8D70', lw = 2.5)

    guess_2_median_counts = count(medians['2'])
    for i, x in enumerate(guess_2_median_counts):
        plot([x + 0.065] * 2, [0, guess_2_median_counts[x]], '-', color = '#99C2FF', lw = 2.5)

    axvline(dot_count, linestyle='-', color='k', alpha = 0.25, lw = 15.0)

    for i, x in enumerate(guess_1_median_counts):
        plot(x - 0.065, guess_1_median_counts[x], 'o', color = '#FF8D70', ms = 9.0)

    for i, x in enumerate(guess_2_median_counts):
        plot(x + 0.065, guess_2_median_counts[x], 'o', color = '#99C2FF', ms = 9.0)

    xlim(dot_count - 8, dot_count + 8)
    ylim(0, 0.5)

    subplot2grid((100, 3), (50, 0), rowspan = 50)
    xlabel("Mean estimate for " + str(dot_count) + " dots")
    ylabel("Frequency")
    bins = [x + 0.5 for x in xrange(71, 86)]
    hist(means['1'], alpha = 0.5, normed = 1.0, color = '#FF3300', bins = bins)
    hist(means['2'], alpha = 0.5, normed = 1.0, color = '#2E78E6', bins = bins)
    axvline(dot_count, linestyle='-', color='k', alpha = 0.25, lw = 15.0)

    xlim(dot_count - 8, dot_count + 8)
    ylim(0, 0.5)


def figure_4(mg_game):
    figure(figsize = PLOT_SIZE)
    guess_symmetry(mg_game, '3')
    estimators_2_3_guess(mg_game, 50)
    tight_layout()


'''
Figure 4 Left
'''
def guess_symmetry(game, section):
    subplot2grid((100, 3), (0, 0), rowspan = 70)
    ylabel("Frequency")
    xlabel(r'$|(g_{max}\ -\ g_{mid})\ -\ (g_{mid}\ -\ g_{min})|$')

    x = []
    y = []
    for entry in game[section]:
        if entry[1] in TRAINING_INDICES:
            continue

        guesses = sorted(entry[3:-1])
        x.append(guesses[2] - guesses[1])
        y.append(guesses[1] - guesses[0])

    counts = {}
    for i in xrange(len(x)):
        difference = abs(x[i] - y[i])
        if difference > 20:
            difference = 23

        counts[difference] = counts.get(difference, 0) + 1.0 / len(x)

    for entry in counts:
        bar(entry, counts[entry], color = 'grey')

    xticks(
        [entry + 0.4 for entry in counts if entry % 5 == 0] + [23.4],
        [entry for entry in counts if entry % 5 == 0] + ['>20']
        )
    xlim(-0.25, 24)


'''
Figure 4 Right
'''
def estimators_2_3_guess(game, sample_size):
    data, dot_counts = index_by_dot_count(game)
    dot_counts = sorted(list(dot_counts))
    median_mse_ratios = []

    for image in dot_counts:
        print 'Bootstrapping Image (' + str(image) + ') for Figure 4'

        guesses_2 = data['2'][image]
        guesses_3 = data['3'][image]
        medians = {'2' : [], '3_mean_guess' : [], '3_median_guess' : []}

        for _ in xrange(NUM_BOOTSTRAP_SAMPLES):
            gusses_2_sample = sample_with_replacement(guesses_2, sample_size)
            average_guesses = [np.mean(guess) for guess in gusses_2_sample]
            weights = [1.0 / (np.max(guess) - np.min(guess)) for guess in gusses_2_sample]
            weighted_median = get_weighted_median(average_guesses, weights)

            medians['2'].append(weighted_median)

            gusses_3_sample = sample_with_replacement(guesses_3, sample_size)
            average_guesses = [np.mean(guess) for guess in gusses_3_sample]
            middle_guesses = [np.median(guess) for guess in gusses_3_sample]
            weights = [1.0 / (np.max(guess) - np.min(guess)) for guess in gusses_3_sample]

            weighted_median_mean_guess = get_weighted_median(average_guesses, weights)
            weighted_median_middle_guess = get_weighted_median(middle_guesses, weights)

            medians['3_mean_guess'].append(weighted_median_mean_guess)
            medians['3_median_guess'].append(weighted_median_middle_guess)

        mse_2_guess = get_mse(medians['2'], image)
        mse_3_mean_guess = get_mse(medians['3_mean_guess'], image)
        mse_3_median_guess = get_mse(medians['3_median_guess'], image)
        median_mse_ratios.append((mse_3_mean_guess / mse_2_guess, mse_3_median_guess / mse_2_guess))

    subplot2grid((100, 3), (7, 1), colspan = 2, rowspan = 63)
    images = [str(image) for image in dot_counts]
    count = 0
    x = []
    for mean_ratio, median_ratio in median_mse_ratios:
        plot(count,
            mean_ratio,
            'o',
            color = 'black',
            ms = 8,
            label = '3 Guess: ' + r'$\bar g$, $g_{max}\ -\ g_{min}$' if count == 0 else '')

        plot(count,
            median_ratio,
            'o',
            color = 'red',
            ms = 8,
            label = '3 Guess: ' + r'$g_{mid}$, $g_{max}\ -\ g_{min}$' if count == 0 else '')

        x.append(count)
        count += 2

    axvspan(-1, 9, alpha = 0.25, color = 'grey')
    axhline(y = 1, color = 'gray')
    xlim(-1.75, 2 * len(x))
    xticks(x, images)

    tick_params(axis='x', labelsize=14)
    yscale('log')
    xlabel('Dot Count')
    ylabel('3 Guess MSE / 2 Guess MSE')
    ylim(ymin =0.01, ymax = 100)
    legend(loc = 'upper center', numpoints = 1, ncol = 2, bbox_to_anchor=(0, 0, 1, 1.15))


def figure_5(interval_game):
    figure(figsize = PLOT_SIZE)
    guesses_percent_between_histogram(interval_game)
    time_per_image(interval_game)
    estimators_2_interval_guess(interval_game, 50)
    tight_layout()


'''
Figure 5 Top Left
'''
def guesses_percent_between_histogram(interval_game):
    subplot2grid((100, 100), (0, 0), colspan = 50, rowspan = 50)
    ylabel("Freuqency")
    xlabel("Number of Images between responses/inside interval")

    users = index_by_user_id(interval_game)

    def count_in_between(users, section):
        user_percentages = {}
        for user in users:
            in_range = 0.0
            total = 0.0
            for guess_1, guess_2, image in users[user][section]:
                total += 1.0

                low = min(guess_1, guess_2)
                high = max(guess_1, guess_2)

                if image >= low and image <= high:
                    in_range += 1.0

            user_percentages[in_range] = user_percentages.get(in_range, 0) + 1.0

        total = float(np.sum(user_percentages.values()))
        keys = sorted([key for key in user_percentages])

        for key in keys:
            user_percentages[key] /= total

        return keys, user_percentages

    # 2 Guess
    keys, two_guess_percentages = count_in_between(users, '2')
    bar(
        [in_range for in_range in keys],
        [two_guess_percentages[in_range] for in_range in keys],
        alpha = 0.50,
        color = '#2E78E6',
        label = '2 Guess')

    axvline(11.5, linestyle='--', color='k', alpha = 0.5)

    # Interval Guess
    keys, interval_guess_percentages = count_in_between(users, 'interval')
    bar(
        [in_range + 12 for in_range in keys],
        [interval_guess_percentages[in_range] for in_range in keys],
        alpha = 0.50,
        color = '#ffa500',
        label = 'Interval')

    # Binomial
    x = xrange(0, 11)
    pmf = scipy.stats.binom.pmf(x, 10, 0.5)
    bar(xrange(12, 23), pmf, color = 'grey', label = 'Binomial', alpha = 0.5)
    bar(xrange(0, 11), pmf, color = 'grey', alpha = 0.5)

    ticks = [str(int(key)) for key in x] * 2
    xticks([key + 0.4 for key in x] + [key + 12.4 for key in x], ticks)

    xlim(-1, 24)
    ylim(0, 0.3)
    legend(loc = 'upper right', prop={'size': 18})


'''
Figure 5 Bottom Left
'''
def time_per_image(game):
    subplot2grid((100, 100), (50, 0), colspan = 50, rowspan = 50)
    ylabel("Time spent (seconds)")
    xlabel("Image index")

    indices = {}
    for section in game:
        for entry in game[section]:
            time = entry[2] / 1000.0
            index = entry[1]
            if index not in indices:
                indices[index] = []

            indices[index].append(time)

        xs_2 = []
        ys_2 = []
        lower_2 = []
        upper_2 = []

        xs_interval = []
        ys_interval = []
        lower_interval = []
        upper_interval = []
        for index in indices:
            y = np.median(indices[index])
            if section == '2':
                x = index - 0.13
                xs_2.append(x)
                ys_2.append(y)
                plot(x, y, 'ro', color = '#2E78E6', ms = 8.0, label = '2 Guess' if index == 1 else '')
                lower_2.append(abs(y - np.percentile(indices[index], 5)))
                upper_2.append(abs(y - np.percentile(indices[index], 95)))
            else:
                x = index + 0.13
                xs_interval.append(x)
                ys_interval.append(y)
                plot(x, y, 'ro', color = '#ffa500', ms = 8.0, label = 'Interval' if index == 1 else '')
                lower_interval.append(abs(y - np.percentile(indices[index], 5)))
                upper_interval.append(abs(y - np.percentile(indices[index], 95)))

        errorbar(xs_2, ys_2, yerr = [lower_2, upper_2], fmt = None, ecolor = '#2E78E6')
        errorbar(xs_interval, ys_interval, yerr = [lower_interval, upper_interval], fmt = None, ecolor = '#ffa500')

    x = range(3, 13) + range(15, 25)
    labels = [str(i) for i in xrange(1, 21)]
    xticks(x, labels)

    axvspan(0.5, 2.5, alpha = 0.25, color = 'grey')
    axvspan(12.5, 14.5, alpha = 0.25, color = 'grey')

    annotate('Tutorial', xy=(2, 1), xytext=(1.8, 125), color = 'grey', rotation = 90)
    annotate('Tutorial', xy=(2, 1), xytext=(13.8, 125), color = 'grey', rotation = 90)
    legend(loc = 'upper right', prop={'size': 18}, numpoints = 1)
    xlim(0, 25)


'''
Figure 5 Top and Bottom Right
'''
def estimators_2_interval_guess(game, sample_size):
    data, dot_counts = index_by_dot_count(game)
    dot_counts = sorted(list(dot_counts))
    mean_mse_ratios = []
    median_mse_ratios = []

    for image in dot_counts:
        print 'Bootstrapping Image (' + str(image) + ') for Figure 5'
        guesses_2 = data['2'][image]
        guesses_interval = data['interval'][image]
        means = {'2' : [], 'interval' : []}
        medians = {'2' : [], 'interval' : []}

        for _ in xrange(NUM_BOOTSTRAP_SAMPLES):
            gusses_2_sample = sample_with_replacement(guesses_2, sample_size)
            average_guesses = [np.mean(guess) for guess in gusses_2_sample]
            weights = [1.0 / (np.max(guess) - np.min(guess)) for guess in gusses_2_sample]
            squared_weights = [weight ** 2 for weight in weights]

            weighted_mean = np.sum([squared_weights[i] * average_guesses[i]
                for i in xrange(sample_size)]) / np.sum(squared_weights)
            weighted_median = get_weighted_median(average_guesses, weights)

            means['2'].append(weighted_mean)
            medians['2'].append(weighted_median)

            guesses_interval_sample = sample_with_replacement(guesses_interval, sample_size)
            average_guesses = [np.mean(guess) for guess in guesses_interval_sample]
            weights = [1.0 / (np.max(guess) - np.min(guess)) for guess in guesses_interval_sample]
            squared_weights = [weight ** 2 for weight in weights]

            weighted_mean = np.sum([squared_weights[i] * average_guesses[i]
                for i in xrange(sample_size)]) / np.sum(squared_weights)
            weighted_median = get_weighted_median(average_guesses, weights)

            means['interval'].append(weighted_mean)
            medians['interval'].append(weighted_median)

        mean_mse = {'2' : get_mse(means['2'], image), 'interval' : get_mse(means['interval'], image)}
        median_mse = {'2' : get_mse(medians['2'], image), 'interval' : get_mse(medians['interval'], image)}

        mean_mse_ratios.append(mean_mse['2'] / mean_mse['interval'])
        median_mse_ratios.append(median_mse['2'] / median_mse['interval'])

    # PLOT MEDIANS
    subplot2grid((100, 100), (0, 50), colspan = 50, rowspan = 50)
    x = range(len(dot_counts))
    images = [str(image) for image in dot_counts]
    for i, ratio in enumerate(median_mse_ratios):
        plot(i, ratio, 'o', color = 'black', ms = 8)

    axhline(y = 1, color = 'gray')
    xlim(-1, len(dot_counts))
    xticks(x, images)
    yscale('log')
    xlabel('Dot Count')
    ylabel('Median: 2 guess MSE / interval guess MSE')
    ylim(ymin =10 ** (-2.5), ymax = 10 ** 2.5)
    axvspan(-.5, 4.5, alpha = 0.25, color = 'grey')
    tick_params(axis='x', labelsize=14)

    # PLOT MEANS
    subplot2grid((100, 100), (50, 50), colspan = 50, rowspan = 50)
    x = range(len(dot_counts))
    images = [str(image) for image in dot_counts]
    for i, ratio in enumerate(mean_mse_ratios):
        plot(i, ratio, 'o', color = 'black', ms = 8)

    axhline(y = 1, color = 'gray')
    xlim(-1, len(dot_counts))
    xticks(x, images)
    yscale('log')
    xlabel('Dot Count')
    ylabel('Mean: 2 guess MSE / interval Guess MSE')
    ylim(ymin =10 ** (-2.5), ymax = 10 ** 2.5)
    axvspan(-.5, 4.5, alpha = 0.25, color = 'grey')
    tick_params(axis='x', labelsize=14)


def main():
    font = {
        'family' : 'Bitstream Vera Sans',
        'weight' : 'normal',
        'size'   : 20
        }
    rc('font', **font)

    mg_scores = format_scores(readCSV('mge_data/mge_users.csv'))
    mg_game = getGame({
        '1' : 'mge_data/mge_1guess.csv',
        '2' : 'mge_data/mge_2guess.csv',
        '3' : 'mge_data/mge_3guess.csv'
        })
    mg_game = clean_game(mg_game, mg_scores)

    figure_3(mg_game)
    figure_4(mg_game)

    interval_scores = format_scores(readCSV('ice_data/ice_users.csv'))
    interval_game = getGame({
        '2' : 'ice_data/ice_2guess.csv',
        'interval' : 'ice_data/ice_interval.csv',
        })
    interval_game = clean_game(interval_game, interval_scores)

    figure_5(interval_game)
    show()


if __name__ == "__main__":
    main()
