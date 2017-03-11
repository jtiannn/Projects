"""The Game of Hog."""

from dice import four_sided, six_sided, make_test_dice
from ucb import main, trace, log_current_line, interact

GOAL_SCORE = 100  # The goal of Hog is to score 100 points.


######################
# Phase 1: Simulator #
######################


def roll_dice(num_rolls, dice=six_sided):
    """Simulate rolling the DICE exactly NUM_ROLLS times. Return the sum of
    the outcomes unless any of the outcomes is 1. In that case, return 0.
    """
    # These assert statements ensure that num_rolls is a positive integer.
    assert type(num_rolls) == int, 'num_rolls must be an integer.'
    assert num_rolls > 0, 'Must roll at least once.'
    # BEGIN Question 1
    total, pigout = 0, False
    while num_rolls > 0:
        x = dice()
        if x == 1:
            pigout = True
        total += x
        num_rolls -= 1
    if pigout:
        return 0
    else:
        return total

    # END Question 1

def is_prime(x):
    """
    >>>is_prime(2)
    True
    >>>is_prime(0)
    False
    >>>is_prime(5)
    True
    """
    if x == 0 or x == 1:
        return False
    n = 2
    while n < x:
        if x % n == 0:
            return False
        n += 1
    return True

def next_prime(x):
    y = x + 1
    while is_prime(y) == False:
        y = y + 1
    return y


def take_turn(num_rolls, opponent_score, dice=six_sided):
    """Simulate a turn rolling NUM_ROLLS dice, which may be 0 (Free Bacon).

    num_rolls:       The number of dice rolls that will be made.
    opponent_score:  The total score of the opponent.
    dice:            A function of no args that returns an integer outcome.
    """
    assert type(num_rolls) == int, 'num_rolls must be an integer.'
    assert num_rolls >= 0, 'Cannot roll a negative number of dice.'
    assert num_rolls <= 10, 'Cannot roll more than 10 dice.'
    assert opponent_score < 100, 'The game should be over.'
    # BEGIN Question 2
    digit = opponent_score % 10
    if num_rolls == 0:
        final = max(digit, (opponent_score - digit) // 10) + 1
    else:
        final = roll_dice(num_rolls, dice)
    if is_prime(final):
        final = next_prime(final)
    return final
    # END Question 2


def select_dice(score, opponent_score):
    """Select six-sided dice unless the sum of SCORE and OPPONENT_SCORE is a
    multiple of 7, in which case select four-sided dice (Hog wild).
    """
    # BEGIN Question 3
    if (score + opponent_score) % 7 == 0:
        return four_sided
    else:
        return six_sided
    # END Question 3


def is_swap(score0, score1):
    """Returns whether the last two digits of SCORE0 and SCORE1 are reversed
    versions of each other, such as 19 and 91.
    """
    # BEGIN Question 4
    x1 = score0 % 100 % 10
    x2 = (score0 % 100 - x1) // 10
    y1 = score1 % 100 % 10
    y2 = (score1 % 100 - y1) // 10
    if x1 == y2 and y1 == x2:
        return True
    return False

    # END Question 4


def other(player):
    """Return the other player, for a player PLAYER numbered 0 or 1.

    >>> other(0)
    1
    >>> other(1)
    0
    """
    return 1 - player


def play(strategy0, strategy1, score0=0, score1=0, goal=GOAL_SCORE):
    """Simulate a game and return the final scores of both players, with
    Player 0's score first, and Player 1's score second.

    A strategy is a function that takes two total scores as arguments
    (the current player's score, and the opponent's score), and returns a
    number of dice that the current player will roll this turn.

    strategy0:  The strategy function for Player 0, who plays first
    strategy1:  The strategy function for Player 1, who plays second
    score0   :  The starting score for Player 0
    score1   :  The starting score for Player 1
    """
    player = 0  # Which player is about to take a turn, 0 (first) or 1 (second)
    # BEGIN Question 5
    while score0 < goal and score1 < goal:
        if player == 0:
            rolls = strategy0(score0, score1)
            turn_score = take_turn(rolls, score1, select_dice(score0, score1))
            if turn_score == 0:
                score1 += rolls
            else:
                score0 += turn_score
        elif player == 1:
            rolls = strategy1(score1, score0)
            turn_score = take_turn(rolls, score0, select_dice(score0, score1))
            if turn_score == 0:
                score0 += rolls
            else:
                score1 += turn_score
        if is_swap(score0, score1):
            temp = score0
            score0 = score1
            score1 = temp
        player = other(player)
    # END Question 5
    return score0, score1


#######################
# Phase 2: Strategies #
#######################


def always_roll(n):
    """Return a strategy that always rolls N dice.

    A strategy is a function that takes two total scores as arguments
    (the current player's score, and the opponent's score), and returns a
    number of dice that the current player will roll this turn.

    >>> strategy = always_roll(5)
    >>> strategy(0, 0)
    5
    >>> strategy(99, 99)
    5
    """
    def strategy(score, opponent_score):
        return n

    return strategy


# Experiments

def make_averaged(fn, num_samples=1000):
    """Return a function that returns the average_value of FN when called.

    To implement this function, you will have to use *args syntax, a new Python
    feature introduced in this project.  See the project description.

    >>> dice = make_test_dice(3, 1, 5, 6)
    >>> averaged_dice = make_averaged(dice, 1000)
    >>> averaged_dice()
    3.75
    >>> make_averaged(roll_dice, 1000)(2, dice)
    5.5

    In this last example, two different turn scenarios are averaged.
    - In the first, the player rolls a 3 then a 1, receiving a score of 0.
    - In the other, the player rolls a 5 and 6, scoring 11.
    Thus, the average value is 5.5.
    Note that the last example uses roll_dice so the hogtimus prime rule does
    not apply.
    """
    # BEGIN Question 6
    def average(*args):
        result = 0
        trials = num_samples
        while trials > 0:
            result += fn(*args)
            trials -= 1
        return result / num_samples
    return average

    # END Question 6


def max_scoring_num_rolls(dice=six_sided, num_samples=1000):
    """Return the number of dice (1 to 10) that gives the highest average turn
    score by calling roll_dice with the provided DICE over NUM_SAMPLES times.
    Assume that the dice always return positive outcomes.

    >>> dice = make_test_dice(3)
    >>> max_scoring_num_rolls(dice)
    10
    """
    # BEGIN Question 7
    num_dice = 1
    best_dice = 1
    best_score = 0
    while num_dice <= 10:
        avg_score = make_averaged(roll_dice, num_samples)
        current_score = avg_score(num_dice, dice)
        if current_score > best_score:
            best_score = current_score
            best_dice = num_dice
        num_dice += 1
    return best_dice
    # END Question 7


def winner(strategy0, strategy1):
    """Return 0 if strategy0 wins against strategy1, and 1 otherwise."""
    score0, score1 = play(strategy0, strategy1)
    if score0 > score1:
        return 0
    else:
        return 1


def average_win_rate(strategy, baseline=always_roll(5)):
    """Return the average win rate of STRATEGY against BASELINE. Averages the
    winrate when starting the game as player 0 and as player 1.
    """
    win_rate_as_player_0 = 1 - make_averaged(winner)(strategy, baseline)
    win_rate_as_player_1 = make_averaged(winner)(baseline, strategy)

    return (win_rate_as_player_0 + win_rate_as_player_1) / 2


def run_experiments():
    """Run a series of strategy experiments and report results."""
    if False:  # Change to False when done finding max_scoring_num_rolls
        six_sided_max = max_scoring_num_rolls(six_sided)
        print('Max scoring num rolls for six-sided dice:', six_sided_max)
        four_sided_max = max_scoring_num_rolls(four_sided)
        print('Max scoring num rolls for four-sided dice:', four_sided_max)

    if False:  # Change to True to test always_roll(8)
        print('always_roll(8) win rate:', average_win_rate(always_roll(8)))

    if False:  # Change to True to test bacon_strategy
        print('bacon_strategy win rate:', average_win_rate(bacon_strategy))

    if False:  # Change to True to test swap_strategy
        print('swap_strategy win rate:', average_win_rate(swap_strategy))

    if True: 
        print('swap_strategy win rate:', average_win_rate(final_strategy))


# Strategies

def bacon_strategy(score, opponent_score, margin=8, num_rolls=5):
    """This strategy rolls 0 dice if that gives at least MARGIN points,
    and rolls NUM_ROLLS otherwise.
    """
    # BEGIN Question 8
    roll_zero = max(opponent_score // 10, opponent_score % 10) + 1
    if is_prime(roll_zero):
        roll_zero = next_prime(roll_zero)
    if roll_zero >= margin:
        return 0
    else:
        return num_rolls
    # END Question 8


def swap_strategy(score, opponent_score, num_rolls=5):
    """This strategy rolls 0 dice when it results in a beneficial swap and
    rolls NUM_ROLLS otherwise.
    """
    # BEGIN Question 9
    roll_zero = max(opponent_score // 10, opponent_score % 10) + 1
    if is_prime(roll_zero):
        roll_zero = next_prime(roll_zero)
    new_score = score + roll_zero
    if is_swap(new_score, opponent_score) and new_score < opponent_score:
        return 0
    else:
        return num_rolls  
    # END Question 9

def dont_swap_strategy(score, opponent_score):
    """This strategy trys its best to not return rolls that may result 
    in an unbeneficial swap
    """
    x1 = opponent_score % 10
    x2 = opponent_score // 10
    roll_zero = max(x1, x2) + 1
    if is_prime(roll_zero):
        roll_zero = next_prime(roll_zero)
    new_score = score + roll_zero
    if is_swap(new_score, opponent_score) and new_score > opponent_score:
        return 5
    else:
        return 0  

def final_strategy(score, opponent_score):
    """
    This function combines previous functions to produce the most optimal strategy.
    See comments after each if-statement.
    """
    # BEGIN Question 10
    difference = opponent_score - score
    if bacon_strategy(score, opponent_score, GOAL_SCORE - score) == 0: 
        return 0 # when bacon rule is beneficial
    elif score >= 97: 
        return 1 # when risk of pig-out rule is manageable 
    elif swap_strategy(score, opponent_score) == 0:
        return 0 # when bacon rule helps achieve beneficial swap
    elif dont_swap_strategy(score, opponent_score) == 5:
        return 5 # when swap is not beneficial
    elif bacon_strategy(score, opponent_score, 5) == 0:
        return 0 # when margin of rolling 0 dice is 5 or greater
    elif select_dice(score, opponent_score) == six_sided:
        return 3 # best num_roll for six_sided dice
    elif select_dice(score, opponent_score) == four_sided:
        return 2 # best num_roll for four_sided dice
    # END Question 10


##########################
# Command Line Interface #
##########################


# Note: Functions in this section do not need to be changed. They use features
#       of Python not yet covered in the course.


@main
def run(*args):
    """Read in the command-line argument and calls corresponding functions.

    This function uses Python syntax/techniques not yet covered in this course.
    """
    import argparse
    parser = argparse.ArgumentParser(description="Play Hog")
    parser.add_argument('--run_experiments', '-r', action='store_true',
                        help='Runs strategy experiments')

    args = parser.parse_args()

    if args.run_experiments:
        run_experiments()
