# This is a very simple implementation of the UCT Monte Carlo Tree Search algorithm in Python 2.7.
# The function UCT(rootstate, itermax, verbose = False) is towards the bottom of the code.
# It aims to have the clearest and simplest possible code, and for the sake of clarity, the code
# is orders of magnitude less efficient than it could be made, particularly by using a
# state.GetRandomMove() or state.DoRandomRollout() function.
#
# Example GameState classes for Nim, OXO and Othello are included to give some idea of how you
# can write your own GameState use UCT in your 2-player game. Change the game to be played in
# the UCTPlayGame() function at the bottom of the code.
#
# Written by Peter Cowling, Ed Powley, Daniel Whitehouse (University of York, UK) September 2012.
#
# Licence is granted to freely use and distribute for any sensible/legal purpose so long as this comment
# remains in any distributed code.
#
# For more information about Monte Carlo Tree Search check out our web site at www.mcts.ai
#
# Modifications to the original code by Peter Cowlin, Ed Powley and Daniel Whitehouse have been made
# by Angel Seiji Morimoto Burgos.

import argparse
import itertools
import multiprocessing as mp
import os
import random
import sklearn
import time
from joblib import dump, load
from math import *
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier


class GameState:
    """ A state of the game, i.e. the game board. These are the only functions which are
        absolutely necessary to implement UCT in any 2-player complete information deterministic
        zero-sum game, although they can be enhanced and made quicker, for example by using a
        GetRandomMove() function to generate a random move during rollout.
        By convention the players are numbered 1 and 2.
    """

    def __init__(self):
        # At the root pretend the player just moved is player 2 - player 1 has the first move
        self.playerJustMoved = 2

    def Clone(self):
        """ Create a deep clone of this game state.
        """
        st = GameState()
        st.playerJustMoved = self.playerJustMoved
        return st

    def DoMove(self, move):
        """ Update a state by carrying out the given move.
            Must update playerJustMoved.
        """
        self.playerJustMoved = 3 - self.playerJustMoved

    def GetMoves(self):
        """ Get all possible moves from this state.
        """

    def GetResult(self, playerjm):
        """ Get the game result from the viewpoint of playerjm.
        """

    def __repr__(self):
        """ Don't need this - but good style.
        """
        pass


class OthelloState:
    """ A state of the game of Othello, i.e. the game board.
        The board is a 2D array where 0 = empty (.), 1 = player 1 (X), 2 = player 2 (O).
        In Othello players alternately place pieces on a square board - each piece played
        has to sandwich opponent pieces between the piece played and pieces already on the
        board. Sandwiched pieces are flipped.
        This implementation modifies the rules to allow variable sized square boards and
        terminates the game as soon as the player about to move cannot make a move (whereas
        the standard game allows for a pass move).
    """

    def __init__(self, sz=8):
        # At the root pretend the player just moved is p2 - p1 has the first move
        self.playerJustMoved = 2
        self.board = []  # 0 = empty, 1 = player 1, 2 = player 2
        self.size = sz
        assert sz == int(sz) and sz % 2 == 0  # size must be integral and even
        for y in range(sz):
            self.board.append([0]*sz)
        self.board[int(sz/2)][int(sz/2)
                              ] = self.board[int(sz/2-1)][int(sz/2-1)] = 1
        self.board[int(sz/2)][int(sz/2-1)
                              ] = self.board[int(sz/2-1)][int(sz/2)] = 2

    def Clone(self):
        """ Create a deep clone of this game state.
        """
        st = OthelloState()
        st.playerJustMoved = self.playerJustMoved
        st.board = [self.board[i][:] for i in range(self.size)]
        st.size = self.size
        return st

    def DoMove(self, move):
        """ Update a state by carrying out the given move.
            Must update playerToMove.
        """
        (x, y) = (move[0], move[1])
        assert x == int(x) and y == int(y) and self.IsOnBoard(
            x, y) and self.board[x][y] == 0
        m = self.GetAllSandwichedCounters(x, y)
        self.playerJustMoved = 3 - self.playerJustMoved
        self.board[x][y] = self.playerJustMoved
        for (a, b) in m:
            self.board[a][b] = self.playerJustMoved

    def GetMoves(self):
        """ Get all possible moves from this state.
        """
        return [(x, y) for x in range(self.size) for y in range(self.size) if self.board[x][y] == 0 and self.ExistsSandwichedCounter(x, y)]

    def AdjacentToEnemy(self, x, y):
        """ Speeds up GetMoves by only considering squares which are adjacent to an enemy-occupied square.
        """
        for (dx, dy) in [(0, +1), (+1, +1), (+1, 0), (+1, -1), (0, -1), (-1, -1), (-1, 0), (-1, +1)]:
            if self.IsOnBoard(x+dx, y+dy) and self.board[x+dx][y+dy] == self.playerJustMoved:
                return True
        return False

    def AdjacentEnemyDirections(self, x, y):
        """ Speeds up GetMoves by only considering squares which are adjacent to an enemy-occupied square.
        """
        es = []
        for (dx, dy) in [(0, +1), (+1, +1), (+1, 0), (+1, -1), (0, -1), (-1, -1), (-1, 0), (-1, +1)]:
            if self.IsOnBoard(x+dx, y+dy) and self.board[x+dx][y+dy] == self.playerJustMoved:
                es.append((dx, dy))
        return es

    def ExistsSandwichedCounter(self, x, y):
        """ Does there exist at least one counter which would be flipped if my counter was placed at (x,y)?
        """
        for (dx, dy) in self.AdjacentEnemyDirections(x, y):
            if len(self.SandwichedCounters(x, y, dx, dy)) > 0:
                return True
        return False

    def GetAllSandwichedCounters(self, x, y):
        """ Is (x,y) a possible move (i.e. opponent counters are sandwiched between (x,y) and my counter in some direction)?
        """
        sandwiched = []
        for (dx, dy) in self.AdjacentEnemyDirections(x, y):
            sandwiched.extend(self.SandwichedCounters(x, y, dx, dy))
        return sandwiched

    def SandwichedCounters(self, x, y, dx, dy):
        """ Return the coordinates of all opponent counters sandwiched between (x,y) and my counter.
        """
        x += dx
        y += dy
        sandwiched = []
        while self.IsOnBoard(x, y) and self.board[x][y] == self.playerJustMoved:
            sandwiched.append((x, y))
            x += dx
            y += dy
        if self.IsOnBoard(x, y) and self.board[x][y] == 3 - self.playerJustMoved:
            return sandwiched
        else:
            return []  # nothing sandwiched

    def IsOnBoard(self, x, y):
        return x >= 0 and x < self.size and y >= 0 and y < self.size

    def GetResult(self, playerjm):
        """ Get the game result from the viewpoint of playerjm.
        """
        jmcount = len([(x, y) for x in range(self.size)
                       for y in range(self.size) if self.board[x][y] == playerjm])
        notjmcount = len([(x, y) for x in range(self.size)
                          for y in range(4) if self.board[x][y] == 3 - playerjm])
        if jmcount > notjmcount:
            return 1.0
        elif notjmcount > jmcount:
            return 0.0
        else:
            return 0.5  # draw

    def __repr__(self):
        s = ""
        for y in range(self.size-1, -1, -1):
            for x in range(self.size):
                s += ".XO"[self.board[x][y]]
            s += "\n"
        return s


class Node:
    """ A node in the game tree. Note wins is always from the viewpoint of playerJustMoved.
        Crashes if state not specified.
    """

    def __init__(self, move=None, parent=None, state=None):
        self.move = move  # the move that got us to this node - "None" for the root node
        self.parentNode = parent  # "None" for the root node
        self.childNodes = []
        self.wins = 0
        self.visits = 0
        self.untriedMoves = state.GetMoves()  # future child nodes
        # the only part of the state that the Node needs later
        self.playerJustMoved = state.playerJustMoved

    def UCTSelectChild(self):
        """ Use the UCB1 formula to select a child node. Often a constant UCTK is applied so we have
            lambda c: c.wins/c.visits + UCTK * sqrt(2*log(self.visits)/c.visits to vary the amount of
            exploration versus exploitation.
        """
        s = sorted(self.childNodes, key=lambda c: c.wins /
                   c.visits + sqrt(2*log(self.visits)/c.visits))[-1]
        return s

    def AddChild(self, m, s):
        """ Remove m from untriedMoves and add a new child node for this move.
            Return the added child node
        """
        n = Node(move=m, parent=self, state=s)
        self.untriedMoves.remove(m)
        self.childNodes.append(n)
        return n

    def Update(self, result):
        """ Update this node - one additional visit and result additional wins. result must be from the viewpoint of playerJustmoved.
        """
        self.visits += 1
        self.wins += result

    def __repr__(self):
        return "[M:" + str(self.move) + " W/V:" + str(self.wins) + "/" + str(self.visits) + " U:" + str(self.untriedMoves) + "]"

    def TreeToString(self, indent):
        s = self.IndentString(indent) + str(self)
        for c in self.childNodes:
            s += c.TreeToString(indent+1)
        return s

    def IndentString(self, indent):
        s = "\n"
        for i in range(1, indent+1):
            s += "| "
        return s

    def ChildrenToString(self):
        s = ""
        for c in self.childNodes:
            s += str(c) + "\n"
        return s


def UCT(rootstate, itermax, model=None, verbose=0):
    """ Conduct a UCT search for itermax iterations starting from rootstate.
        Return the best move from the rootstate.
        Assumes 2 alternating players (player 1 starts), with game results in the range [0.0, 1.0].
    """

    rootnode = Node(state=rootstate)

    for i in range(itermax):
        node = rootnode
        state = rootstate.Clone()

        # Select
        while node.untriedMoves == [] and node.childNodes != []:  # node is fully expanded and non-terminal
            node = node.UCTSelectChild()
            state.DoMove(node.move)

        # Expand
        # if we can expand (i.e. state/node is non-terminal)
        if node.untriedMoves != []:
            m = random.choice(node.untriedMoves)
            state.DoMove(m)
            node = node.AddChild(m, state)  # add child and descend tree

        # Rollout - this can often be made orders of magnitude quicker using a state.GetRandomMove() function
        if model is None:
            while state.GetMoves() != []:
                state.DoMove(random.choice(state.GetMoves()))
        else:
            while state.GetMoves() != []:  # While state is non-terminal.
                if random.random() <= 0.10:
                    state.DoMove(random.choice(state.GetMoves()))
                else:
                    state.DoMove(PredictActionWithModel(state, model))

        # Backpropagate
        while node != None:  # backpropagate from the expanded node and work back to the root node
            # state is terminal. Update node with result from POV of node.playerJustMoved
            node.Update(state.GetResult(node.playerJustMoved))
            node = node.parentNode

    # Output some information about the tree - can be omitted
    if (verbose == 2):
        print(rootnode.TreeToString(0))
    elif (verbose == 1):
        print(rootnode.ChildrenToString())

    # return the move that was most visited
    return sorted(rootnode.childNodes, key=lambda c: c.visits)[-1].move


def PredictActionWithModel(state, model):
    '''
    Predicts an action given an Othello game state using a machine learning model.

    :param state: Othello game state from which the action will be predicted.

    :param model: the learning model used for predicting the action.

    :returns: the action predicted in the form of a two-element tuple, where the first element
    specifies the row position of the board and the second the column position.
    '''
    # Format the state so that it is a valid input to the model.
    board_flat = []
    for row in state.board:
        board_flat.extend(row)
    current_player = 3 - state.playerJustMoved
    X = [board_flat + [current_player]]

    # Predict the action using the model.
    action = model.predict(X)[0]
    row_pos = int(action[0])
    col_pos = int(action[1])

    # Return the action predicted by the model if it is a legal action. Otherwise, pick a random
    # legal action.
    if state.IsOnBoard(row_pos, col_pos) and state.board[row_pos][col_pos] == 0:
        return (row_pos, col_pos)
    else:
        return random.choice(state.GetMoves())


def UCTPlayGame(gameId, itermax_player1, itermax_player2, model_player1=None, model_player2=None, name_player1=None, name_player2=None):
    '''
    Plays an Othello game between two UCT players, where each player gets its own number of UCT
    iterations (= simulations = tree nodes), its own learning model to use as the playout/simulation
    policy and its own name.

    :param gameId: integer representing the number of game being played between player 1 and 2.

    :param itermax_player1: number of maximum iterations to run in UCT when using player 1.

    :param itermax_player2: number of maximum iterations to run in UCT when using player 2.

    :param model_player1: learning model to use as the playout policy of player 1. If None, the
    uniform random distribution will be used as the playout policy.

    :param model_player2: learning model to use as the playout policy of player 2. If None, the
    uniform random distribution will be used as the playout policy.

    :param name_player1: the name of the player 1. Used for displaying info to the console.
    Necessary if the winner should be returned from this function.

    :param name_player2: the name of the player 2. Used for displaying info to the console.
    Necessary if the winner should be returned from this function.

    :returns: a tuple of three elements. The first one is a list with two elements, each one being
    a list of the states and actions taken by each player. The second one is a list with two
    elements, each one being a list of the times required for each player to take an action. The
    third one is the name of the player who won or None (if draw or if name_player1 and
    name_player2 are not known).
    '''
    # Initialize variables.
    datasets = [[], []]
    times = [[], []]
    state = OthelloState()

    # Print information about the match if the name of the players is known.
    if name_player1 and name_player2:
        print(f'Starting game {gameId} between {name_player1} and {name_player2}')
    
    # Make the current player perform a move while the game is not over (there are still moves left)
    while (state.GetMoves() != []):
        # Use UCT with the correct parameters according to the current player. 
        current_player = 3 - state.playerJustMoved
        if current_player == 1:
            start_time = time.time()
            m = UCT(rootstate=state, itermax=itermax_player1,
                    model=model_player1)
            end_time = time.time()
        else:
            start_time = time.time()
            m = UCT(rootstate=state, itermax=itermax_player2,
                    model=model_player2)
            end_time = time.time()
        
        # Flatten the board, obtain the x and y coordinates of the action taken and put that info
        # (along with the current player) as part of the data collected. Also, register how much
        # time was needed to perform the UCT function.
        board_flat = []
        for row in state.board:
            board_flat.extend(row)
        row_pos, col_pos = m
        action_label = str(row_pos) + str(col_pos)
        datasets[current_player -
                 1].append((board_flat, current_player, action_label))
        times[current_player - 1].append(end_time - start_time)

        # Perform the move selected by the UCT function.
        state.DoMove(m)
    
    # If the names of the players are known, find who is the winner and print in the console the
    # result of the match (who won/lost or if it was a draw).
    if name_player1 and name_player2:
        if state.GetResult(state.playerJustMoved) == 1.0:
            if state.playerJustMoved == 1:
                winner = name_player1
                loser = name_player2
            else:
                winner = name_player2
                loser = name_player1
            print(f'Player {winner} wins game {gameId} against {loser}!')
        elif state.GetResult(state.playerJustMoved) == 0.0:
            if state.playerJustMoved == 1:
                winner = name_player2
                loser = name_player1
            else:
                winner = name_player1
                loser = name_player2
            print(f'Player {winner} wins game {gameId} against {loser}!')
        else:
            winner = None
            print(
                f'Draw in game {gameId} between {name_player1} and {name_player2}!')
        return (datasets, times, winner)
    return (datasets, times, None)


def UCTPlayGames(games=1, itermax_player1=100, itermax_player2=100, model_player1=None, model_player2=None, name_player1=None, name_player2=None):
    '''
    Plays a series of Othello games between two players where each player gets its own number of UCT
    iterations (= simulations = tree nodes), its own learning model to use as the playout/simulation
    policy and its own name.

    See UCTPlayGame() for a description of most of the parameters.

    :param games: number of games to play between the two players. If not a multiple of the number
    of a CPU cores in the system, it will be modified to equal the next multiple of CPU cores.

    :returns: a list with size equal to the number of games played. Each position within the list
    is the three-element tuple returned by a UCTPlayGame() call.
    '''
    # Initialize variables.
    results = []

    # Set the number of games to a multiple of the number of CPU cores in the system.
    if games % mp.cpu_count() != 0:
        games = games + (mp.cpu_count() - (games % mp.cpu_count()))

    # Iterate until all games have been played. In each iteration 'n' games will be played in
    # parallel, where 'n' is the number of CPU cores in the system.
    for i in range(int(games / mp.cpu_count())):
        t = time.time()
        with mp.Pool(processes=mp.cpu_count()) as pool:
            gamesId = [id for id in range(
                i * mp.cpu_count(), (i + 1) * mp.cpu_count())]
            args = [(id, itermax_player1, itermax_player2, model_player1, model_player2,
                     name_player1, name_player2) for id in gamesId]
            results += pool.starmap(UCTPlayGame, args)
    
    # Return the results given by each call to UCTPlayGame.
    return results


def UCTPlayGamesAndCollectData(games=1, itermax_player1=100, itermax_player2=100, model_player1=None, model_player2=None, name_player1=None, name_player2=None):
    '''
    Plays a series of Othello games between two players where each player gets its own number of UCT
    iterations (= simulations = tree nodes), its own learning model to use as the playout/simulation
    policy and its own name. It collects data from these games and returns it.

    See UCTPlayGames() for a description of the parameters, which are the same.

    :returns: a tuple of two elements. The first one is a list with two elements, each one being
    a list of the states and actions taken by each player during the games. The second one is a
    list with two elements, each one being a list of the times required for each player to take an
    action during the games.
    '''
    # Initialize variables.
    datasets = [[], []]
    times = [[], []]

    # Log data to the console if the players' names are known.
    if name_player1 and name_player2:
        print(f'Starting games between {name_player1} and {name_player2} for collecting data...')
        start_time = time.time()
    
    # Play the games.
    results = UCTPlayGames(games, itermax_player1, itermax_player2,
                           model_player1, model_player2, name_player1, name_player2)

    # Log data to the console if the players' names are known.
    if name_player1 and name_player2:
        duration = time.time() - start_time
        print(f'Finished games in {duration} seconds.')

    # Merge all the data and times collected of the games played into only two collections.
    for (dataset, timeset, _) in results:
        datasets[0].extend(dataset[0])
        datasets[1].extend(dataset[1])
        times[0].extend(timeset[0])
        times[1].extend(timeset[1])

    # Return the collections of data and times of the games played.
    return (datasets, times)


def UCTPlayTournament(games, model_name, competitors, models_dir, out_file):
    '''
    Makes an agent play a set of games against some competitors.

    :param games: number of games the agent will play against each competitor.

    :param model_name: name of the agent that will play against other.

    :param competitors: list of strings. Each element corresponds to the name of an agent that will
    compete against 'model_name'.

    :param models_dir: string representing the path to the directory containing all the decision
    tree models.

    :param out_file: string representing the path to the file where the results of the games will
    be printed to.
    '''
    # Breaks up the name of the model to find out the details of it. For a model with name DT_i_j,
    # 'i' specifies the number of iterations to use in MCTS and 'j' represents which decision tree
    # it represents (e.g. j = 2 represent that model is the second decision tree generated). If 'j'
    # is 0, then no decision model should be loaded.
    model_name_split = model_name.split('_')
    itermax_player1 = int(model_name_split[1])
    if int(model_name_split[2]) == 0:
        model = None
    else:
        # Loads the model from disk.
        model_type = '_'.join(model_name_split[:-1])
        model = load(f'{models_dir}{model_type}/{model_name}.joblib')

    # Iterate over the list of competitors.
    for competitor in competitors:
        # Breaks up the name of the competitor model to find out details of it. See above comment.
        competitor_name_split = competitor.split('_')
        itermax_player2 = int(competitor_name_split[1])
        if int(competitor_name_split[2]) == 0:
            model_competitor = None
        else:
            # Loads the model corresponding to the competitor from disk.
            model_competitor_type = '_'.join(competitor_name_split[:-1])
            model_competitor = load(f'{models_dir}{model_competitor_type}/{competitor}.joblib')
        
        # Play half of the games with 'model' as player 1 and the other half with it as player 2.
        print(f'Playing games between {model_name} and {competitor}.')
        results = UCTPlayGames(int(games / 2), itermax_player1, itermax_player2,
                               model, model_competitor, model_name, competitor)
        results += UCTPlayGames(int(games / 2), itermax_player2, itermax_player1,
                                model_competitor, model, competitor, model_name)

        # Count the number of times 'model' won/lost/draw against 'model_competitor'.
        game_count = len(results)
        win_count = 0
        draw_count = 0
        for (_, _, winner) in results:
            if winner == model_name:
                win_count = win_count + 1
            elif winner is None:
                draw_count = draw_count + 1
        loss_count = game_count - win_count - draw_count

        # Write the results to the output file.
        with open(out_file, 'a') as file:
            file.write(f'{model_name} vs {competitor},{game_count},{win_count},{draw_count},{loss_count}\n')


def UCTTrainModels(n, games_per_model, iterations):
    '''
    Generates decision tree models from data collected in UCT procedures using a random playout
    policy and previous decision tree models.

    :param n: number of decision tree models to generate.

    :param games_per_model: number of games from which to collect data before generating a decision
    tree model.

    :param iterations: number of iterations to run the MCTS procedure with.
    '''
    # Initialize variables.
    model = None
    min_samples_leaf = None
    min_samples_split = None
    X = []
    y = []

    # Constants representing the paths to the directories that will hold the trained models, the
    # datasets collected and the information gathered about execution time taken to make MCTS/UCT
    # decisions.
    models_root_path = "models/"
    datasets_root_path = "data/games_data/"
    times_root_path = "data/times_data/"

    # Paths to the specific directories for the type of models that will be created in this run of
    # the function call.
    paths = [f'{models_root_path}DT_{iterations}',
             f'{datasets_root_path}DT_{iterations}',
             f'{times_root_path}DT_{iterations}']

    # Create the directories if they don't exist yet.
    for path in paths:
        if not os.path.exists(path):
            os.mkdir(path)

    # Iteration for generating 'n' decision tree models.
    for i in range(n):
        # Collect data by running 'games_per_model' games employing the last created model as the
        # playout policy in MCTS/UCT. If no model has been created yet, use a random playout policy.
        print(f'Collecting data for model DT_{iterations}_{i + 1}...')
        model_name = f'DT_{iterations}_{i}'
        (datasets, times) = UCTPlayGamesAndCollectData(games_per_model, iterations, iterations,
            model, model, f'{model_name}-1', f'{model_name}-2')

        # Save the data collected for training a model (and metadata) to disk.
        print(f'Saving data collected for model DT_{iterations}_{i + 1}...')
        for dataset in datasets:
            WriteDatasetToCSVFile(dataset, f'{paths[1]}/{model_name}.csv')
        for timeset in times:
            WriteTimesToFile(timeset, f'{paths[2]}/{model_name}.csv')

        # Put the corresponding parts of the dataset collected in this iteration with the inputs and
        # outputs collected in previous iterations of this for-loop.
        data = datasets[0] + datasets[1]
        X.extend([record[0] + [record[1]] for record in data])
        y.extend([record[2] for record in data])

        # If the hyper-params of the DT models have not been set, do a grid search to find the ones
        # that have the best performance.
        if min_samples_leaf is None or min_samples_split is None:
            print(f'Selecting hyper-parameters for model...')
            params = SelectModelHyperParams(
                X, y, f'{paths[0]}/grid_search_results.txt')
            min_samples_leaf = params['min_samples_leaf']
            min_samples_split = params['min_samples_split']

        # Train a decision tree model by using all the data collected thus far (from all iterations
        # in this for-loop).
        print(f'Training model DT_{iterations}_{i + 1}...')
        model = DecisionTreeClassifier(
            min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split, random_state=0)
        model.fit(X, y)

        # Save the trained model to disk.
        print(f'Saving model DT_{iterations}_{i + 1} to disk...')
        dump(model, f'{paths[0]}/DT_{iterations}_{i + 1}.joblib')


def WriteDatasetToCSVFile(dataset, csv_file, output_mode='a'):
    '''
    Writes a dataset of states and actions to disk.

    :param dataset: the dataset to save to disk.

    :param csv_file: the path to the file where the dataset will be saved to.

    :param output_mode: mode in which the 'csv_file' will be opened.
    '''
    with open(csv_file, output_mode) as file:
        for (board_flat, player, action_label) in dataset:
            line = ','.join(str(val) for val in board_flat) + ',' + str(player) + \
                ',' + f'\"{action_label}\"' + '\n'
            file.write(line)


def WriteTimesToFile(times, file_name, output_mode='a'):
    '''
    Writes a dataset of times to disk.

    :param times: the dataset to save to disk.

    :param file_name: the path to the file where the dataset will be saved to.

    :param output_mode: mode in which the 'file_name' will be opened.
    '''
    with open(file_name, output_mode) as file:
        for time in times:
            file.write(str(time) + '\n')


def SelectModelHyperParams(X, y, file_name):
    '''
    Performs a grid search to select the best hyper-parameters for a decision tree model.

    :param X: the input data that will be used for the grid search.

    :param y: the output data that will be used for the grid search.

    :param file_name: the path to the file where the best params and scores will be saved to.

    :returns: a dictionary where the keys are the names of the hyper-parameters optimized and the
    values are the best values found for those hyper-parameters in the grid search.
    '''
    # Grid of parameters over which to perform the grid search.
    param_grid = [
        {'min_samples_split': range(2, 41),
         'min_samples_leaf': range(1, 20)}
    ]

    # Perform the grid search.
    search = GridSearchCV(DecisionTreeClassifier(random_state=0), param_grid)
    start = time.time()
    search.fit(X, y)
    duration = time.time() - start

    # Output to the file the best params, the best score achieved and the time taken to perform the
    # grid search.
    with open(file_name, 'w') as file:
        for param, value in search.best_params_.items():
            file.write(f'{param},{value}\n')
        file.write(f'accuracy,{search.best_score_}\n')
        file.write(f'time,{duration}\n')

    # return the best params found by the grid search.
    return search.best_params_


if __name__ == "__main__":
    '''
    Train decision tree models from UCT games and/or play a series of games between agents.
    '''
    
    # Specification and parsing of command-line arguments.
    parser = argparse.ArgumentParser(description='Train or Play Games')
    parser.add_argument('iter', nargs='*', type=int)
    parser.add_argument('-t', '--games_train', dest='games_per_model', type=int)
    parser.add_argument('-p', '--games_play', dest='games_play', type=int)
    parser.add_argument('-n', '--n_models', dest='n_models', type=int)
    parser.add_argument('-m', '--models', dest='model_names', nargs='*', type=str)
    args = parser.parse_args()
    
    # If the user specified a number of games to train some models, execute the training process
    # for each agent type (agent type = number of MCTS iterations used).
    if args.games_per_model:
        for iteration in args.iter:
            UCTTrainModels(args.n_models, args.games_per_model, iteration)
    
    # If the user specified a number of games to play...
    if args.games_play:
        # If the user generated 'n_models' before, for each agent type, take the last model and
        # it compete against all the previous models of the same agent type.
        if args.n_models:
            for iteration in args.iter:
                model_name = f'DT_{iteration}_{args.n_models}'
                competitors = [f'DT_{iteration}_{i}' for i in range(args.n_models)]
                models_dir = f'models/'
                out_file = f'results/DT_{iteration}.csv'
                UCTPlayTournament(args.games_play, model_name, competitors, models_dir, out_file)
        # If the user specified the names of some models, make them compete between each other (all
        # combinations of two player matches).
        elif args.model_names:
            for match in itertools.combinations(args.model_names, 2):
                model_name, competitor = match
                models_dir = f'models/'
                out_file = f'results/{"-".join(args.model_names)}.csv'
                UCTPlayTournament(args.games_play, model_name, [competitor], models_dir, out_file)
