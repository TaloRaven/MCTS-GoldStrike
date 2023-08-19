from modules.TestManager import TestMapManager, TestGamesManager
from modules.GoldStrike import GoldStrike
from modules.MCTS import MCTS
# game_test=TestGamesManager()
# game_manager=TestMapManager()
import pickle
import time

import multiprocessing as mp
from functools import partial

'''
D:\PythonProjects\Alpha_gold\.venv\Scripts\python.exe  D:\PythonProjects\Alpha_gold\debug.py
'''
N_ITER=100
M_SIM=100
M_ROLL=100
LEVEL=20




def mcts_play(args):
    game, preloaded_map = args
    mcts_n_simulations = M_SIM
    mcts_rollout_max_depth = M_ROLL
    display_board = False
    mcts = MCTS(game)

    while not mcts.root.game_state.isTerminated():
        mcts.run(mcts_n_simulations, mcts_rollout_max_depth)
        action = mcts.select_action()
        mcts.root.game_state.make_move(action, preloaded_map)
        mcts=MCTS(mcts.root.game_state)

        if display_board:
            mcts.root.game_state.display_pretty_board()
    
    print(mcts.root.game_state.game_result)
    return mcts.root.game_state.game_result

def mcts_play(args):
    game, preloaded_map = args
    mcts_n_simulations = M_SIM
    mcts_rollout_max_depth = M_ROLL
    display_board = False
    mcts = MCTS(game)

    while not mcts.root.game_state.isTerminated():
        mcts.run(mcts_n_simulations, mcts_rollout_max_depth)
        action = mcts.select_action()
        mcts.root.game_state.make_move(action, preloaded_map)
        mcts=MCTS(mcts.root.game_state)

        if display_board:
            mcts.root.game_state.display_pretty_board()
    
    print(mcts.root.game_state.game_result)
    return mcts.root.game_state.game_result




if __name__ == '__main__':
    print({
        'level':LEVEL,
        'm_sim':M_SIM,
        'rolls':M_ROLL,
        
    })
    start_time = time.time()
    test_map_manager = TestMapManager()
    game_states = [GoldStrike(LEVEL) for _ in range(N_ITER)]

    test_maps = [test_map_manager.load_test_map(LEVEL, iteration) for iteration in range(0, N_ITER)]

    game_map_pairs = list(zip(game_states, test_maps))

    pool = mp.Pool(processes=6)
    results = pool.map(mcts_play, game_map_pairs)

    print(results)
    end_time = time.time()
    elapsed_time = end_time - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    formatted_time = "{:02}:{:02}:{:02}".format(int(hours), int(minutes), int(seconds))
    print(formatted_time)
    # Save the results list to a file using pickle
    with open(r'D:\PythonProjects\Alpha_gold\data\results.pickle', 'wb') as f:
        pickle.dump(results, f)

