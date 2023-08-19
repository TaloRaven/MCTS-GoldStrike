import logging
import json
from tqdm import tqdm
import multiprocessing

from modules.GoldStrike import GoldStrike
from modules.MCTS import MCTS
from settings import *
LOOK_AHEAD
import datetime
import os
import numpy as np
from collections import deque
from termcolor import colored
from collections import defaultdict
# SETTINGS Constants
from copy import deepcopy
import random
import torch.nn as nn
import torch.nn.functional as F
import torch
from torchsummary import summary
import matplotlib.pyplot as plt


from modules.TestManager import TestMapManager, TestGamesManager
import time
from pathlib import Path
import datetime
import pickle
import numpy as np
from concurrent.futures import ThreadPoolExecutor
'''
D:\PythonProjects\Alpha_gold\.venv\Scripts\python.exe  D:\PythonProjects\Alpha_gold\score_test.py
'''
NAME_TEST='rollout_simple'


import logging
import json
from tqdm import tqdm
import multiprocessing

from modules.GoldStrike import GoldStrike
from modules.MCTS import MCTS
# ... your other imports ...

if not os.path.exists('D:\PythonProjects\Alpha_gold\data\TEST_RESULTS'):
    os.makedirs('logs')

current_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")

def run_simulation(number_of_sims, level):
    results = {f'{number_of_sims}': []}
    
    # Removed the for loop from here
    test_map = TestMapManager().load_test_map(level, 21)
    game = GoldStrike(level, pre_loaded_map=test_map)
    [game.next_wave(test_map) for _ in range(5)]

    mcts = MCTS(game)
    moves = 0
    while not mcts.root.game_state.isTerminated():
        mcts.run(number_of_sims, 50, None)
        action = mcts.select_action()
        mcts.root.game_state.make_move(action, preloaded_map=test_map)
        # mcts = MCTS(mcts.root.game_state)
        if action != NEXT_WAVE:
            #To keep this branch simulation history if we didnt make next wave 
            mcts.set_pervious_child_as_root(mcts.root.children[action])
            # temp_num_sims = 100
        else:
            mcts = MCTS(mcts.root.game_state)
            # temp_num_sims = number_of_sims

        moves += 1

    results[str(number_of_sims)].append((mcts.root.game_state.get_game_result(), mcts.root.game_state.get_total_score()))
        
    return results

def run_simulation_pool(params):
    number_of_sims, level = params
    return run_simulation(number_of_sims, level)
# Convert numpy types to native Python types before saving to JSON
def default_converter(o):
    if isinstance(o, np.integer):
        return int(o)
    raise TypeError

def test_function(args):
    # num_cpu = multiprocessing.cpu_count()
    num_cpu = args['cpu']
    print(num_cpu)
    
    parameters = [(sim_n, args['level']) for sim_n in args['mcts_simulations_n'] for _ in range(args['n_tests'])]

    with multiprocessing.Pool(processes=int(num_cpu)) as pool:
        simulation_results = list(tqdm(pool.imap_unordered(run_simulation_pool, parameters), total=len(parameters)))

    combined_results = defaultdict(list)
    for res in simulation_results:
        for key, val in res.items():
            combined_results[key].extend(val)

    print(combined_results)

    with open(f"D:\PythonProjects\Alpha_gold\data\TEST_RESULTS\{NAME_TEST}_{current_time}_level{args['level']}_ntest{args['n_tests']}.json", 'w') as f:
        json.dump(combined_results, f, default=default_converter)

if __name__ == "__main__":
    multiprocessing.freeze_support()
    # 100 na 100 mamy jeszcze 200 i 50 i call it for the day 
    print(NAME_TEST)
    args = {
        'cpu':4,
        'n_tests':100,
        'mcts_simulations_n': [100],
        'level': 10
    }
    test_function(args)