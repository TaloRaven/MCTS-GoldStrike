import numpy as np
import os
import pickle
import random 
from pathlib import Path
from datetime import datetime
import pandas as pd 
import logging
from tqdm import tqdm
import json

from settings import *
from modules.GoldStrike import GoldStrike
from modules.MCTS import MCTS

import time

class TestMapManager:
    def __init__(self, num_levels=30, num_maps_per_level=1000):
        self.num_levels = num_levels
        self.num_maps_per_level = num_maps_per_level
        self.map_dir = TEST_MAPS_MAIN_PATH  # Directory where test maps are saved
    def generate_test_maps(self):
        for level in range(1, self.num_levels + 1):
            for map_index in range(self.num_maps_per_level):
                os.makedirs(self.map_dir.format(lvl=level), exist_ok=True)
                map = self._generate_map(level, map_index)  # Replace this with your map generation code
                self._save_map(map, level, map_index)
        print(f'TEST MAP SETS CREATED ! {self.map_dir }')
    def load_test_map(self, level, map_index):
        filename = self._get_filename(level, map_index)
        with open(f"{filename}.pkl", "rb") as f:
            return pickle.load(f)

    def _generate_map(self, level, map_index):
        unique_seed = level * map_index + map_index  # Create a unique seed from level and index
        np.random.seed(unique_seed)  # Set the seed
        level_length = get_max_level(level)
        return np.random.randint(1, 5, size=(BOARD_HEIGHT, level_length))

    def _save_map(self, map, level, map_index):
        filename = self._get_filename(level, map_index)
        with open(f"{filename}.pkl", "wb") as f:
            pickle.dump(map, f)

    def _get_filename(self, level, map_index):
        return os.path.join(self.map_dir.format(lvl=level), str(map_index))


class TestGamesManager:
    AVAILABLE_METHODS = ['random_play', 'mcts_play']
    def __init__(self):
        current_time = datetime.now()
        self.main_path = Path(TEST_RESULTS_MAIN_PATH).joinpath(current_time.strftime('%Y_%m_%d_%H_%M_%S'))
        self.list_result_path=self.main_path.joinpath('list_results')
        self.csv_path=self.main_path.joinpath('result.csv')
        self.log_path=str(self.main_path.joinpath('test_games_manager.log'))

    def set_load_result_path(self, path: str):
        self.main_path = path

    def get_file_path(self, level: int) -> Path:
        lvl_path=self.list_result_path.joinpath(f'lvl_{level}')
        if not lvl_path.exists():
            lvl_path.mkdir(exist_ok=True, parents=True)
        return lvl_path.joinpath(f'result_list.pkl')

    def load_results(self, level: int):
        mcts_file = self.get_file_path(level)
        try:
            with open(mcts_file, 'rb') as f:
                    results_mcts = pickle.load(f)
        except Exception:
            results_mcts = []

        return results_mcts

    def save_params(self, params):
        if not self.main_path.exists():
            self.main_path.mkdir(exist_ok=True, parents=True)
        params_file = self.main_path.joinpath('params.txt')
        with open(params_file, 'w') as f:
            for key, value in params.items():
                f.write(f'{key}: {value}\n')
                
    def save_result(self, level, results_mcts: list):
        if not self.main_path.exists():
            self.main_path.mkdir(exist_ok=True, parents=True)
        mcts_file = self.get_file_path(level)
        with open(mcts_file, 'wb') as f:
            pickle.dump(results_mcts, f)


    def make_test_multiprcoess(self, params):
        if not self.main_path.exists():
            self.main_path.mkdir(exist_ok=True, parents=True)
        
        logging.basicConfig(filename=self.log_path, level=logging.INFO,
                            format='%(asctime)s [%(levelname)s] - %(message)s',
                            datefmt='%m/%d/%Y %I:%M:%S %p')
        logging.info('Starting test with params: %s', json.dumps(params, indent=4))
        save_results = params.get('save_results', False)
        repeats_iterations = params.get('repeats_iterations', 200)
        mcts_n_simulations = params.get('mcts_n_simulations', 200)
        mcts_rollout_max_depth = params.get('mcts_rollout_max_depth', 100)
        display_board = params.get('display_board', False)
        min_level = params.get('min_level', 1)
        max_level = params.get('max_level', 10)
        test_function = params.get('test_function', 'random_play')  

        if save_results:
            self.save_params(params)

        test_map_manager = TestMapManager()

        

    def make_test(self, params):
        if not self.main_path.exists():
            self.main_path.mkdir(exist_ok=True, parents=True)

        self.log_path = self.main_path.joinpath('test_games_manager.log')

        logging.basicConfig(filename=self.log_path, level=logging.INFO,
                            format='%(asctime)s [%(levelname)s] - %(message)s',
                            datefmt='%m/%d/%Y %I:%M:%S %p')

        logging.info('Starting test with params: %s', json.dumps(params, indent=4))

        save_results = params.get('save_results', False)
        repeats_iterations = params.get('repeats_iterations', 200)
        mcts_n_simulations = params.get('mcts_n_simulations', 200)
        mcts_rollout_max_depth = params.get('mcts_rollout_max_depth', 100)
        display_board = params.get('display_board', False)
        min_level = params.get('min_level', 1)
        max_level = params.get('max_level', 10)
        test_function = params.get('test_function', 'random_play') 
        
        if save_results:
            self.save_params(params)

        test_map_manager = TestMapManager()

        for level in tqdm(range(min_level, max_level + 1), desc='Level'):
            results_mcts = []
            # Before starting the next level
            logging.info(f"\n\n---------------------------------------------------\n")
            logging.info(f"Starting tests for level {level}")
            logging.info(f"---------------------------------------------------\n\n")

            for iteration in tqdm(range(repeats_iterations), desc='Iteration', leave=False):
                test_map = test_map_manager.load_test_map(level, 21)

                if save_results:
                    results_mcts = self.load_results(level)

                game = GoldStrike(level=level, current_wave=0, single_cube_spawn=SPAWN_SINGLE_CUBES, pre_loaded_map=test_map)
                #TODO import function and parse params 
                if test_function =='random_play':
                    result = self.random_play(game, display_board,test_map)

                elif test_function =='random_greedy':
                    result, _ = self.random_play_greedy(game, display_board,test_map)

                elif test_function =='mcts_play':
                    result, _ = self.mcts_play(game, mcts_n_simulations,mcts_rollout_max_depth, display_board, test_map)

                else:
                    raise Exception(f"Method {test_function} doesn't exists")
                results_mcts.append(result)
                logging.info("%s level: %s result:%s %s/%s, wins:%s/%s",
                             test_function, level, result, len(results_mcts), repeats_iterations, 
                             len([x for x in results_mcts if x == 1]), repeats_iterations)
                if save_results:
                    self.save_result(level, results_mcts)

            logging.info(f"\n\n###################################################\n")
            logging.info(f"Completed tests for level {level}. Total wins: {len([x for x in results_mcts if x == 1])}")
            logging.info(f"\n###################################################\n\n")
            print(results_mcts)
        logging.info('Test completed')
    def random_play(self, game: GoldStrike, printGamestate, preloaded_map=None):
        """
         random moves without any rule
        """
        while game.game_result == GAME_IN_PROGRESS:
            # Get actions and select one based on the policy
            actions = list([x for x in game.get_valid_moves().keys()])
            # Roll move
            action=random.choice(actions)
            
            game.make_move(action,preloaded_map)
            # Print the game state
            if printGamestate:
                game.display_pretty_board()

        return (game.get_game_result(), game.get_total_score())
    
    def random_play_greedy(self, game: GoldStrike, printGamestate, preloaded_map=None):
        """
        Random moves to clear as much of possible moves when no more moves left , make a next wave 
        """
        while game.game_result == GAME_IN_PROGRESS:

            # Get actions and select one based on the policy
            actions = list([x for x in game.get_valid_moves().keys()])

            if len(actions) > 1:
                actions = actions.remove(NEXT_WAVE)

            # Roll move
            action=random.choice(actions)
            
            game.make_move(action,preloaded_map)
            # Print the game state
            if printGamestate:
                game.display_pretty_board()

        return game.game_result
    

    def mcts_play(self, game: GoldStrike, mcts_n_simulations: int,mcts_rollout_max_depth: int, display_board: bool=False,preloaded_map=None, time_move=None):
        mcts = MCTS(game)

        while not mcts.root.game_state.isTerminated():

            mcts.run(mcts_n_simulations, mcts_rollout_max_depth,time_move)
            policy_list=mcts.get_policy()
            action = mcts.select_action()
            mcts.root.game_state.make_move(action,preloaded_map)
            mcts=MCTS(mcts.root.game_state)


            if display_board:
                mcts.root.game_state.display_pretty_board(policy_list)
        

        return mcts.root.game_state.game_result, mcts
    
    def create_csv_dataframe(self):
        level_dict={}
        level_num=1
        for path in self.list_result_path.iterdir():
            with open(path.joinpath('result_list.pkl'), "rb") as f:
                list=pickle.load(f)
                level_num = str(path.stem).split('_')[1]
                level_dict[f'level_{level_num}'] = list

        df = pd.DataFrame(level_dict)
        sorted_columns = sorted(df.columns, key=lambda x: int(x.split('_')[1]))

        
        df.replace(-1, 0, inplace=True)

        df = df[sorted_columns]
        df.to_csv(self.csv_path)


    def alphazero_play(self):
        pass
    
    def __repr__(self):
        return f'<TestGamesManager(main_path={self.main_path}, available_methods={self.AVAILABLE_METHODS})>'
    

class TestGamesSummary():
    pass

    def _check_sample_size():
        pass
    def _check_exists():
        pass

    def _check_schemas():
        pass
