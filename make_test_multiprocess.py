


from modules.TestManager import TestMapManager, TestGamesManager
import time
import logging
'''
D:\PythonProjects\Alpha_gold\.venv\Scripts\python.exe  D:\PythonProjects\Alpha_gold\make_test.py
'''

if __name__ == "__main__":
    #TODO LOOK AHEAD PARAM
    #TODO seed od do 
    #TOTO by zmenijszyć liczbę stanów do liczenia robimy next wave dla 5 ruchów by 
    start_time = time.time()
    msg="""Rework drzewa z rolloutem dla next wave, nieograniczony rollout jednak ograniczona liczba next waveów. test dla wielu leveli . LH=0 """
    game_test = TestGamesManager()
    params = {
        'msg':msg,
        'start_time': start_time,
        'save_results': True,
        'repeats_iterations': 200,
        'mcts_n_simulations': 250,
        'mcts_max_new_wave':3,
        'mcts_greedy_selection':True,
        'mcts_rollout_max_depth': 50,
        'display_board': False,
        'min_level': 5,
        'max_level': 5,
        'test_function': 'mcts_play'
    }
    try:
        game_test.make_test(params)
        game_test.create_csv_dataframe()
        game_test.main_path.joinpath('_SUCCESS').touch()

        end_time = time.time()
        elapsed_time = end_time - start_time
        hours, remainder = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        formatted_time = "{:02}:{:02}:{:02}".format(int(hours), int(minutes), int(seconds))

        logging.info(f"Execution time: {formatted_time}")

    except KeyboardInterrupt as keyboard:
        end_time = time.time()
        elapsed_time = end_time - start_time
        hours, remainder = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        formatted_time = "{:02}:{:02}:{:02}".format(int(hours), int(minutes), int(seconds))

        logging.info(f"Execution interrupted after: {formatted_time} seconds")
        logging.error('Interapted')

        if not game_test.main_path.exists():
            game_test.main_path.mkdir(exist_ok=True, parents=True)
        failed_file = game_test.main_path.joinpath('FAILED.txt')
        with open(failed_file, 'w') as f:
            f.write('PARAMS\n')
            for key, value in params.items():
                f.write(f'{key}: {value}\n')
            f.write('\nERROR\n')
            f.write('Interapted')

    except Exception as err:
        end_time = time.time()
        elapsed_time = end_time - start_time
        hours, remainder = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        formatted_time = "{:02}:{:02}:{:02}".format(int(hours), int(minutes), int(seconds))

        logging.error(f"Execution failed after: {formatted_time} seconds", exc_info=True)

        if not game_test.main_path.exists():
            game_test.main_path.mkdir(exist_ok=True, parents=True)
        failed_file = game_test.main_path.joinpath('FAILED.txt')
        with open(failed_file, 'w') as f:
            f.write('PARAMS\n')
            for key, value in params.items():
                f.write(f'{key}: {value}\n')
            f.write('\nERROR\n')
            f.write(str(err))  # You should write 'err' not 'Exception'
