
import numpy as np
from collections import deque
from termcolor import colored
from collections import defaultdict
from copy import deepcopy
import os
from settings import *
from IPython.display import clear_output
import torch


class GoldStrike:

    def __init__(self, level: int=1,
                current_wave: int=0,
                single_cube_spawn: bool=SPAWN_SINGLE_CUBES,pre_loaded_map=None):
        
        self.board = np.zeros((BOARD_HEIGHT, BOARD_WIDTH), dtype=int)
        self.level = level
        self.current_wave = current_wave
        self.max_wave = get_max_level(level)

        self.column_cleared = 0 
        self.safe_column_space= self.max_wave - BOARD_WIDTH
        
        self.progression = round(self.current_wave / self.max_wave, 2)
        self.game_result=GAME_IN_PROGRESS
        self.single_cube_spawn = single_cube_spawn
        self.pre_loaded_map = pre_loaded_map
        self.incoming_waves = self._generate_incoming_waves(pre_loaded_map)
        self.total_score=0
        self.extra_bonus=0

    def score_break_gold(self,cubes_to_drop_len):
        self.total_score+=(cubes_to_drop_len*(10*cubes_to_drop_len))

    def post_game_column_cleared_bonus(self):
        # This checks if all values in a column are 0
        self.board[self.board == 6] = 0
        
        number_of_rows=np.sum(np.all(self.board == 0, axis=0))
        self.extra_bonus=self.level*number_of_rows*100
        self.total_score+=self.extra_bonus

    def reset_incoming_waves(self,pre_loaded_map):
        self.incoming_waves=self._generate_incoming_waves(pre_loaded_map)

    def _generate_incoming_waves(self, pre_loaded_map) -> deque:

        # Initialize incoming_waves with pre_loaded_map if provided
        incoming_waves = deque()

        if pre_loaded_map is not None:
            # Get the initial waves from the pre_loaded_map
            for i in range(1):
                if i < pre_loaded_map.shape[1]:
                    wave = pre_loaded_map[:, i].reshape((BOARD_HEIGHT, 1))
                    incoming_waves.append(wave)
        else:
            # Else, generate waves randomly
            for _ in range(3):
                new_wave = np.random.randint(1, 5, size=(BOARD_HEIGHT, 1))
                incoming_waves.append(new_wave)
        return incoming_waves

    def get_safe_value(self):
        if self.column_cleared >= self.safe_column_space:
            return 1.0
        else:
            return round(self.column_cleared / self.safe_column_space, 2)
    
    def get_level_number(self):
        return self.level
    
    def get_progression(self):
        return self.progression

    def get_game_result(self):
        return self.game_result
    
    def get_total_score(self):
        return self.total_score

    def isTerminated(self):
        return self.game_result != GAME_IN_PROGRESS
    
    def isIncoming_waves(self):
        return len(self.incoming_waves) != 0
    
    def _determine_game_outcome(self) -> None:
        if np.count_nonzero(self.board[-1]) + 1 > BOARD_WIDTH:
            self.game_result = GAME_LOST  # lose

        elif self.current_wave + 1 > self.max_wave:
            self.game_result = GAME_WON  # win
            self.post_game_column_cleared_bonus()

        else: 
            self.current_wave += 1 # Game in progress, adding wave 

    def _add_wave(self):
        new_wave = self.incoming_waves.popleft()  # Get the new wave



    def _spawn_single_cube(self):
        #TODO nie działa 
        pass
        # if self.single_cube_spawn:
        #     # Generate a random number and only spawn a cube if the number is less than or equal to SINGLE_CUBE_SPAWN_CHANCE
        #     if np.random.random() <= SINGLE_CUBE_SPAWN_CHANCE:
        #         # Check if board is empty
        #         if np.any(self.board):
        #             # Find the column with the rightmost non-empty cell
        #             max_filled_column = max(np.nonzero(self.board)[1])
        #             # Find the columns that are not fully filled and are less than or equal to max_filled_column
        #             valid_columns = [i for i in range(max_filled_column + 1) if 0 in self.board[:, i]]
        #         else:
        #             # If board is empty, valid columns are up to the rightmost column of the new wave
        #             valid_columns = list(range(self.incoming_waves[0].shape[1]))

        #         # Select a random column among the valid ones
        #         selected_column = np.random.choice(valid_columns)
        #         for i in range(BOARD_HEIGHT - 1, -1, -1):
        #             if self.board[i, selected_column] == 0:
        #                 # Fill the cell with a random cube
        #                 self.board[i, selected_column] = np.random.choice(range(1, 5))
     
        #                 break


    def next_wave_still_block(self):
        self._determine_game_outcome()
        if not self.isTerminated():
            new_wave = np.full((BOARD_HEIGHT, 1), 6)
            new_wave_cols = deque(new_wave.T)  

            # Start with the new wave on the left
            updated_board = list(new_wave_cols)

            for col in self.board.T:
                # Append each column from the existing board
                updated_board.append(col)

            while len(updated_board) > BOARD_WIDTH:
                gap_indices = [i for i, col in enumerate(updated_board) if np.all(col == 0)]
                
                if gap_indices:
                    del updated_board[gap_indices[0]]
                else:  # If no columns with gaps were found
                    # Remove the leftmost column
                    updated_board.pop(0)

            
            self.progression = round(self.current_wave / self.max_wave, 2)

            self.board = np.array(updated_board).T  

        return self.isTerminated()
    
    def guided_rollout_with_still_blocks(self):

        pass        


    def next_wave(self, pre_loaded_map=None):
        
        self._determine_game_outcome()

        if not self.isTerminated():

            # # Only add a new wave if game is not complited
            # if self.max_wave - self.current_wave >= LOOK_AHEAD: #!!
            if pre_loaded_map is not None:
                new_wave = pre_loaded_map[:, self.current_wave:self.current_wave+1]
            else:
                new_wave = np.random.randint(1, 5, size=(BOARD_HEIGHT, 1)) 
            # self.incoming_waves.append(new_wave)
    
            new_wave_cols = deque(new_wave.T)  

            updated_board = list(new_wave_cols)

            for col in self.board.T:
                updated_board.append(col)

            while len(updated_board) > BOARD_WIDTH:
                gap_indices = [i for i, col in enumerate(updated_board) if np.all(col == 0)]
                
                if gap_indices:
                    del updated_board[gap_indices[0]]
                else:  
                    updated_board.pop(0)

            
            self.progression = round(self.current_wave / self.max_wave, 2)

            self.board = np.array(updated_board).T  # Transpose back to original form

            self._spawn_single_cube()

        return self.isTerminated()
    
    def clear_board(self):
        # Code to clear the board when a level is completed
        self.board = np.zeros((16,10), dtype=int)

    def display(self):
        clear_output(wait=True)  # clear the output of the cell
        print(f'Level: {self.level}\n'
              f'Current wave: {self.current_wave}/{self.max_wave}\n'
              f'Progression: {self.progression}\n'
              f'Board:')
        print(self.board)

    def display_pretty_board(self, policy=None):
        clear_output(wait=True)  # clear the output of the cell
        color_map = {0: 'black', 1: 'red', 2: 'green', 3: 'yellow', 4: 'blue', 'black':6}
        print(f'Level: {self.level}\n'
              f'Current wave: {self.current_wave}/{self.max_wave}\n'
              f'Progression: {self.progression}\n'
              f'Score: {self.total_score}\n'
              f'Bonus: {self.extra_bonus}\n'
              f'Board:')
        for row in self.board:
            for cell in row:
                print(colored(f'[{cell}]', color_map[cell]), end=' ')  
            print()
    def display_pretty_board_coordinates(self,clear_terminal_first=True):
        clear_output(wait=True)  # clear the output of the cell
        if clear_terminal_first:
            os.system('cls' if os.name == 'nt' else 'clear')
        color_map = {0: 'black', 1: 'red', 2: 'green', 3: 'yellow', 4: 'blue'}
        for y, row in enumerate(self.board):
            for x, cell in enumerate(row):
                print(colored(f'[{y},{x}:{cell}]', color_map[cell]), end=' ')  
            print()  
    

    def _dfs(self, i, j, value, group_id):
        stack = [(i, j)]
        while stack:
            x, y = stack.pop()
            if x < 0 or x >= len(self.board) or y < 0 or y >= len(self.board[0]) or self.visited[x][y] or self.board[x][y] != value:
                continue
            self.visited[x][y] = group_id
            stack.extend([(x-1, y), (x+1, y), (x, y-1), (x, y+1)])

    def _find_groups(self):
        """
        
        """
        self.visited = [[0]*len(self.board[0]) for _ in range(len(self.board))]
        group_id = 0
        group_dict = defaultdict(list)

        for i in range(len(self.board)):
            for j in range(len(self.board[0])):
                if self.visited[i][j] == 0 and self.board[i][j] != 0:
                    group_id += 1
                    self._dfs(i, j, self.board[i][j], group_id)

        for i in range(len(self.board)):
            for j in range(len(self.board[0])):
                if self.visited[i][j] != 0:
                    group_dict[self.visited[i][j]].append((i, j, self.board[i][j]))

        return group_dict
    
    def get_valid_moves(self) -> dict:
        """
        Find possible moves to make
        """
        groups = self._find_groups()
        valid_moves = {}

        # Filter out the groups with only one element
        for group_num in groups.keys():
            if groups[group_num][0][2] == 6:
                continue
            elif len(groups[group_num]) > 1:
                valid_moves[group_num] = groups[group_num]

        # Reindex the keys of valid_moves
        reindexed_moves = {i+1: group for i, group in enumerate(valid_moves.values()) }
        reindexed_moves.update({NEXT_WAVE: []})
        return reindexed_moves 
    
    def _remove_group_and_apply_gravity(self, group):
        """
         Update state after making a move that resolve in breaking cubes
        """
        # Count the number of empty columns before gravity
        empty_before = np.sum(np.all(self.board == 0, axis=0))

        # Remove group
        for cell in group:
            self.board[cell[0], cell[1]] = 0

        # Apply gravity

        for j in range(BOARD_WIDTH):
            column = self.board[:, j]
            mask = column != 0
            self.board[:, j] = np.concatenate([column[~mask], column[mask]])

        # Count the number of empty columns after gravity
        empty_after = np.sum(np.all(self.board == 0, axis=0))

        # Increase the number of cleared columns by the new empty columns
        self.column_cleared += max(0, empty_after - empty_before)

    def make_move(self, action, preloaded_map=None):
        """
        Make a decision to break available groups of cubes or spawn next wave
        """
        _val_moves=self.get_valid_moves()
        
        if action == NEXT_WAVE:
            
            self.next_wave() if preloaded_map is  None else self.next_wave(preloaded_map)
        else:
            _, _, group_number=_val_moves[int(action)][0]
            
            if group_number == GOLD_CUBES:
                self.score_break_gold(len(_val_moves[int(action)]))
            self._remove_group_and_apply_gravity(_val_moves[int(action)])

    def get_copy(self):
        """
        Create copy of GoldStrike instance for MCTS expand function
        """
        # Create a new GoldStrike object
        new_game_state = GoldStrike(level=self.level, current_wave=self.current_wave, single_cube_spawn=self.single_cube_spawn)

        # Copy all the properties
        new_game_state.board = np.copy(self.board)
        new_game_state.level = self.level
        new_game_state.current_wave = self.current_wave
        new_game_state.max_wave = self.max_wave
        new_game_state.progression = self.progression
        new_game_state.incoming_waves = self._generate_incoming_waves(pre_loaded_map=None) #! 
        # new_game_state.incoming_waves = self.incoming_waves
        new_game_state.game_result = self.game_result
        new_game_state.total_score = self.total_score
        return new_game_state
    
    def get_encoded_state(self):
        # Initialize the encoded state as a 8-depth 3D array filled with zeros
        encoded_state = np.zeros((8, self.board.shape[0], self.board.shape[1]))

        # One-hot encode the colors on the board
        for color in range(1, 5):  # colors range from 1 to 4
            encoded_state[color-1] = (self.board == color).astype(int)

        # Add 'no block' state
        encoded_state[4] = (self.board == 0).astype(int)

        # Add normalized level, progression, safe_net information
        encoded_state[5] = self.safe_column_space
        encoded_state[6] = self.progression


        # Convert the numpy array to a PyTorch tensor
        encoded_state = torch.from_numpy(encoded_state).float()
        encoded_state = encoded_state.unsqueeze(0)

        return encoded_state


