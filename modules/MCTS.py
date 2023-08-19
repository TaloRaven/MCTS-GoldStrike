
import numpy as np
import random

from settings import *
from modules.GoldStrike import GoldStrike
import time 



class Node:
    def __init__(self,
                  game_state: GoldStrike,
                  parent=None,
                  move=None,
                  policy_prio=0,
                  isTerminated=None):
        
        self.game_state = game_state
        self.parent = parent
        self.move = move

        if isTerminated is  None:
            self.is_terminal = self.game_state.get_game_result() != GAME_IN_PROGRESS
        else:
            self.is_terminal=isTerminated

        self.is_fully_expanded = self.is_terminal
        self.children = {}  # Changed to dictionary
        self.num_visits = 0
        self.total_reward = 0
        self.prior=policy_prio
        self.all_results=[]

    def add_child(self, action, child):
        """Add a new child node for the action."""
        self.children[action] = child  # store children as key-value pairs

    def __repr__(self):
        parent_str = str(self.parent.move) if self.parent is not None else None
        return (
            # f"Node(game_state={self.game_state}, "
            # f"parent={parent_str}, "
            f"move={self.move}, "
            f"is_terminal={self.is_terminal}, "
            # f"is_fully_expanded={self.is_fully_expanded}, "
            # f"children={list(self.children.keys())}, "
            f"num_visits={self.num_visits}, "
            f"total_reward={self.total_reward})"
            # f"policy_prio={self.prio}"
        )


class MCTS:
    def __init__(self, game: GoldStrike):
        self.root = Node(game)
        self.expand(self.root)
        self.D=5000
        self.local_max_score=0

    def update_max_score(self, new_score):
        self.local_max_score = new_score

    def get_max_score(self):
        return self.local_max_score
        
    def set_pervious_child_as_root(self, child: Node):
        self.root = child
        self.root.parent=None
        self.root.move=None

    def modified_UCB(self, node: Node) -> float:
        if node.num_visits == 0:
            return float("inf")  
        
        ucb_value = node.total_reward / node.num_visits + np.sqrt(2 * np.log(node.parent.num_visits) / node.num_visits)

        Px2 = sum(result**2 for result in node.all_results)
        expected_result_squared = node.num_visits * (node.total_reward / node.num_visits)**2
        variability_value = (Px2 - expected_result_squared + self.D*node.game_state.get_safe_value()) / node.num_visits

        return ucb_value + variability_value



    # def UCB1(self, node: Node) -> float:
    #     if node.num_visits == 0:
    #         return float("inf") 
    #     else:
    #         if self.local_max_score < node.total_reward:
    #             self.update_max_score(node.total_reward) 
    #             # print(self.local_max_score)
    #     avg_score = node.total_reward / node.num_visits
    #     normilized_score = avg_score / self.local_max_score
    #     # print(normilized_score)
    #     return normilized_score + np.sqrt(2 * np.log(node.parent.num_visits) / node.num_visits)

    def UCB1(self, node: Node) -> float:
        if node.num_visits == 0:
            return float("inf")  # Prioritize unexplored nodes
        else:
            return node.total_reward / node.num_visits + np.sqrt(2 * np.log(node.parent.num_visits) / node.num_visits)

    def select(self, node: Node) -> Node:
        while True:
            if node.is_terminal:
                return node

            if node.is_fully_expanded:
                if node.children:  # Node has children
                    node = max(node.children.values(), key=self.UCB1)
                else:  # If it has no children, return the node
                    return node
            else:
                return node


    def expand(self, node: Node) -> Node:
        valid_moves = node.game_state.get_valid_moves()
        for action in valid_moves:

            if action not in node.children:
                if action != NEXT_WAVE:
                    new_game_state = node.game_state.get_copy()
                    new_game_state.make_move(action)
                    child_node = Node(new_game_state, parent=node, move=action)
                    node.add_child(action, child_node)

                else:
                    new_game_state = node.game_state.get_copy()
                    _check=new_game_state.next_wave_still_block()  
                    del new_game_state
                    copy_state=node.game_state.get_copy()
                    child_node = Node(copy_state, parent=node, move=NEXT_WAVE,isTerminated=_check)
                    node.add_child(NEXT_WAVE, child_node)

        # if next wave == -1 and len(actions)>1 drop next wave move 
        if _check == 1 and len(valid_moves) > 1:
            del valid_moves[NEXT_WAVE]

        node.is_fully_expanded = True
        return node

    def rollout_guided(self, node: Node, max_steps: int = 200, max_next_waves=8) -> float:
        current_state = node.game_state.get_copy()
        step_count = 0
        next_waves=0

        #Rollout for child node is made in a sens of making alwas random next wave, there will be no child node next wave if state after next wave was terminated 
        if node.move == NEXT_WAVE:
            current_state.make_move(NEXT_WAVE)
            # step_count += 1  
            next_waves+=1

        while not current_state.isTerminated() and step_count < max_steps:
            valid_moves = current_state.get_valid_moves()
            actions = list(valid_moves.keys())

            if len(actions)>1:
                if np.count_nonzero(current_state.board[-1]) + 1 > BOARD_WIDTH:
                    actions.remove(NEXT_WAVE)

            if next_waves >= max_next_waves: 
                if NEXT_WAVE in actions:
                    actions.remove(NEXT_WAVE) 
                if len(actions)==0:
                    break
            action = np.random.choice(actions)

            if action == NEXT_WAVE:
                next_waves+=1
            current_state.make_move(action)
        
            step_count += 1

        survival_rate=step_count/max_steps    
        value = self.get_state_value(current_state,survival_rate)


        del current_state
        return value
    
    def rollout_guided_tabucolor(self, node: Node, max_steps: int = 200, max_next_waves=8) -> float:
        current_state = node.game_state.get_copy()
        step_count = 0
        next_waves=0

        #Rollout for child node is made in a sens of making alwas random next wave, there will be no child node next wave if state after next wave was terminated 
        if node.move == NEXT_WAVE:
            current_state.make_move(NEXT_WAVE)
            # step_count += 1  
            next_waves+=1

        while not current_state.isTerminated() and step_count < max_steps:
            valid_moves = current_state.get_valid_moves()
            group_color={k:v[0][2] for k,v in valid_moves.items() if k != NEXT_WAVE}
            group_color.update({NEXT_WAVE:[]})
            actions = list(valid_moves.keys())

            if len(actions)>1:
                if np.count_nonzero(current_state.board[-1]) + 1 > BOARD_WIDTH:
                    actions.remove(NEXT_WAVE)
                    del group_color[NEXT_WAVE]



            # if next_waves >= max_next_waves: 
            #     if NEXT_WAVE in actions:
            #         actions.remove(NEXT_WAVE)
            #         del group_color[NEXT_WAVE]

            #     if len(actions)==0:
            #         break

            if GOLD_CUBES in group_color.values():
                actions_without_gold = [a for a in actions if group_color.get(a) if group_color[a] != GOLD_CUBES]

                # If there are actions left after removing gold moves, use those actions
                if actions_without_gold:
                    actions = actions_without_gold

            action = np.random.choice(actions)

            if action == NEXT_WAVE:
                next_waves+=1
                current_state.next_wave_still_block()
            else:
                current_state.make_move(action)

            step_count += 1

        survival_rate=step_count/max_steps    
        value = self.get_state_value(current_state,survival_rate)


        del current_state
        return value
    
    def rollout_simple(self, node: Node, max_steps: int = 200, max_next_waves=8) -> float:
        current_state = node.game_state.get_copy()
        step_count = 0
        next_waves=0

        #Rollout for child node is made in a sens of making alwas random next wave, there will be no child node next wave if state after next wave was terminated 
        if node.move == NEXT_WAVE:
            current_state.make_move(NEXT_WAVE)
            # step_count += 1  
            next_waves+=1

        while not current_state.isTerminated() and step_count < max_steps:
            valid_moves = current_state.get_valid_moves()
            actions = list(valid_moves.keys())
            action = np.random.choice(actions)
            current_state.make_move(action)
            step_count += 1

        survival_rate=step_count/max_steps    
        value = self.get_state_value(current_state,survival_rate)


        del current_state
        return value
    
    def get_state_value(self, current_state: GoldStrike, survival_rate):
        # value = current_state.game_result
        # if value == GAME_IN_PROGRESS or value==GAME_WON:
        #     value = current_state.get_total_score() * current_state.get_safe_value()
        # elif value == GAME_LOST:
        #     value = -(1 - survival_rate) * current_state.get_total_score() 

        # current_state.get_total_score()



        return current_state.get_total_score() * current_state.get_safe_value()
    
    def get_prio_of_state(self, game_state):
        return 0  # Or any other default value
    

    def backpropagate(self, node: Node, value: float) -> None:
        while node is not None:
            node.num_visits += 1
            node.total_reward += value
            node.all_results.append(value)  
            node = node.parent

    def _search(self):
        selected_node = self.select(self.root)
        if selected_node.move !=NEXT_WAVE:
            expanded_node = self.expand(selected_node)
            value = self.rollout_guided_tabucolor(expanded_node, self.mcts_rollout_max_depth)
            self.backpropagate(expanded_node, value)

        else:
            value = self.rollout_guided_tabucolor(selected_node, self.mcts_rollout_max_depth)
            self.backpropagate(selected_node, value)

    def run(self, num_simulations,mcts_rollout_max_depth,time_limit=None) -> Node:
        
        self.mcts_rollout_max_depth=mcts_rollout_max_depth
        if time_limit is None:
            for iteration in range(num_simulations):
                self._search()
        else:
            start_time = time.time()  
            time_limit=5
            while True:
                if time.time() - start_time > time_limit:  
                    print("Time limit reached!")
                    break
            self._search()


    def get_policy(self, temperature=1) -> list:
        visits = np.array([child.num_visits for child in self.root.children.values()], dtype=float)

        if temperature == 0:  
            action_probs = np.zeros_like(visits)
            action_probs[np.argmax(visits)] = 1.0
        else:  
            visits_raised = np.power(visits, 1.0/temperature)
            action_probs = visits_raised / np.sum(visits_raised)

        return action_probs

    def get_action_list(self):
        return [x for x in self.root.game_state.get_valid_moves().keys()]
    
    def select_action(self, greedy=True) -> int:
        policy = self.get_policy()
        valid_moves_list=self.get_action_list()
        
        if greedy:
            list_index=np.argmax(policy)
        else:
            list_index=np.random.choice(len(policy), p=policy)
        return valid_moves_list[list_index]
