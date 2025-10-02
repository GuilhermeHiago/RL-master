import random
import pickle

"""
    Piece 11 = Player1 piece size 1
    Piece 12 = Player1 piece size 2
    Piece 13 = Player1 piece size 3
"""
class GlobbletGobblers:
    def __init__(self):
        self.board = [[0] for _ in range(9)] # [[0]] * 9 gives 9 ref to same list
        self.player1 = 10
        self.player2 = 20

        self.player_pieces = {self.player1: [11, 11, 12, 12, 13, 13], 
                              self.player2: [21, 21, 22, 22, 23, 23]}
        
        self.current_player = self.player1
        pass

    def get_state(self) -> tuple:
        # state = [x[-1] for x in self.board]

        state = [[-1, -1, -1, -1] for _ in range(9)]

        # copy board pieces in stack positions
        for i in range(len(self.board)):
            piece_stack = self.board[i]
            for p in range(len(piece_stack)):
                state[i][p] = piece_stack[p]
            
            # convert stack to tuple
            state[i] = tuple(state[i])

        state.append(self.player_pieces[self.player1].count(11))
        state.append(self.player_pieces[self.player1].count(12))
        state.append(self.player_pieces[self.player1].count(13))
        state.append(self.player_pieces[self.player2].count(21))
        state.append(self.player_pieces[self.player2].count(22))
        state.append(self.player_pieces[self.player2].count(23))

        state.append(self.current_player)
        
        return tuple(state)
    
    def get_flat_state(self) -> tuple:
        # state = [x[-1] for x in self.board]

        state = []

        for i in range(len(self.board)):
            piece_stack = self.board[i]
            for p in range(4):
                if p >= len(piece_stack):
                    state.append(-1)
                else:
                    state.append(piece_stack[p])

        state.append(self.player_pieces[self.player1].count(11))
        state.append(self.player_pieces[self.player1].count(12))
        state.append(self.player_pieces[self.player1].count(13))
        state.append(self.player_pieces[self.player2].count(21))
        state.append(self.player_pieces[self.player2].count(22))
        state.append(self.player_pieces[self.player2].count(23))

        state.append(self.current_player)
        
        return state

    def get_state_old1(self) -> tuple:
        # add only the top pieces of the board
        state = [x[-1] for x in self.board]

        # add number of each pieces last outside
        state.append(self.player_pieces[self.player1].count(11))
        state.append(self.player_pieces[self.player1].count(12))
        state.append(self.player_pieces[self.player1].count(13))
        state.append(self.player_pieces[self.player2].count(21))
        state.append(self.player_pieces[self.player2].count(22))
        state.append(self.player_pieces[self.player2].count(23))

        return tuple(state)
    
    def is_game_over(self):
        if self.check_win(10) or self.check_win(20): # or self.is_draw()
            return True
        return False
    
    def check_win(self, player):
        """ player == 10 or player == 20 """
        win_conditions = [(0, 1, 2), (3, 4, 5), (6, 7, 8),
                          (0, 3, 6), (1, 4, 7), (2, 5, 8),
                          (0, 4, 8), (2, 4, 6)]

        return any(all(self.board[i][-1] != 0 and self.board[i][-1] >> 3 == player >> 3 for i in wc) for wc in win_conditions)
    
    def is_draw(self):
        return self.check_win(self.player1) and self.check_win(self.player2)
    
    def get_available_actions(self, player):
        # moves from initial placement
        # placement_pos = [i for i, cell in enumerate(self.board) if cell[-1] == 0]

        # placement_moves = [("p", piece, pos) for piece in set(self.player_pieces[player]) for pos in placement_pos]

        # placement including smaller piece covering
        placement_moves = [("p", piece, pos) for piece in set(self.player_pieces[player]) for pos, cell in enumerate(self.board) if cell[-1] % 10 < piece % 10]

        move_moves = [("m", orig, dest) for orig in range(9) for dest in range(9) if self.board[orig][-1] >> 3 == player >> 3 and orig != dest and self.board[orig][-1] % 10 > self.board[dest][-1] % 10]

        return placement_moves + move_moves
    
    def get_possible_piece_moves(self, player):
        directions = [
            (-1, 0),  # cima
            (1, 0),   # baixo
            (0, -1),  # esquerda
            (0, 1),   # direita
            (-1, -1), # diagonal superior esquerda
            (-1, 1),  # diagonal superior direita
            (1, -1),  # diagonal inferior esquerda
            (1, 1)    # diagonal inferior direita
        ]

        move_moves = []
        # moves from piece move
        for p in range(len(self.board)):
            if self.board[p][-1] >> 3 == player >> 3:
                # grid pos from list pos
                row, col = p // 3, p % 3

                for dr, dc in directions:
                    new_row, new_col = row + dr, col + dc

                    # if pos inside grid
                    if 0 <= new_row < 3 and 0 <= new_col < 3:
                        neighbor_index = new_row * 3 + new_col
                        
                        # if piece in p is bigger than piece in the neighbor_index
                        if self.board[p][-1] % 10 > self.board[neighbor_index][-1] % 10:
                            move_moves.append(('m', p, neighbor_index))
    
    def make_move(self, action, player):
        if action in self.get_available_actions(player):
            # action <m, orig_pos, targ_pos>
            if action[0] == 'm':
                if self.board[action[1]][-1] >> 3 == player >> 3:
                    if self.board[action[1]][-1] % 10 > self.board[action[2]][-1] % 10:
                        piece = self.board[action[1]].pop()
                        self.board[action[2]].append(piece)
                        self.current_player = self.player1 if self.current_player == self.player2 else self.player2
                        return True
            # action <p, piece, targ_pos>
            elif action[0] == 'p':
                # if is piece from right player and it has this piece
                if action[1] >> 3 == player >> 3 and action[1] in self.player_pieces[player]:
                    if action[1] % 10 > self.board[action[2]][-1] % 10:
                        self.board[action[2]].append(action[1])
                        self.player_pieces[player].remove(action[1])
                        self.current_player = self.player1 if self.current_player == self.player2 else self.player2
                        return True
                    
        print("Falhou movimento")
        return False

    def reset(self):
        self.board = [[0] for _ in range(9)]

        self.player_pieces = {self.player1: [11, 11, 12, 12, 13, 13], 
                              self.player2: [21, 21, 22, 22, 23, 23]}

    def draw_board(self):
        """
            Desenha o tabuleiro do Jogo GobbltGobblers no console.
            'X1', 'X2', 'X3' para o jogador 1, 'O1', 'O2', 'O3' para o jogador 2, e ' ' para espaços vazios.
        """
        
        print("\n")
        print(" " + self.number_to_draw(self.board[0][-1]) + " | " + self.number_to_draw(self.board[1][-1]) + " | " + self.number_to_draw(self.board[2][-1]) + " ")
        print("----+----+----")
        print(" " + self.number_to_draw(self.board[3][-1]) + " | " + self.number_to_draw(self.board[4][-1]) + " | " + self.number_to_draw(self.board[5][-1]) + " ")
        print("----+----+----")
        print(" " + self.number_to_draw(self.board[6][-1]) + " | " + self.number_to_draw(self.board[7][-1]) + " | " + self.number_to_draw(self.board[8][-1]) + " ")
        print("\n")

    def number_to_draw(self, number) -> str:
        if number == 0:
            return '  '

        char = 'X' if number >> 3 == self.player1 >> 3 else 'O'
        return char + str(number % 10)

class QLearningAgent:
    def __init__(self, learning_rate=0.1, discount_factor=0.9, exploration_rate=1.0, q_table=None):
        self.q_table = q_table if q_table is not None else {}
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.min_exploration_rate = 0.01

    def get_q_value(self, state, action):
        return self.q_table.get(state, {}).get(action, 0.0)

    def choose_action(self, state, available_actions):
        if state not in self.q_table:
            # add unknow to all actions if not visited state
            self.q_table[state] = {a: 0 for a in available_actions}

        # print('STATE: ', state)

        if random.uniform(0, 1) < self.exploration_rate:
            return random.choice(available_actions)
        else:
            q_values = self.q_table.get(state, {})
            if not q_values:
                return random.choice(available_actions)
            
            best_action = None
            max_q = -float('inf')
            
            for action in available_actions:
                q_value = q_values.get(action, 0)
                if q_value > max_q:
                    max_q = q_value
                    best_action = action
            
            return best_action

    def update_q_table(self, state, action, reward, next_state):
        max_q_next = 0
        if next_state in self.q_table:
            max_q_next = max(self.q_table[next_state].values())

        old_q = self.get_q_value(state, action)
        new_q = old_q + self.learning_rate * (reward + self.discount_factor * max_q_next - old_q)
        self.q_table[state][action] = new_q

    # def decay_exploration_rate(self, episode, total_episodes):
    #     decay_base = 0.99
    #     self.exploration_rate = self.min_exploration_rate + (1.0 - self.min_exploration_rate) * (decay_base ** episode)
    
    def decay_exploration_rate(self, episode, total_episodes):
        if episode < total_episodes:
            return max(self.min_exploration_rate, 1.0 - episode / total_episodes)
        else:
            return self.min_exploration_rate

# --- Funções de Salvar e Carregar ---
def save_model(agent, filename="q_table.pkl"):
    """Salva a Q-table do agente em um arquivo."""
    with open(filename, "wb") as f:
        pickle.dump(agent.q_table, f)
    print(f"Modelo salvo em {filename}")

def load_model(filename="q_table.pkl"):
    """Carrega a Q-table de um arquivo."""
    try:
        with open(filename, "rb") as f:
            q_table = pickle.load(f)
            print(f"Modelo carregado de {filename}. Tamanho: {len(q_table)}")
            return q_table
    except FileNotFoundError:
        print(f"Arquivo {filename} não encontrado. Iniciando do zero.")
        return None

class RandomAgent():
    def choose_action(self, state, available_actions):
        return random.choice(available_actions)

# --- Loop de Treinamento ---
def train(agent, total_episodes=20000, debug=False):
    game = GlobbletGobblers()
    oponent_agent = RandomAgent()
    last_table_update = 0

    for episode in range(total_episodes):
        game.reset()
        is_game_over = False
        
        if episode == total_episodes // 2:
            agent = QLearningAgent(q_table=mirro_q_table(agent.q_table.copy()))
            last_table_update = 0
        if episode >= total_episodes // 2 and last_table_update >= 300000:
            last_table_update = 0
            agent = QLearningAgent(q_table=mirro_q_table(agent.q_table.copy()))

        last_table_update += 1

        while not is_game_over:
            state = game.get_state()
            available_actions = game.get_available_actions(game.player1)
            
            # Agent's turn (Player 1)
            action = agent.choose_action(state, available_actions)
            game.make_move(action, game.player1)

            
            if debug:
                print("P1 action: ", action)
                game.draw_board()

            reward = 0
            
            # case is placement move and covers other player piece
            if 'p' == action[0]: 
                # -2 because action already made (player 1 piece already on top)
                if game.board[action[2]][-2] >> 3 == game.player2 >> 3:
                    reward = 0.1
                if game.board[action[2]][-2] >> 3 == game.player1 >> 3:
                    reward = -0.1

            if game.is_game_over():
                if game.check_win(10): reward = 100
                elif game.is_draw(): reward = 50
                elif game.check_win(20): reward = -100
                is_game_over = True

                if debug:
                    print("game over Player1", reward)
            
            agent.update_q_table(state, action, reward, game.get_state())

            # Opponent's turn (Player 2 - random)
            if not is_game_over:
                opponent_actions = game.get_available_actions(game.player2)
                if opponent_actions:
                    # if episode == total_episodes // 2:
                    #     opponent_action = oponent_agent.choose_action(game.get_state, opponent_actions)
                    #     game.make_move(opponent_action, game.player2)
                    # else:    
                    opponent_action = oponent_agent.choose_action(game.get_state, opponent_actions)
                    game.make_move(opponent_action, game.player2)

                    if debug:
                        print("P2 action: ", opponent_action)
                        game.draw_board()

                    if game.check_win(20):
                        agent.update_q_table(state, action, -1, game.get_state())
                        is_game_over = True
                        
                        if debug:
                            print("game over Player2 win")
                    
                    # elif game.check_line_of_two(game.player2):
                        
        agent.decay_exploration_rate(episode, total_episodes)


def mirror_state(original_state) -> tuple:
    """
    Mirrors the game state for the perspective of the other player.

    Args:
        original_state (tuple): A tuple representing the game state.
        Example: ((-1,-1,-1,-1), ..., (2,1,0), (1,2,0), 10)
    Returns:
        tuple: A new tuple with the inverted state.
    """
    # Convert the tuple to a list for easier modification
    state_list = list(original_state)

    # Invert the 9 board positions (each is a 4-element tuple)
    mirrored_board = []
    for position in state_list[0:9]:
        mirrored_position = []
        for piece in position:
            if 11 <= piece <= 13:
                # Player 1's piece becomes player 2's piece
                mirrored_position.append(piece + 10)
            elif 21 <= piece <= 23:
                # Player 2's piece becomes player 1's piece
                mirrored_position.append(piece - 10)
            else:
                # Board bottom (0) or empty position (-1) remain the same
                mirrored_position.append(piece)
        mirrored_board.append(tuple(mirrored_position))

    # Invert the piece counters
    p1_counters = state_list[9]
    p2_counters = state_list[10]
    p3_counters = state_list[11]

    p4_counters = state_list[12]
    p5_counters = state_list[13]
    p6_counters = state_list[14]
    
    # Invert the turn indicator
    original_turn = state_list[15]
    mirrored_turn = 20 if original_turn == 10 else 10

    # Create the new tuple with the mirrored state
    mirrored_state = tuple(
        mirrored_board + 
        [p4_counters] + [p5_counters] + [p6_counters] + 
        [p1_counters] + [p2_counters] + [p3_counters] + 
        [mirrored_turn]
    )
    
    return mirrored_state

def mirro_q_table(original_q_table):
    new_q_table = {}

    for state in original_q_table.keys():
        new_state = mirror_state(state)
        new_actions_reward_dict = {}

        # for action, reward in original_q_table[s].items():
        #     new_action = (action[0], 20 + action[1] % 10, action[2]) if action[0] == 'p' else tuple(action)
        #     new_actions_reward_dict[new_action] = reward
            
        # MAYBE REWARD * -1
        new_actions_reward_dict = {(action[0], 20 + action[1] % 10, action[2]) if action[0] == 'p' else tuple(action) : reward * -1 for action, reward in original_q_table[state].items()}

        new_q_table[new_state] = new_actions_reward_dict

    return new_q_table

def play_against_human(agent):
    game = GlobbletGobblers()

    is_game_over = False
        
    while not is_game_over:
        state = game.get_state()
        available_actions = game.get_available_actions(game.player1)
        
        # Agent's turn (Player 1)
        agent.exploration_rate = 0
        action = agent.choose_action(state, available_actions)
        game.make_move(action, game.player1)

        game.draw_board()

        if game.check_win(10):
            print("AI win")
            is_game_over = True
            break

        # print(game.get_state())

        valid_action_p2 = False
        while not valid_action_p2:
            player_action = str(input('place <p, piece, targ_pos> \nmove <m, orig_pos, target_pos>: '))
            player_action = player_action.split()
            
            try:
                valid_action_p2 = game.make_move((player_action[0], int(player_action[1]), int(player_action[2])), game.player2)

                if game.check_win(20):
                    print("Human win")
                    is_game_over = True
                elif game.is_draw():
                    print("Draw")
                    is_game_over = True
                
                game.draw_board()
                # print(game.get_state())

            except Exception as e:
                print("Input error. Try again.\n")

def print_qtable_metrics(q_table):
    q_values = [reward for state, actions in q_table.items() for action, reward in actions.items()]
    print("")
    if q_values:
        mean_q = sum(q_values) / len(q_values)
        max_q = max(q_values)
        min_q = min(q_values)
        std_q = (sum((x - mean_q) ** 2 for x in q_values) / len(q_values)) ** 0.5
        print(f"Média dos valores Q: {mean_q}")
        print(f"Valor Q máximo: {max_q}")
        print(f"Valor Q mínimo: {min_q}")
        print(f"Desvio padrão dos valores Q: {std_q}")
    else:
        print("Q-table vazia.")

    num_states = len(q_table)
    print(f"Número de estados explorados: {num_states}")

    actions_per_state = [len(actions) for state, actions in q_table.items()]
    if actions_per_state:
        mean_actions = sum(actions_per_state) / len(actions_per_state)
        max_actions = max(actions_per_state)
        min_actions = min(actions_per_state)
        print(f"Média de ações por estado: {mean_actions}")
        print(f"Máximo de ações em um estado: {max_actions}")
        print(f"Mínimo de ações em um estado: {min_actions}")
    else:
        print("Q-table vazia.")

    non_zero_count = sum(1 for state, actions in q_table.items() for action, reward in actions.items() if reward != 0)
    print(f"Número de pares estado-ação com valores Q não nulos: {non_zero_count}")

    optimal_q_values = [max(actions.values(), default=0) for state, actions in q_table.items()]
    if optimal_q_values:
        mean_optimal_q = sum(optimal_q_values) / len(optimal_q_values)
        max_optimal_q = max(optimal_q_values)
        min_optimal_q = min(optimal_q_values)
        print(f"Média dos valores Q ótimos por estado: {mean_optimal_q}")
        print(f"Valor Q ótimo máximo: {max_optimal_q}")
        print(f"Valor Q ótimo mínimo: {min_optimal_q}")
    else:
        print("Q-table vazia.")

def train_debug(load=False):
    if load:
        loaded_q_table = load_model("q_table-R&AI.pkl")
        # agent = QLearningAgent(q_table=loaded_q_table)
        agent = QLearningAgent(learning_rate=0.0001, discount_factor=0.9, exploration_rate=0.5, q_table=loaded_q_table)
    else:
        agent = QLearningAgent(learning_rate=0.0001, discount_factor=0.9, exploration_rate=0.5)

    train(agent, total_episodes=2, debug=False)

    print(agent.q_table)
    # print_qtable_metrics(agent.q_table)

def train_play():
    agent = QLearningAgent(learning_rate=0.001, discount_factor=0.9, exploration_rate=0.9)
    
    # Treine o agente
    train(agent, total_episodes=1_000_000)

    # Salve o modelo após o treino
    save_model(agent)

    print("-------" * 7)
    print_qtable_metrics(agent.q_table)
    print("-------" * 7)

    play_against_human(agent)

def play_loaded():
    loaded_q_table = load_model("q_table-R1M.pkl")
    agent = QLearningAgent(learning_rate=0.0001, discount_factor=0.9, exploration_rate=0.5, q_table=loaded_q_table)
    play_against_human(agent)

# train_debug()
# train_play()
# play_loaded()