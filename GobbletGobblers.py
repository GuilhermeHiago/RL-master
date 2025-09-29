import random
import pickle

"""
    Piece 11 = Player1 piece size 1
    Piece 12 = Player1 piece size 2
    Piece 13 = Player1 piece size 3
"""
class GlobbletGobblers:
    def __init__(self):
        self.board = [[0] for _ in range(9)]# [[0]] * 9 gives 9 ref to same list
        self.player1 = 10
        self.player2 = 20

        self.player_pieces = {self.player1: [11, 11, 12, 12, 13, 13], 
                              self.player2: [21, 21, 22, 22, 23, 23]}
        pass

    def get_state(self) -> tuple:
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
        placement_pos = [i for i, cell in enumerate(self.board) if cell[-1] == 0]

        placement_moves = [("p", piece, pos) for piece in set(self.player_pieces[player]) for pos in placement_pos]

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

        return placement_moves + move_moves
    
    def make_move(self, action, player):
        if action in self.get_available_actions(player):
            # action <p, orig_pos, targ_pos>
            if action[0] == 'm':
                if self.board[action[1]][-1] >> 3 == player >> 3:
                    if self.board[action[1]][-1] % 10 > self.board[action[2]][-1] % 10:
                        piece = self.board[action[1]].pop()
                        self.board[action[2]].append(piece)
                        return True
            # action <p, piece, pos>
            elif action[0] == 'p':
                # if is piece from right player and it has this piece
                if action[1] >> 3 == player >> 3 and action[1] in self.player_pieces[player]:
                    if action[1] % 10 > self.board[action[2]][-1] % 10:
                        self.board[action[2]].append(action[1])
                        self.player_pieces[player].remove(action[1])
                        return True
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

    def decay_exploration_rate(self, episode, total_episodes):
        self.exploration_rate = self.min_exploration_rate + (1.0 - self.min_exploration_rate) * (0.01 ** (episode / total_episodes))

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

# --- Loop de Treinamento ---
def train(agent, total_episodes=20000):
    game = GlobbletGobblers()
    
    for episode in range(total_episodes):
        game.reset()
        is_game_over = False
        
        while not is_game_over:
            state = game.get_state()
            available_actions = game.get_available_actions(game.player1)
            
            # Agent's turn (Player 1)
            action = agent.choose_action(state, available_actions)
            game.make_move(action, game.player1)
            
            # game.draw_board()

            reward = 0
            if game.is_game_over():
                if game.check_win(1): reward = 1
                elif game.is_draw(): reward = 0.5
                is_game_over = True

                # game.draw_board()
                # print("game over 1: ", reward)
            
            agent.update_q_table(state, action, reward, game.get_state())

            # Opponent's turn (Player 2 - random)
            if not is_game_over:
                opponent_actions = game.get_available_actions(game.player2)
                if opponent_actions:
                    opponent_action = random.choice(opponent_actions)
                    game.make_move(opponent_action, game.player2)
                    
                    if game.check_win(2):
                        agent.update_q_table(state, action, -1, game.get_state())
                        is_game_over = True
                        # print("game over 2: ")

                    # game.draw_board()
            
        agent.decay_exploration_rate(episode, total_episodes)

def play_against_human(agent):
    game = GlobbletGobblers()

    is_game_over = False
        
    while not is_game_over:
        state = game.get_state()
        available_actions = game.get_available_actions(game.player1)
        
        # Agent's turn (Player 1)
        action = agent.choose_action(state, available_actions)
        game.make_move(action, game.player1)

        if game.check_win(10):
            print("AI win")
            is_game_over = True
            break

        game.draw_board()

        player_action = str(input('place <p, piece, targ_pos> \nmove <m, orig_pos, target_pos>: '))
        player_action = player_action.split()
        
        game.make_move((player_action[0], int(player_action[1]), int(player_action[2])), game.player2)

        if game.check_win(20):
            print("Human win")
            is_game_over = True

        if game.is_draw():
            print("Draw")
            is_game_over = True

# # Tente carregar o modelo existente. Se não houver, crie um novo.
# loaded_q_table = load_model()
# agent = QLearningAgent(q_table=loaded_q_table)

agent = QLearningAgent()

# Treine o agente
train(agent, total_episodes=10000)

# Salve o modelo após o treino
save_model(agent)

play_against_human(agent)

# game = GlobbletGobblers()
# print(game.get_available_actions(10))

# game.draw_board()
# print(game.is_game_over())

# print(game.player_pieces)
# game.make_move(("p", 4, 5), 10)
# game.draw_board()
# print(game.player_pieces)