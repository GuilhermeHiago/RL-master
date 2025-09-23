import random
import pickle

# --- Classes (Mesmas que no exemplo anterior) ---
class TicTacToe:
    def __init__(self):
        self.board = [0] * 9
        self.current_player = 1

    def get_state(self):
        return tuple(self.board)

    def is_game_over(self):
        if self.check_win(1) or self.check_win(2) or self.is_draw():
            return True
        return False

    def check_win(self, player):
        win_conditions = [(0, 1, 2), (3, 4, 5), (6, 7, 8),
                          (0, 3, 6), (1, 4, 7), (2, 5, 8),
                          (0, 4, 8), (2, 4, 6)]
        return any(all(self.board[i] == player for i in wc) for wc in win_conditions)

    def is_draw(self):
        return 0 not in self.board

    def get_available_actions(self):
        return [i for i, cell in enumerate(self.board) if cell == 0]

    def make_move(self, action, player):
        if self.board[action] == 0:
            self.board[action] = player
            return True
        return False

    def reset(self):
        self.board = [0] * 9
        self.current_player = 1

    def draw_board(self):
        """
            Desenha o tabuleiro do Jogo da Velha no console.
            'X' para o jogador 1, 'O' para o jogador 2, e ' ' para espaços vazios.
        """
        chars = [' ', 'X', 'O']
        
        print("\n")
        print(" " + chars[self.board[0]] + " | " + chars[self.board[1]] + " | " + chars[self.board[2]] + " ")
        print("---+---+---")
        print(" " + chars[self.board[3]] + " | " + chars[self.board[4]] + " | " + chars[self.board[5]] + " ")
        print("---+---+---")
        print(" " + chars[self.board[6]] + " | " + chars[self.board[7]] + " | " + chars[self.board[8]] + " ")
        print("\n")

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
        if state not in self.q_table:
            self.q_table[state] = {a: 0 for a in range(9)}

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
    game = TicTacToe()
    
    for episode in range(total_episodes):
        game.reset()
        is_game_over = False
        
        while not is_game_over:
            state = game.get_state()
            available_actions = game.get_available_actions()
            
            # Agent's turn (Player 1)
            action = agent.choose_action(state, available_actions)
            game.make_move(action, 1)
            
            reward = 0
            if game.is_game_over():
                if game.check_win(1): reward = 1
                elif game.is_draw(): reward = 0.5
                is_game_over = True
            
            agent.update_q_table(state, action, reward, game.get_state())

            # Opponent's turn (Player 2 - random)
            if not is_game_over:
                opponent_actions = game.get_available_actions()
                if opponent_actions:
                    opponent_action = random.choice(opponent_actions)
                    game.make_move(opponent_action, 2)
                    
                    if game.check_win(2):
                        agent.update_q_table(state, action, -1, game.get_state())
                        is_game_over = True
            
        agent.decay_exploration_rate(episode, total_episodes)

def play_against_human(agent):
    game = TicTacToe()

    is_game_over = False
        
    while not is_game_over:
        state = game.get_state()
        available_actions = game.get_available_actions()
        
        # Agent's turn (Player 1)
        action = agent.choose_action(state, available_actions)
        game.make_move(action, 1)

        if game.check_win(1):
            print("AI win")
            is_game_over = True

        game.draw_board()

        player_action = int(input('Number from 0 to 8: '))
        game.make_move(player_action, 2)

        if game.check_win(2):
            print("Human win")
            is_game_over = True

        if game.is_draw():
            print("Draw")
            is_game_over = True

# --- Execução Principal ---
if __name__ == "__main__":
    # Tente carregar o modelo existente. Se não houver, crie um novo.
    loaded_q_table = load_model()
    agent = QLearningAgent(q_table=loaded_q_table)
    
    # Treine o agente
    train(agent, total_episodes=10000)
    
    # Salve o modelo após o treino
    save_model(agent)

    play_against_human(agent)