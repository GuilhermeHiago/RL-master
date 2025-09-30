import random
import pickle

# Classe para representar o ambiente do jogo
class GobbletGobblersEnv:
    def __init__(self):
        # 0: Vazio, 1: Jogador 1 (Branco), 2: Jogador 2 (Laranja)
        # Tabuleiro 3x3, cada célula com uma lista de pilhas de peças
        self.board = [[[] for _ in range(3)] for _ in range(3)]
        self.players_pieces = {1: [3, 3, 3], 2: [3, 3, 3]} # [pequena, media, grande]
        self.current_player = 1
        self.winner = None

    def reset(self):
        self.board = [[[] for _ in range(3)] for _ in range(3)]
        self.players_pieces = {1: [3, 3, 3], 2: [3, 3, 3]}
        self.current_player = 1
        self.winner = None
        return self._get_state_key()

    def _get_state_key(self):
        # Cria uma representação do estado hashable para a Q-table
        # O estado inclui o tabuleiro e as peças de cada jogador
        board_tuple = tuple(tuple(tuple(cell) for cell in row) for row in self.board)
        pieces_tuple = (tuple(self.players_pieces[1]), tuple(self.players_pieces[2]))
        return (board_tuple, pieces_tuple, self.current_player)

    def _is_valid_move(self, action):
        action_type, *details = action
        
        if action_type == 'place':
            piece_size, row, col = details
            if not (0 <= row < 3 and 0 <= col < 3):
                return False
            # Verifica se a peça está na reserva
            if self.players_pieces[self.current_player][piece_size - 1] == 0:
                return False
            # Verifica se a peça pode "engolir" a peça no tabuleiro
            if self.board[row][col]:
                top_piece = self.board[row][col][-1]
                if top_piece[0] == self.current_player:
                    # Não pode engolir uma peça sua
                    return False
                if piece_size <= top_piece[1]:
                    # Não pode engolir uma peça maior ou do mesmo tamanho
                    return False
            return True
        
        elif action_type == 'move':
            from_row, from_col, to_row, to_col = details
            if not (0 <= from_row < 3 and 0 <= from_col < 3 and 0 <= to_row < 3 and 0 <= to_col < 3):
                return False
            # Verifica se a posição de origem tem uma peça sua
            if not self.board[from_row][from_col] or self.board[from_row][from_col][-1][0] != self.current_player:
                return False
            # Verifica se a peça pode "engolir" a peça no destino
            top_piece_from = self.board[from_row][from_col][-1]
            if self.board[to_row][to_col]:
                top_piece_to = self.board[to_row][to_col][-1]
                if top_piece_to[0] == self.current_player:
                    return False
                if top_piece_from[1] <= top_piece_to[1]:
                    return False
            return True
        return False

    def get_valid_actions(self):
        actions = []
        # Ações de colocar peça da reserva
        for piece_size in range(1, 4):
            for r in range(3):
                for c in range(3):
                    action = ('place', piece_size, r, c)
                    if self._is_valid_move(action):
                        actions.append(action)
        # Ações de mover peça no tabuleiro
        for r_from in range(3):
            for c_from in range(3):
                if self.board[r_from][c_from]:
                    for r_to in range(3):
                        for c_to in range(3):
                            action = ('move', r_from, c_from, r_to, c_to)
                            if self._is_valid_move(action):
                                actions.append(action)
        return actions

    def step(self, action):
        self._apply_action(action)
        self.winner = self._check_winner()
        reward = self._get_reward()
        done = self.winner is not None or self._is_draw()
        
        self.current_player = 1 if self.current_player == 2 else 2
        return self._get_state_key(), reward, done

    def _apply_action(self, action):
        action_type, *details = action
        
        if action_type == 'place':
            piece_size, row, col = details
            self.players_pieces[self.current_player][piece_size - 1] -= 1
            if self.board[row][col]:
                # Peça 'engolida' é colocada na pilha
                self.board[row][col].append((self.current_player, piece_size))
            else:
                self.board[row][col] = [(self.current_player, piece_size)]
        
        elif action_type == 'move':
            from_row, from_col, to_row, to_col = details
            piece_to_move = self.board[from_row][from_col].pop()
            if not self.board[to_row][to_col]:
                self.board[to_row][to_col] = [piece_to_move]
            else:
                self.board[to_row][to_col].append(piece_to_move)

    def _check_winner(self):
        player_to_check = 1 if self.current_player == 2 else 2
        
        # Checar linhas, colunas e diagonais para 3 peças seguidas
        lines_to_check = []
        # Linhas
        lines_to_check.extend([[(r, c) for c in range(3)] for r in range(3)])
        # Colunas
        lines_to_check.extend([[(r, c) for r in range(3)] for c in range(3)])
        # Diagonais
        lines_to_check.append([(0, 0), (1, 1), (2, 2)])
        lines_to_check.append([(0, 2), (1, 1), (2, 0)])

        for line in lines_to_check:
            first_piece_owner = None
            if self.board[line[0][0]][line[0][1]]:
                first_piece_owner = self.board[line[0][0]][line[0][1]][-1][0]
            
            if first_piece_owner and all(
                self.board[r][c] and self.board[r][c][-1][0] == first_piece_owner
                for r, c in line
            ):
                return first_piece_owner
        return None

    def _is_draw(self):
        return not self.get_valid_actions()

    def _get_reward(self):
        if self.winner == 1:
            return 1  # Recompensa pela vitória
        elif self.winner == 2:
            return -1  # Recompensa pela derrota
        return 0 # Recompensa neutra

    def render(self):
        # Função para visualizar o tabuleiro
        piece_map = {
            1: {1: 'Wp', 2: 'Wm', 3: 'Wg'},
            2: {1: 'Op', 2: 'Om', 3: 'Og'},
        }
        print("  0   1   2")
        for r in range(3):
            row_str = f"{r} "
            for c in range(3):
                cell_content = self.board[r][c]
                if cell_content:
                    player, size = cell_content[-1]
                    row_str += f"|{piece_map[player][size]}| "
                else:
                    row_str += "|  | "
            print(row_str)
        print(f"Peças J1: {self.players_pieces[1]}")
        print(f"Peças J2: {self.players_pieces[2]}")

# Classe do agente de Q-Learning
class QLearningAgent:
    def __init__(self, learning_rate=0.1, discount_factor=0.9, epsilon=0.9, epsilon_decay=0.9999, epsilon_min=0.1):
        self.q_table = {}
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.current_player = 1

    def choose_action(self, state_key, valid_actions):
        if not valid_actions:
            return None
        
        # Epsilon-greedy para explorar ou explorar
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(valid_actions)
        else:
            if state_key not in self.q_table:
                self.q_table[state_key] = {action: 0 for action in valid_actions}
            
            q_values = self.q_table[state_key]
            max_q = -float('inf')
            best_actions = []
            
            for action in valid_actions:
                if q_values.get(action, 0) > max_q:
                    max_q = q_values.get(action, 0)
                    best_actions = [action]
                elif q_values.get(action, 0) == max_q:
                    best_actions.append(action)
            return random.choice(best_actions)

    def learn(self, state_key, action, reward, next_state_key, next_valid_actions):
        if state_key not in self.q_table:
            self.q_table[state_key] = {}
        if action not in self.q_table[state_key]:
            self.q_table[state_key][action] = 0

        # Encontra o valor Q máximo para o próximo estado
        next_max_q = 0
        if next_state_key in self.q_table and next_valid_actions:
            next_q_values = self.q_table[next_state_key]
            next_max_q = max(next_q_values.get(act, 0) for act in next_valid_actions)
        
        # Aplica a equação de Bellman para atualizar a Q-table
        current_q = self.q_table[state_key][action]
        new_q = current_q + self.lr * (reward + self.gamma * next_max_q - current_q)
        self.q_table[state_key][action] = new_q
        
        # Decaimento do Epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_q_table(self, filename="q_table.pkl"):
        with open(filename, 'wb') as f:
            pickle.dump(self.q_table, f)
        print(f"Q-table salva em '{filename}'.")

    def load_q_table(self, filename="q_table.pkl"):
        try:
            with open(filename, 'rb') as f:
                self.q_table = pickle.load(f)
            print(f"Q-table carregada de '{filename}'.")
        except FileNotFoundError:
            print(f"Arquivo '{filename}' não encontrado. Iniciando com Q-table vazia.")

# --- Funções de Treinamento e Jogo ---
def train_agent(agent, num_episodes):
    env = GobbletGobblersEnv()
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        while not done:
            # Jogada do Agente (Jogador 1)
            valid_actions = env.get_valid_actions()
            action = agent.choose_action(state, valid_actions)
            if not action:
                break
            
            next_state, reward, done = env.step(action)
            agent.learn(state, action, reward, next_state, env.get_valid_actions())
            state = next_state
            
            if done:
                break
            
            # Jogada do Oponente Aleatório (Jogador 2)
            valid_actions_p2 = env.get_valid_actions()
            if valid_actions_p2:
                action_p2 = random.choice(valid_actions_p2)
                env.step(action_p2)

        if (episode + 1) % 10000 == 0:
            print(f"Episódio {episode + 1}/{num_episodes} completado. Tamanho da Q-table: {len(agent.q_table)}")

def play_against_agent(agent):
    env = GobbletGobblersEnv()
    state = env.reset()
    done = False
    
    print("--- Bem-vindo ao Gobblet Gobblers! ---")
    print("Você (Jogador 2, Laranja) vai jogar contra a IA (Jogador 1, Branco).")
    print("Para colocar uma peça: 'place <tamanho> <linha> <coluna>' (ex: place 1 0 0)")
    print("Tamanhos: 1=pequena, 2=media, 3=grande.")
    print("Para mover uma peça: 'move <linh_o> <col_o> <linh_d> <col_d>' (ex: move 0 0 0 1)")
    
    while not done:
        # Jogada da IA (Jogador 1)
        if env.current_player == 1:
            env.render()
            valid_actions = env.get_valid_actions()
            action_ia = agent.choose_action(state, valid_actions)
            if not action_ia:
                print("A IA não tem jogadas válidas. Fim de jogo.")
                break
            
            print(f"IA jogou: {action_ia}")
            state, reward, done = env.step(action_ia)

        # Jogada do Humano (Jogador 2)
        else:
            env.render()
            valid_actions_p2 = env.get_valid_actions()
            human_action = None
            while human_action not in valid_actions_p2:
                try:
                    move_str = input("Sua vez. Digite sua jogada: ")
                    parts = move_str.strip().split()
                    if parts[0] == 'place' and len(parts) == 4:
                        human_action = ('place', int(parts[1]), int(parts[2]), int(parts[3]))
                    elif parts[0] == 'move' and len(parts) == 5:
                        human_action = ('move', int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4]))
                    else:
                        raise ValueError
                except (ValueError, IndexError):
                    print("Formato de jogada inválido. Tente novamente.")
                    continue
                if human_action not in valid_actions_p2:
                    print("Essa jogada não é válida. Tente novamente.")
            
            state, reward, done = env.step(human_action)
            print(env._get_state_key())
    
    env.render()
    if env.winner == 1:
        print("A IA venceu! :(")
    elif env.winner == 2:
        print("Parabéns, você venceu!")
    else:
        print("Empate!")

# --- Execução Principal ---
if __name__ == "__main__":
    # Inicializa o agente e o ambiente
    agent = QLearningAgent()
    
    # Treinamento
    print("Iniciando treinamento da IA. Isso pode levar alguns minutos...")
    # Você pode ajustar o número de episódios para mais ou menos
    train_agent(agent, num_episodes=100)
    agent.save_q_table()
    
    # Jogar contra o modelo treinado
    play_against_agent(agent)