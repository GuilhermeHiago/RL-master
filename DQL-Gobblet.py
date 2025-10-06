import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# Importações necessárias do PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from GobbletGobblers import GlobbletGobblers, RandomAgent

# Defina o dispositivo para usar a GPU se disponível, senão CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQN_Network(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN_Network, self).__init__()
        # Definimos as camadas da rede no construtor
        self.layer1 = nn.Linear(state_size, 128)
        self.layer2 = nn.Linear(128, 128)
        # A camada de saída tem neurônios para cada ação possível
        self.layer3 = nn.Linear(128, action_size)

    def forward(self, x):
        # A função forward define como a entrada (x) passa pelas camadas
        # Usamos a função de ativação ReLU nas camadas ocultas
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        # A camada de saída não tem ativação (linear)
        return self.layer3(x)

import numpy as np
import random
from collections import deque

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)

        # Hiperparâmetros
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999
        self.learning_rate = 0.001

        # Criar a rede principal (policy_net) e a rede alvo (target_net)
        # .to(device) move a rede para a GPU se disponível
        self.policy_net = DQN_Network(state_size, action_size).to(device)
        self.target_net = DQN_Network(state_size, action_size).to(device)
        self.update_target_net()
        # A rede alvo não é treinada, então a colocamos em modo de avaliação
        self.target_net.eval()

        # O otimizador Adam irá ajustar os pesos da policy_net
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        # Usamos Mean Squared Error como função de perda
        self.criterion = nn.MSELoss()


    def update_target_net(self):
        # Copia os pesos da policy_net para a target_net
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def choose_action(self, state, legal_actions_indices):
        if not legal_actions_indices: # Checagem de segurança
            return None

        if np.random.rand() <= self.epsilon:
            return random.choice(legal_actions_indices)
        
        # Coloca a rede em modo de avaliação
        self.policy_net.eval()
        with torch.no_grad(): # Não precisamos calcular gradientes para a predição
            # Converte o estado (numpy) para um tensor PyTorch e o envia para o device
            state_tensor = torch.tensor(state, dtype=torch.float32).to(device)
            act_values = self.policy_net(state_tensor)
        
        # Converte de volta para numpy para a lógica de mascaramento
        act_values_np = act_values.cpu().numpy()[0]
        
        masked_act_values = np.full(self.action_size, -np.inf)
        masked_act_values[legal_actions_indices] = act_values_np[legal_actions_indices]
        
        return np.argmax(masked_act_values)

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)

        # --- INÍCIO DA CORREÇÃO ---
        # Filtra as transições onde a ação é inválida (-1)
        # A ação é o segundo elemento (índice 1) da tupla (state, action, reward, next_state, done)
        valid_transitions = [t for t in minibatch if t[1] != -1]
        
        # Se o lote filtrado estiver vazio, não há nada para treinar neste passo
        if len(valid_transitions) == 0:
            return
        # --- FIM DA CORREÇÃO ---

        # Descompacta o minibatch e converte para tensores PyTorch de uma vez (vetorização)
        # Usa o lote filtrado 'valid_transitions' em vez do 'minibatch' original
        states = torch.tensor(np.array([e[0] for e in valid_transitions]), dtype=torch.float32).squeeze(1).to(device)
        actions = torch.tensor([e[1] for e in valid_transitions], dtype=torch.int64).to(device)
        rewards = torch.tensor([e[2] for e in valid_transitions], dtype=torch.float32).to(device)
        next_states = torch.tensor(np.array([e[3] for e in valid_transitions]), dtype=torch.float32).squeeze(1).to(device)
        dones = torch.tensor([e[4] for e in valid_transitions], dtype=torch.float32).to(device)
        
        # 1. Calcula os Q-values para os estados atuais usando a policy_net
        # Pegamos apenas os Q-values das ações que foram realmente tomadas
        self.policy_net.train() # Coloca a rede em modo de treinamento
        q_values = self.policy_net(states)

        # Adicione estas linhas para depuração:
        # print(f"Shape dos Q-values: {q_values.shape}")
        # print(f"Ações (min/max): {actions.min().item()} / {actions.max().item()}")
        
        # Verifique se há alguma ação fora do intervalo válido
        num_actions = q_values.shape[1]
        if actions.max().item() >= num_actions or actions.min().item() < 0:
            print("ERRO: Ações fora do intervalo encontrado!")
            # Você pode querer inspecionar o tensor inteiro aqui:
            # print(actions)

        current_q_values = q_values.gather(1, actions.unsqueeze(1))

        # 2. Calcula os Q-values para os próximos estados usando a target_net
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
        
        # 3. Calcula o alvo (target) usando a Equação de Bellman
        # target = recompensa + gamma * max_q_do_proximo_estado
        # Se o jogo terminou (done=1), a parte do gamma é zerada
        target = rewards + self.gamma * next_q_values * (1 - dones)

        # 4. Calcula a perda (loss)
        loss = self.criterion(current_q_values, target.unsqueeze(1))

        # 5. Otimiza o modelo
        self.optimizer.zero_grad()  # Zera os gradientes
        loss.backward()             # Calcula os novos gradientes (backpropagation)
        self.optimizer.step()       # Atualiza os pesos da rede

        # Decaimento do Epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Inicialização (mesma de antes)

env = GlobbletGobblers() # Sua classe de jogo deve gerenciar os turnos
state_size = 9*4 + 7 # env.board.flatten().shape[0]
# state_size = env.get_state().shape[0]
action_size = 126 # env.board.size
agent = DQNAgent(state_size, action_size) # Nosso agente IA (Jogador 1)
batch_size = 64
EPISODES = 3000
UPDATE_TARGET_EVERY = 10

all_possible_actions = [
    ('p', 11, 0), ('p', 11, 1), ('p', 11, 2), ('p', 11, 3), ('p', 11, 4), ('p', 11, 5), ('p', 11, 6), ('p', 11, 7), ('p', 11, 8),
    ('p', 12, 0), ('p', 12, 1), ('p', 12, 2), ('p', 12, 3), ('p', 12, 4), ('p', 12, 5), ('p', 12, 6), ('p', 12, 7), ('p', 12, 8),
    ('p', 13, 0), ('p', 13, 1), ('p', 13, 2), ('p', 13, 3), ('p', 13, 4), ('p', 13, 5), ('p', 13, 6), ('p', 13, 7), ('p', 13, 8),
    ('p', 21, 0), ('p', 21, 1), ('p', 21, 2), ('p', 21, 3), ('p', 21, 4), ('p', 21, 5), ('p', 21, 6), ('p', 21, 7), ('p', 21, 8),
    ('p', 22, 0), ('p', 22, 1), ('p', 22, 2), ('p', 22, 3), ('p', 22, 4), ('p', 22, 5), ('p', 22, 6), ('p', 22, 7), ('p', 22, 8),
    ('p', 23, 0), ('p', 23, 1), ('p', 23, 2), ('p', 23, 3), ('p', 23, 4), ('p', 23, 5), ('p', 23, 6), ('p', 23, 7), ('p', 23, 8),
    ('m', 0, 1), ('m', 0, 2), ('m', 0, 3), ('m', 0, 4), ('m', 0, 5), ('m', 0, 6), ('m', 0, 7), ('m', 0, 8),
    ('m', 1, 0), ('m', 1, 2), ('m', 1, 3), ('m', 1, 4), ('m', 1, 5), ('m', 1, 6), ('m', 1, 7), ('m', 1, 8),
    ('m', 2, 0), ('m', 2, 1), ('m', 2, 3), ('m', 2, 4), ('m', 2, 5), ('m', 2, 6), ('m', 2, 7), ('m', 2, 8),
    ('m', 3, 0), ('m', 3, 1), ('m', 3, 2), ('m', 3, 4), ('m', 3, 5), ('m', 3, 6), ('m', 3, 7), ('m', 3, 8),
    ('m', 4, 0), ('m', 4, 1), ('m', 4, 2), ('m', 4, 3), ('m', 4, 5), ('m', 4, 6), ('m', 4, 7), ('m', 4, 8),
    ('m', 5, 0), ('m', 5, 1), ('m', 5, 2), ('m', 5, 3), ('m', 5, 4), ('m', 5, 6), ('m', 5, 7), ('m', 5, 8),
    ('m', 6, 0), ('m', 6, 1), ('m', 6, 2), ('m', 6, 3), ('m', 6, 4), ('m', 6, 5), ('m', 6, 7), ('m', 6, 8),
    ('m', 7, 0), ('m', 7, 1), ('m', 7, 2), ('m', 7, 3), ('m', 7, 4), ('m', 7, 5), ('m', 7, 6), ('m', 7, 8),
    ('m', 8, 0), ('m', 8, 1), ('m', 8, 2), ('m', 8, 3), ('m', 8, 4), ('m', 8, 5), ('m', 8, 6), ('m', 8, 7)
]

action_to_int = {action: i for i, action in enumerate(all_possible_actions)}
int_to_action = {i: action for i, action in enumerate(all_possible_actions)}

ACTION_SIZE_GLOBAL = len(all_possible_actions)

for e in range(EPISODES):
    env.reset() # O ambiente é reiniciado
    state = env.get_flat_state()
    state = np.reshape(state, [1, state_size])
    done = False
    
    while not done:
        # --- TURNO DO AGENTE (JOGADOR 1) ---
        # Certifique-se de que é a vez do Jogador 1 (sua classe env deve controlar isso)
        # if env.current_player == 1:
        legal_actions_tuples = env.get_available_actions(env.player1)

        # Garante que a ação exista no nosso dicionário (segurança)
        legal_actions_indices = [action_to_int[action] for action in legal_actions_tuples if action in action_to_int]

        if not legal_actions_indices:
            # Se não há ações legais para o agente, é um empate ou fim de jogo
            done = True
            reward = 0 # Recompensa neutra
            agent.remember(state, -1, reward, state, done) # Ação -1 para indicar que não houve ação
            break

        chosen_action_index = agent.choose_action(state, legal_actions_indices)
        action_tuple_to_execute = int_to_action[chosen_action_index]
        env.make_move(action_tuple_to_execute, env.player1)

        # env.draw_board()
        # print()

        # Verifica se o Agente venceu com esta jogada
        if env.check_win(env.player1):
            reward = 1  # Recompensa alta por vencer
            done = True
            next_state = env.get_flat_state() # Pega o estado final
            next_state = np.reshape(next_state, [1, state_size])
            
            # Armazena a última e vitoriosa experiência
            agent.remember(state, chosen_action_index, reward, next_state, done)
            break # Sai do loop while

        # --- TURNO DO OPONENTE (JOGADOR 2) ---
        # Oponente joga aleatoriamente
        
        legal_actions_opponent_tuples = env.get_available_actions(env.player2)
        # Se não há jogadas legais (empate), o jogo acaba
        if not legal_actions_opponent_tuples:
            reward = 0 # Recompensa neutra por empate
            done = True
            next_state = env.get_flat_state()
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, chosen_action_index, reward, next_state, done)
            break

        opponent_action_tuple = random.choice(legal_actions_opponent_tuples)
        env.make_move(opponent_action_tuple, env.player2) # Ambiente executa a jogada do oponente

        # Agora obtemos o "next_state" real e a recompensa final do turno
        next_state = env.get_flat_state()
        next_state = np.reshape(next_state, [1, state_size])

        # Verifica o resultado APÓS a jogada do oponente
        if env.check_win(env.player2):
            reward = -1 # Punição alta por perder
            done = True
        elif env.is_draw():
            reward = 0.5 # Empate
            done = True
        else:
            reward = 0 # Recompensa 0 para jogadas que não terminam o jogo
            done = False

        # 3. Agente armazena a experiência completa do seu turno
        agent.remember(state, chosen_action_index, reward, next_state, done)

        # 4. Atualiza o estado para o próximo turno do Agente
        state = next_state

    # Fim do episódio (while not done)
    print(f"Episódio: {e+1}/{EPISODES}, Epsilon: {agent.epsilon:.2f}")

    # Treina o agente e atualiza a rede alvo (fora do loop do jogo)
    agent.replay(batch_size)
    if e % UPDATE_TARGET_EVERY == 0:
        agent.update_target_net()


def play_against_human(agent):
    game = env
    game.reset()

    is_game_over = False
        
    while not is_game_over:
        state = game.get_flat_state()
        state = np.reshape(state, [1, state_size])
        available_actions = game.get_available_actions(game.player1)
        
        # Agent's turn (Player 1)
        # --- TURNO DO AGENTE (JOGADOR 1) ---
        # Certifique-se de que é a vez do Jogador 1 (sua classe env deve controlar isso)
        # if env.current_player == 1:
        legal_actions_tuples = game.get_available_actions(game.player1)

        # Garante que a ação exista no nosso dicionário (segurança)
        legal_actions_indices = [action_to_int[action] for action in legal_actions_tuples if action in action_to_int]

        if not legal_actions_indices:
            # Se não há ações legais para o agente, é um empate ou fim de jogo
            is_game_over = True
            print("invalid action of Agent")
            break
        else:
            chosen_action_index = agent.choose_action(state, legal_actions_indices)
            action_tuple_to_execute = int_to_action[chosen_action_index]
            game.make_move(action_tuple_to_execute, env.player1)

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

play_against_human(agent)