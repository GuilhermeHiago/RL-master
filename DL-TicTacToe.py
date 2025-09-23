import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os

# Game Constants
# Defines numerical values for each "entity" on the board,
# making the code more readable and easier to manage.
AI_PLAYER = 1
OPPONENT_PLAYER = 2
EMPTY = 0
AI_PLAYER_SYMBOL = 'X'
HUMAN_PLAYER_SYMBOL = 'O'

# Training Hyperparameters
GAMMA = 0.99  # Discount factor for future rewards. A high value (close to 1)
              # makes the AI value long-term rewards more.
EPSILON_START = 1.0  # Initial probability of choosing a random action (exploration).
EPSILON_END = 0.01   # Minimum probability of exploration.
EPSILON_DECAY = 0.9995  # Epsilon decay rate. Decreases exploration over time.
LEARNING_RATE = 0.001  # Learning rate for the Adam optimizer.
EPISODES = 100000  # Total number of games for training.
MODEL_PATH = 'dqn_tic_tac_toe_model.pth' # File name to save the model.

# 1. Model: The Neural Network
# Defines the neural network's architecture. It's a simple network with 3 linear layers.
# Input: a 9-position vector (representing the board).
# Output: a 9-value Q-vector, one for each possible move.
class QNetwork(nn.Module):
    def __init__(self):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(9, 64)
        self.fc2 = nn.Linear(64, 64)
        self.output = nn.Linear(64, 9)

    def forward(self, x):
        # The 'relu' (Rectified Linear Unit) activation function introduces non-linearity,
        # allowing the network to learn more complex relationships.
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.output(x)

# 2. Training Agent
# Class that manages the training process and the AI's decision-making.
class DQNAgent:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network = QNetwork().to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=LEARNING_RATE)
        self.epsilon = EPSILON_START
        # A second network (opponent_network) is used to simulate a smarter opponent
        # after half of the training, so the AI learns to play against a stronger adversary.
        self.opponent_network = QNetwork().to(self.device)

    def choose_action(self, state_np, mode='train'):
        state = torch.from_numpy(state_np).float().to(self.device).unsqueeze(0)
        
        # Epsilon-Greedy Strategy:
        # In training mode, the AI explores (plays randomly) with probability epsilon,
        # and exploits (plays the best action) with probability 1-epsilon.
        if mode == 'train' and random.random() < self.epsilon:
            valid_actions = [i for i, pos in enumerate(state_np) if pos == EMPTY]
            if not valid_actions:
                return -1
            return random.choice(valid_actions)
        else:
            # Exploitation: the AI chooses the action with the highest Q-value.
            with torch.no_grad():
                q_values = self.q_network(state)
                # Sets the Q-values for occupied positions to negative infinity to
                # ensure the AI never chooses an invalid move.
                q_values[0, state_np != EMPTY] = -float('inf')
                return q_values.argmax().item()
    
    def choose_opponent_action(self, state_np):
        # The opponent's action logic, using its own Q-network to make decisions.
        state = torch.from_numpy(state_np).float().to(self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.opponent_network(state)
            q_values[0, state_np != EMPTY] = -float('inf')
            return q_values.argmax().item()

    def train(self, state, action, reward, next_state, done):
        # Converts game data to PyTorch tensors so they can be processed.
        state = torch.from_numpy(state).float().to(self.device).unsqueeze(0)
        next_state = torch.from_numpy(next_state).float().to(self.device).unsqueeze(0)
        action = torch.tensor([action], dtype=torch.long, device=self.device)
        reward = torch.tensor([reward], dtype=torch.float, device=self.device)
        done = torch.tensor([done], dtype=torch.bool, device=self.device)

        # The predicted Q-value is the value of the action the AI actually took in the current state.
        q_predicted = self.q_network(state).gather(1, action.unsqueeze(1))
        
        with torch.no_grad():
            # The target Q-value is the immediate reward plus the maximum Q-value
            # of the next state, discounted.
            q_next_state = self.q_network(next_state).max(1)[0]
        
        q_target = reward + (GAMMA * q_next_state) * (~done)

        # Calculates the loss using Mean Squared Error (MSE).
        # The loss represents the difference between the predicted Q-value and the target Q-value.
        loss = nn.functional.mse_loss(q_predicted, q_target.unsqueeze(1))

        # Backpropagation and network weight update.
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def save_model(self, path):
        # Saves the neural network's weights to a file.
        torch.save(self.q_network.state_dict(), path)
        print(f"Model saved to {path}")

    def load_model(self, path):
        # Loads the network's weights from a file for later use.
        if os.path.exists(path):
            self.q_network.load_state_dict(torch.load(path))
            # Sets the network to evaluation mode (not training).
            self.q_network.eval()
            print(f"Model loaded from {path}")
            return True
        return False
    
    def update_opponent_model(self):
        # Syncs the opponent network's weights with the main AI network,
        # so the opponent becomes as good as the AI at that moment.
        self.opponent_network.load_state_dict(self.q_network.state_dict())
        self.opponent_network.eval()


# Game environment functions
def create_board():
    return np.zeros(9, dtype=np.int32)

def check_win(board, player):
    # Winning combinations: 8 combinations of 3 positions.
    win_combinations = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8],
        [0, 3, 6], [1, 4, 7], [2, 5, 8],
        [0, 4, 8], [2, 4, 6]
    ]
    for combo in win_combinations:
        if all(board[i] == player for i in combo):
            return True
    return False

def print_board(board):
    # Function to print the board visually.
    symbols = {EMPTY: ' ', AI_PLAYER: AI_PLAYER_SYMBOL, OPPONENT_PLAYER: HUMAN_PLAYER_SYMBOL}
    print("-" * 13)
    for i in range(3):
        print(f"| {symbols[board[i*3]]} | {symbols[board[i*3+1]]} | {symbols[board[i*3+2]]} |")
        print("-" * 13)

def play_episode(agent, episode_counter):
    board = create_board()
    done = False
    
    # Alternates opponent type: 50% of games with a random opponent, 50% with a trained one.
    is_opponent_random = episode_counter < EPISODES / 2
    
    while not done:
        # AI's turn
        ai_state = np.copy(board)
        ai_action = agent.choose_action(ai_state, mode='train')
        
        if ai_action == -1:
            agent.train(ai_state, 0, 0.5, ai_state, True)
            return "Draw"

        board[ai_action] = AI_PLAYER
        done = check_win(board, AI_PLAYER) or np.all(board != EMPTY)
        
        reward = 0
        if done:
            if check_win(board, AI_PLAYER):
                reward = 10
            elif np.all(board != EMPTY):
                reward = 1
            agent.train(ai_state, ai_action, reward, board, True)
            return "Win"
        
        # Opponent's turn
        valid_opponent_actions = [i for i, pos in enumerate(board) if pos == EMPTY]
        if valid_opponent_actions:
            if is_opponent_random:
                opponent_action = random.choice(valid_opponent_actions)
            else:
                # The opponent uses its own network to choose the best move.
                opponent_action = agent.choose_opponent_action(board)
                if board[opponent_action] != EMPTY:
                    # In case the opponent's network chooses an invalid move (rare but possible),
                    # it falls back to a random move.
                    opponent_action = random.choice(valid_opponent_actions)
            
            board[opponent_action] = OPPONENT_PLAYER
            
            done = check_win(board, OPPONENT_PLAYER) or np.all(board != EMPTY)
            if done:
                reward = -10
                agent.train(ai_state, ai_action, reward, board, True)
                return "Loss"
        
        agent.train(ai_state, ai_action, reward, board, False)
    
    return "Draw"

def play_against_ai(agent):
    print("\n--- Interactive Game Mode ---")
    print(f"You are player '{HUMAN_PLAYER_SYMBOL}' and the AI is player '{AI_PLAYER_SYMBOL}'.")
    board = create_board()
    print_board(board)
    
    while True:
        # Human's turn
        try:
            human_move = int(input("Your turn. Choose a position from 0 to 8: "))
            if human_move not in range(9) or board[human_move] != EMPTY:
                print("Invalid position. Try again.")
                continue
        except ValueError:
            print("Invalid input. Please enter a number from 0 to 8.")
            continue

        board[human_move] = OPPONENT_PLAYER
        print_board(board)

        if check_win(board, OPPONENT_PLAYER):
            print("Congratulations! You won!")
            break
        if np.all(board != EMPTY):
            print("The game is a draw!")
            break

        # AI's turn
        print("\nAI's turn...")
        ai_action = agent.choose_action(board, mode='play')
        board[ai_action] = AI_PLAYER
        print_board(board)
        
        if check_win(board, AI_PLAYER):
            print("The AI won!")
            break
        if np.all(board != EMPTY):
            print("The game is a draw!")
            break

if __name__ == '__main__':
    agent = DQNAgent()

    # Option 1: Train and Save
    print("Starting model training...")
    ai_wins = 0
    ai_losses = 0
    draws = 0
    
    for i in range(EPISODES):
        # Every halfway through training, the opponent's AI is updated to match the main AI.
        # This creates an increasingly strong opponent, forcing the main AI to improve.
        if i == EPISODES / 2:
            print("\nSwitching opponent type to a trained model.")
            agent.update_opponent_model()
        elif i > EPISODES / 2 and (i - EPISODES/2) % 100 == 0:
            agent.update_opponent_model()
        
        result = play_episode(agent, i)
        if result == "Win":
            ai_wins += 1
        elif result == "Loss":
            ai_losses += 1
        else:
            draws += 1
        
        agent.epsilon = max(EPSILON_END, agent.epsilon * EPSILON_DECAY)
        
        if (i + 1) % 1000 == 0:
            print(f"Episode: {i+1}/{EPISODES} - Wins: {ai_wins}, Losses: {ai_losses}, Draws: {draws}, Epsilon: {agent.epsilon:.4f}")
            ai_wins, ai_losses, draws = 0, 0, 0
    
    print("\nTraining completed!")
    agent.save_model(MODEL_PATH)

    # Option 2: Play Against the Trained Model
    if agent.load_model(MODEL_PATH):
        play_against_ai(agent)
    else:
        print("Error: Could not load the model. Interactive game cannot be started.")
