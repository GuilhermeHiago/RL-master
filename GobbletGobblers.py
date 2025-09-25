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
        
        self.board[1].append(13)
        self.board[0].append(12)
        self.board[4].append(11)
        self.board[8].append(13)
        pass

    def get_state(self):
        return tuple(self.board)
    
    def is_game_over(self):
        if self.check_win(10) or self.check_win(20): # or self.is_draw()
            return True
        return False
    
    def check_win(self, player):
        """ player == 10 or player == 20 """
        win_conditions = [(0, 1, 2), (3, 4, 5), (6, 7, 8),
                          (0, 3, 6), (1, 4, 7), (2, 5, 8),
                          (0, 4, 8), (2, 4, 6)]
        
        return any(all(self.board[i][-1] != 0 and self.board[i][-1] < player + 5 for i in wc) for wc in win_conditions)
    
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
                print("found:", self.board[p])
                # grid pos from list pos
                row, col = p // 3, p % 3

                print("r,c: ", row, col)

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

        # action <p, orig_pos, targ_pos>
        if action[0] == 'm':
           if self.board[action[1]][-1] >> 3 == player >> 3:
               if self.board[action[1]][-1] % 10 > self.board[action[2]][-1] % 10:
                   piece = self.board[action[1]].pop()
                   self.board[action[2]].append(piece)
                   return True
        # action <p, piece, pos>
        elif action[0] == 'p':
            if action[1] >> 3 == player >> 3:
                if action[1] % 10 > self.board[action[2]][-1] % 10:
                    self.board[action[2]].append(action[1])
                    return True
        return False

    def reset(self):
        self.board = [[0]] * 9
        self.player1_pieces = [11, 11, 12, 12, 13, 13]
        self.player2_pieces = [21, 21, 22, 22, 23, 23]

game = GlobbletGobblers()
print(game.get_available_actions(10))
print(game.is_game_over())

print(game.board)
game.make_move(("p", 11, 5), 10)
print(game.board)