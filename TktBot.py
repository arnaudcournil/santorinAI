import random
from santorinai.player import Player


# # Basic Rules


def movesPlayer(board, players, x, y):
    z = board[x][y]
    moves = []
    for i in range(-1, 2):
        for j in range(-1, 2):
            if i == 0 and j == 0:
                continue
            if x+i < 0 or x+i >= len(board):
                continue
            if y+j < 0 or y+j >= len(board):
                continue
            if board[x+i][y+j] in (z-2, z-1, z, z+1) and board[x+i][y+j] < 4 and (x+i, y+j) not in players:
                moves.append((x+i, y+j))
    return moves


def constructsPlayer(board, players, x, y):
    places = []
    for i in range(-1, 2):
        for j in range(-1, 2):
            if i == 0 and j == 0:
                continue
            if x+i < 0 or x+i >= len(board):
                continue
            if y+j < 0 or y+j >= len(board):
                continue
            if board[x+i][y+j] < 4 and (x+i, y+j) not in players:
                places.append((x+i, y+j))
    return places


def win(board, x, y):
    return board[x][y] == 3


# # MonteCarlo Simulation


def randomWin(board, players, playerAct):
    for i in range(4):
        if win(board, *players[i]):
            return i
    for i in range(100):
        playerAct = (playerAct + 1) % 4
        moves = movesPlayer(board, players, *players[playerAct])
        if len(moves) > 0:
            pos = moves[random.randint(0, len(moves)-1)]
            if win(board, *pos):
                return playerAct
            constructs = constructsPlayer(board, players, *pos)
            if len(constructs) > 0:
                players[playerAct] = pos
                construct = constructs[random.randint(0, len(constructs)-1)]
                board[construct[0]][construct[1]] += 1
    return -1


def simulate(nbGames, myboard, myplayers, myplayerAct, maximizingPlayer):
    nbWin = 0
    for i in range(nbGames):
        board = myboard[:]
        players = myplayers[:]
        rd = randomWin([row[:] for row in board], players[:], myplayerAct)
        if rd == -1:
            nbGames -= 1
        else:
            nbWin += 1 if rd in (myplayerAct, (myplayerAct+2) % 4) else 0
    if nbGames == 0:
        return 0
    return nbWin/nbGames * 2 - 1 if maximizingPlayer else 1 - nbWin/nbGames * 2


# # Minimax


# do a minimax search to find the best move

def minimax(board, players, playerAct, depth, alpha, beta, maximizingPlayer, n):
    if win(board, *players[(playerAct-1) % 4]):
        return -1 if maximizingPlayer else 1
    if depth == 0:
        return simulate(n, board, players, playerAct, maximizingPlayer)
    isBreak = False
    for move in movesPlayer(board, players, *players[playerAct]):
        if isBreak:
            break
        newPlayers = players[:]
        newPlayers[playerAct] = move
        for construct in constructsPlayer(board, newPlayers, *newPlayers[playerAct]):
            newBoard = [row[:] for row in board]
            newBoard[construct[0]][construct[1]] += 1
            value = minimax(newBoard, newPlayers, (playerAct+1) %
                            4, depth-1, alpha, beta, not maximizingPlayer, n)
            if maximizingPlayer:
                alpha = max(alpha, value)
                if alpha >= 1:
                    isBreak = True
                    break
            else:
                beta = min(beta, value)
                if beta <= -1:
                    isBreak = True
                    break
            if beta <= alpha:
                isBreak = True
                break
    return alpha if maximizingPlayer else beta


def getBestMove(board, players, playerAct, depth, n=100):
    bestValue = -1
    bestMove = None
    bestConstruct = None
    while bestMove == None:
        if depth == 0:
            return movesPlayer(board, players, *players[playerAct])[0], constructsPlayer(board, players, *players[playerAct])[0], 0, 0
        for move in movesPlayer(board, players, *players[playerAct]):
            newPlayers = players[:]
            newPlayers[playerAct] = move
            for construct in constructsPlayer(board, newPlayers, *newPlayers[playerAct]):
                newBoard = [row[:] for row in board]
                newBoard[construct[0]][construct[1]] += 1
                value = minimax(newBoard, newPlayers, (playerAct+1) %
                                4, depth-1, bestValue, 1, False, n)
                if value > bestValue:
                    bestValue = value
                    bestMove = move
                    bestConstruct = construct
                if bestValue >= 1:
                    return bestMove, bestConstruct, bestValue
        depth -= 1
    return bestMove, bestConstruct, bestValue


class TktBot(Player):
    """
    Minimax + Monte Carlo Tree Search bot
    """

    def name(self):
        return "Tkt bot"

    # Placement of the pawns
    def place_pawn(self, board, pawn):
        my_choice = (2, 2) if (2, 2) in board.get_possible_movement_positions(
            pawn) else random.choice(board.get_possible_movement_positions(pawn))
        return my_choice

    # Movement and building
    def play_move(self, board, pawn):
        pos, construct, i = getBestMove(
            board.board, [player.pos for player in board.pawns], pawn.number - 1, 2, 500)
        return pos, construct
