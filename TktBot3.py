from numba import njit
import random
from santorinai.player import Player
import time
import numpy as np
from numba.typed import List

# Basic Rules


@njit
def movesPlayer(board, players, x, y):
    z = board[x, y]
    moves = []
    for i in range(-1, 2):
        for j in range(-1, 2):
            if i == 0 and j == 0:
                continue
            if x+i < 0 or x+i >= board.shape[0]:
                continue
            if y+j < 0 or y+j >= board.shape[1]:
                continue
            if board[x+i, y+j] - z < 2 and board[x+i, y+j] < 4 and (x+i, y+j) not in players:
                moves.append((x+i, y+j))
    return moves


@njit
def constructsPlayer(board, players, x, y):
    places = []
    for i in range(-1, 2):
        for j in range(-1, 2):
            if i == 0 and j == 0:
                continue
            if x+i < 0 or x+i >= board.shape[0]:
                continue
            if y+j < 0 or y+j >= board.shape[1]:
                continue
            if board[x+i, y+j] < 4 and (x+i, y+j) not in players:
                places.append((x+i, y+j))
    return places


@njit
def win(board, x, y):
    return board[x, y] == 3

# Minimax


@njit
def minimax(board, players, playerAct, depth, alpha, beta, maximizingPlayer):
    existAction = False
    if win(board, *players[(playerAct-1) % 4]):
        return -depth - 1 if maximizingPlayer else depth + 1
    if depth == 0:
        return 0
    for move in movesPlayer(board, players, *players[playerAct]):
        newPlayers = players.copy()
        newPlayers[playerAct] = move
        for construct in constructsPlayer(board, newPlayers, *newPlayers[playerAct]):
            if not existAction:
                existAction = True
            newBoard = np.copy(board)
            newBoard[construct[0], construct[1]] += 1
            value = minimax(newBoard, newPlayers, (playerAct+1) %
                            4, depth-1, alpha, beta, not maximizingPlayer)
            if maximizingPlayer:
                alpha = max(alpha, value)
                if alpha >= beta:
                    return alpha
            else:
                beta = min(beta, value)
                if beta <= alpha:
                    return beta

    if not existAction:  # Pawn is blocked
        return minimax(board, players, (playerAct+1) % 4, depth-1, alpha, beta, not maximizingPlayer)

    return alpha if maximizingPlayer else beta


@njit
def getBestMove(board, players, playerAct, depth):
    bestValue = -depth - 1
    bestMove = None
    bestConstruct = None
    while bestMove is None:
        for move in movesPlayer(board, players, *players[playerAct]):
            newPlayers = players.copy()
            newPlayers[playerAct] = move
            for construct in constructsPlayer(board, newPlayers, *newPlayers[playerAct]):
                if depth == 0:
                    return move, construct
                newBoard = np.copy(board)
                newBoard[construct[0], construct[1]] += 1
                value = minimax(newBoard, newPlayers, (playerAct+1) %
                                4, depth-1, bestValue, 1, False)
                if value is not None and value > bestValue:
                    bestValue = value
                    bestMove = move
                    bestConstruct = construct
                    if bestValue >= 1:
                        return bestMove, bestConstruct
        depth -= 1
    return bestMove, bestConstruct


def playProgressive(board, players, playerAct, depth):
    time_start = time.time()
    act_depth = 6
    while act_depth <= depth:
        move, construct = getBestMove(board, players, playerAct, act_depth)
        if time.time() - time_start > 0.5:
            print(act_depth)
            return move, construct
        act_depth += 1
    return move, construct


class TktBot3(Player):
    """
    Minimax + Monte Carlo Tree Search bot
    """

    def name(self):
        return "Tkt bot 3"

    # Placement of the pawns
    def place_pawn(self, board, pawn):
        my_choice = (2, 2) if (2, 2) in board.get_possible_movement_positions(
            pawn) else random.choice(board.get_possible_movement_positions(pawn))
        return my_choice

    # Movement and building
    def play_move(self, board, pawn):
        while True:
            try:
                return playProgressive(
                    np.array(board.board), List([player.pos for player in board.pawns]), pawn.number - 1, 20)
            except:
                print("Error ... Retrying ... (First compilation ?)")
