import random
from santorinai.player import Player

import threading
import time


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
            if board[x+i][y+j] - z < 2 and board[x+i][y+j] < 4 and (x+i, y+j) not in players:
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


def minimax(board, players, playerAct, depth, alpha, beta, maximizingPlayer):
    if win(board, *players[(playerAct-1) % 4]):
        return -1 if maximizingPlayer else 1
    if depth == 0:
        return 0
    for move in movesPlayer(board, players, *players[playerAct]):
        newPlayers = players[:]
        newPlayers[playerAct] = move
        for construct in constructsPlayer(board, newPlayers, *newPlayers[playerAct]):
            newBoard = [row[:] for row in board]
            newBoard[construct[0]][construct[1]] += 1
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
    return alpha if maximizingPlayer else beta


def getBestMove(board, players, playerAct, depth):
    bestValue = -2
    bestMove = None
    bestConstruct = None
    while bestMove == None:
        for move in movesPlayer(board, players, *players[playerAct]):
            newPlayers = players[:]
            newPlayers[playerAct] = move
            for construct in constructsPlayer(board, newPlayers, *newPlayers[playerAct]):
                if depth == 0:
                    return move, construct
                newBoard = [row[:] for row in board]
                newBoard[construct[0]][construct[1]] += 1
                value = minimax(newBoard, newPlayers, (playerAct+1) %
                                4, depth-1, bestValue, 1, False)
                if value != None and value > bestValue:
                    bestValue = value
                    bestMove = move
                    bestConstruct = construct
                    if bestValue >= 1:
                        return bestMove, bestConstruct
        depth -= 1
    return bestMove, bestConstruct


class threadWithReturn(threading.Thread):
    def __init__(self, *args, **kwargs):
        super(threadWithReturn, self).__init__(*args, **kwargs)
        self._return = None

    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args, **self._kwargs)


def playProgressive(board, players, playerAct, depth):
    time_start = time.time()
    act_depth = 0
    while act_depth <= depth:
        thread = threadWithReturn(target=getBestMove, args=(
            board, players, playerAct, act_depth))
        thread.start()
        while time.time() - time_start <= 5 and thread.is_alive():
            time.sleep(0.1)
        if time.time() - time_start > 5:
            print(act_depth - 1)
            return move, construct
        act_depth += 1
        move, construct = thread._return
    print(depth)
    return move, construct


class TktBot2(Player):
    """
    Minimax only
    """

    def name(self):
        return "Tkt bot 2"

    # Placement of the pawns
    def place_pawn(self, board, pawn):
        my_choice = (2, 2) if (2, 2) in board.get_possible_movement_positions(
            pawn) else random.choice(board.get_possible_movement_positions(pawn))
        return my_choice

    # Movement and building
    def play_move(self, board, pawn):
        return playProgressive(
            board.board, [player.pos for player in board.pawns], pawn.number - 1, 20)
