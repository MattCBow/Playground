"""----------------------------------------------
-----------------Matt Bowyer---------------------
----------------------------------------------"""


import math
import numpy

def spiral(n):
    s = int(math.ceil(math.sqrt(n)))
    spiral = numpy.matrix([[-1 for x in range(s)] for y in range(s)])
    move = 0
    position = (0,s-1)
    i = s*s
    while spiral[(position)] == -1:
        spiral[(position)] = (i-1)
        i -= 1
        next_pos = next_move(move, position)
        need_to_move = False
        if next_pos[0] not in range(s) or next_pos[1] not in range(s) or spiral[(next_pos)] != -1 :
            move = (move+1)% 4
            next_pos = next_move(move, position)
        position = next_pos
    return spiral


def next_move(move, (row,collum)):
    if move == 0: #LEFT
        return (row, collum-1)
    if move == 1: #DOWN
        return (row+1, collum)
    if move == 2: #RIGHT
        return (row, collum+1)
    if move == 3: #UP
        return (row-1, collum)
    return (0,0)
