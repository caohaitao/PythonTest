__author__ = 'ck_ch'
from enum import Enum
from Board import *
import numpy as np

# class Color(Enum):
#     red = 1
#     blue = 2
#
# print(Color['red'])
# print(Color(2))

class NODE_STATE(Enum):
    NONE = 0
    GUESS = 1
    DONE = 2
    DEL = 3

class Computer:
    def __init__(self):
        self.m_state = NODE_STATE.NONE
        self.m_father = None
        self.score = 0
        self.b = Board
        self.pos = TPos(0,0)
        self.use_child_pos = TPos(0,0)
        self.level = 0

    def GetAllMovePos(self,tps):
        if self.m_father == None:
            return
        tps.append(self.pos)
        self.m_father.GetAllMovePos(tps)

    def PushScoreToFather(self):
        if self.m_father == None:
            return
        if self.m_state == NODE_STATE.DONE:
            if self.m_father.m_state == NODE_STATE.NONE:
                self.m_father.m_state = NODE_STATE.GUESS
                self.m_father.score = self.score
                self.m_father.use_child_pos = self.pos
                self.m_father.PushScoreToFather()
            elif self.m_father.m_state == NODE_STATE.GUESS:
                if self.m_father.level%2  == 0:
                    if self.score < self.m_father.score:
                        self.m_father.score = self.score
                        self.m_father.use_child_pos = self.pos
                        self.m_father.PushScoreToFather()
                else:
                    if self.score > self.m_father.score:
                        self.m_father.score = self.score
                        self.m_father.use_child_pos = self.pos
                        self.m_father.PushScoreToFather()
        elif self.m_state == NODE_STATE.GUESS:
            if self.m_father.m_state == NODE_STATE.NONE:
                return
            elif self.m_father.m_state == NODE_STATE.GUESS:
                if self.m_father.level%2==1:
                    if self.score<self.m_father.score:
                        self.m_state = NODE_STATE.DEL
                        return
                else:
                    if self.score>self.m_father.score:
                        self.m_state = NODE_STATE.DEL
                        return

        return

    def MakeChildByAPos(self,i,j,whole_level):
        tempc = Computer
        tempc.m_state = NODE_STATE.NONE
        tempc.m_fathre = self
        tempc.b = self.b
        tempc.level = self.level + 1
        tempc.pos = TPos(i,j)
        if self.level%2 == 1:
            vale = 1
        else:
            vale = 2

        tempc.b.m_board[i][j] = vale
        tempc.MakeChild(whole_level)

    def MakeChild(self,whole_level):
        if self.level == whole_level or self.b.IsOver()!=0:
            poss_count = 0
            tps = []
            self.GetAllMovePos(tps)
            self.score = self.b.GetScore()
            self.m_state = NODE_STATE.DONE
            self.PushScoreToFather()
        else:
            next_pos = np.zeros([BOARD_SIZE,BOARD_SIZE])
            self.b.GetNextPos(next_pos)
            for i in range(BOARD_SIZE):
                for j in range(BOARD_SIZE):
                    if next_pos[i][j] != 0:
                        self.MakeChildByAPos(i,j,whole_level)
                        if self.m_state == NODE_STATE.DEL:
                            break

                if self.m_state == NODE_STATE.DEL:
                    break

            if self.m_state != NODE_STATE.DEL:
                self.m_state = NODE_STATE.DONE
                self.PushScoreToFather()


if __name__ == "__main__":
    c = Computer