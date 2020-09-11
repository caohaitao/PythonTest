# -*- coding: utf-8 -*-
__author__ = 'ck_ch'
import numpy as np
from enum import Enum
BOARD_SIZE = 16

CHESS_TYPE_NONE = 0
CHESS_TYPE_SI2 = 1
CHESS_TYPE_HUO2 = 2
CHESS_TYPE_SI3 = 3
CHESS_TYPE_HUO3 = 4
CHESS_TYPE_SI4 = 5
CHESS_TYPE_HUO4 = 6
CHESS_TYPE_CHENG5 = 7

m_chess_type_score = {}
m_chess_type_score[CHESS_TYPE_NONE] = 0
m_chess_type_score[CHESS_TYPE_SI2] = 1
m_chess_type_score[CHESS_TYPE_HUO2] = 2
m_chess_type_score[CHESS_TYPE_SI3] = 3
m_chess_type_score[CHESS_TYPE_HUO3] = 6
m_chess_type_score[CHESS_TYPE_SI4] = 10
m_chess_type_score[CHESS_TYPE_HUO4] = 30
m_chess_type_score[CHESS_TYPE_CHENG5] = 80

class TPos():
    def __init__(self,tx,ty):
        self.x = tx
        self.y = ty

class Board():
    def __init__(self):
        self.m_board=np.zeros([BOARD_SIZE,BOARD_SIZE])
        self.m_xiexia_begin_pos = []
        self.m_xiehouxia_begin_pos = []
        self.InitBeginPos()

    def Reset(self):
        self.m_board = np.zeros([BOARD_SIZE,BOARD_SIZE])

    def ChangeBoard(self):
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if self.m_board[i][j] == 1:
                    self.m_board[i][j] = 2
                elif self.m_board[i][j] == 2:
                    self.m_board[i][j] = 1

    def InitBeginPos(self):
        for i in range(BOARD_SIZE-4):
            self.m_xiexia_begin_pos.append(TPos(i,0))
        for j in range(1,BOARD_SIZE-4):
            self.m_xiexia_begin_pos.append(TPos(0,j))

        for i in range(4,BOARD_SIZE):
            self.m_xiehouxia_begin_pos.append(TPos(i,0))

        for j in range(1,BOARD_SIZE-4):
            self.m_xiehouxia_begin_pos.append(TPos(BOARD_SIZE - 1,j))

    def __str__(self):
        return str(self.m_board)

    def GetScore(self):
        return 10

    def IsOver2(self,num,now_target,i,j):
        if now_target != self.m_board[i][j]:
            now_target = self.m_board[i][j]
            num = 0
        if self.m_board[i][j] != 0:
            num += 1
            if num == 5:
                return now_target,num,now_target
        return 0,num,now_target

    def IsOver(self):
        #纵向判断
        for i in range(BOARD_SIZE):
            num = 0
            now_target = 0
            for j in range(BOARD_SIZE):
                res,num,now_target = self.IsOver2(num,now_target,i,j)
                if res != 0:
                    return now_target
        #横向判断
        for j in range(BOARD_SIZE):
            num = 0
            now_target = 0
            for i in range(BOARD_SIZE):
                res,num,now_target = self.IsOver2(num,now_target,i,j)
                if res != 0:
                    return now_target

        #歇下判断
        for k in self.m_xiexia_begin_pos:
            i = k.x
            j = k.y
            num = 0
            now_target = 0
            while True:
                if i==BOARD_SIZE or j == BOARD_SIZE:
                    break
                res,num,now_target = self.IsOver2(num,now_target,i,j)
                if res != 0:
                    return now_target
                i += 1
                j += 1

        #邂逅下判断
        for k in self.m_xiehouxia_begin_pos:
            i = k.x
            j = k.y
            num = 0
            now_target = 0
            while True:
                if i==-1 or j==BOARD_SIZE:
                    break
                res,num,now_target = self.IsOver2(num,now_target,i,j)
                if res != 0:
                    return now_target
                i -= 1
                j += 1

        return 0

    def PushIJ(self,i,j,next_pos):
        if i<0 or i>(BOARD_SIZE-1) or j<0 or j>(BOARD_SIZE-1):
            return
        if self.m_board[i][j] != 0:
            return
        next_pos[i][j] = 1

    def GetNextPos(self,next_pos):
        width = 1
        for i in range(BOARD_SIZE - 1):
            for j in range(BOARD_SIZE - 1):
                if self.m_board[i][j] != 0:
                    for ii in range(i-width,i+width+1):
                        for jj in range(j-width,j+width+1):
                            self.PushIJ(ii,jj,next_pos)

    def GetPosChessType(self,i,j,add_i,add_j,value):
        old_i = i
        old_j = j
        num = 0
        IsTailLegal = True
        k = num
        i = i + num*add_i
        j = j + num*add_j
        for k in range(num,5):
            if i > (BOARD_SIZE-1) or j>(BOARD_SIZE-1) or i<0 or j<0:
                IsTailLegal = False
                break
            if self.m_board[i][j] != value:
                break
            num += 1
            i += add_i
            j += add_j
        if num < 2:
            return CHESS_TYPE_NONE
        if num == 5:
            return CHESS_TYPE_CHENG5
        du_num = 0
        i_pre = old_i - add_i
        j_pre = old_j - add_j
        if i_pre>(BOARD_SIZE-1) or j_pre>(BOARD_SIZE-1) or i_pre<0 or j_pre<0:
            du_num += 1
        else:
            if self.m_board[i_pre][j_pre] != 0:
                du_num += 1

        if IsTailLegal == False:
            du_num += 1
        else:
            if self.m_board[i][j] != 0:
                du_num += 1
        if du_num == 2:
            return CHESS_TYPE_NONE

        if du_num == 1:
            if num == 2:
                return CHESS_TYPE_SI2
            if num == 3:
                return CHESS_TYPE_SI3
            if num == 4:
                return CHESS_TYPE_SI4

        if num == 2:
            return CHESS_TYPE_HUO2
        if num == 3:
            return CHESS_TYPE_HUO3
        if num == 4:
            return CHESS_TYPE_HUO4
        return CHESS_TYPE_NONE

    def SetMaxTypeByij(self,chess_value,i,j,add_i,add_j,max_type,max_type2):
        ctp = self.GetPosChessType(i,j,add_i,add_j,chess_value)
        if ctp > max_type2:
            if ctp < max_type:
                max_type2 = ctp
            else:
                max_type2 = max_type
                max_type = ctp
        return max_type,max_type2

    def GetChessType2(self):
        wmax_type = CHESS_TYPE_NONE
        wmax_type2 = CHESS_TYPE_NONE
        bmax_type = CHESS_TYPE_NONE
        bmax_type2 = CHESS_TYPE_NONE
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if self.m_board[i][j] == 1:
                    wmax_type,wmax_type2 = self.SetMaxTypeByij(1,i,j,1,0,wmax_type,wmax_type2)
                    wmax_type,wmax_type2 = self.SetMaxTypeByij(1,i,j,0,1,wmax_type,wmax_type2)
                    wmax_type,wmax_type2 = self.SetMaxTypeByij(1,i,j,1,1,wmax_type,wmax_type2)
                    wmax_type,wmax_type2 = self.SetMaxTypeByij(1,i,j,-1,1,wmax_type,wmax_type2)
                elif self.m_board[i][j] == 2:
                    bmax_type,bmax_type2 = self.SetMaxTypeByij(2,i,j,1,0,bmax_type,bmax_type2)
                    bmax_type,bmax_type2 = self.SetMaxTypeByij(2,i,j,0,1,bmax_type,bmax_type2)
                    bmax_type,bmax_type2 = self.SetMaxTypeByij(2,i,j,1,1,bmax_type,bmax_type2)
                    bmax_type,bmax_type2 = self.SetMaxTypeByij(2,i,j,-1,1,bmax_type,bmax_type2)

        return wmax_type,wmax_type2,bmax_type,bmax_type2

    def GetScore2(self):
        wt1,wt2,bt1,bt2 = self.GetChessType2()
        return m_chess_type_score[wt1] + m_chess_type_score[wt2] \
                - m_chess_type_score[bt1] - m_chess_type_score[bt2]

    def GetChessType3SubFunc(self,max_type,max_type2,i,j,all_poss,count):
        index = self.m_board[i][j] - 1
        # row
        max_type[index],max_type2[index] = \
            self.SetMaxTypeByij(self.m_board[i][j], \
                i,j,1,0,max_type[index],max_type2[index])

        max_type[index],max_type2[index] = \
            self.SetMaxTypeByij(self.m_board[i][j], \
                i,j,0,1,max_type[index],max_type2[index])
        max_type[index],max_type2[index] = \
            self.SetMaxTypeByij(self.m_board[i][j], \
                i,j,1,1,max_type[index],max_type2[index])
        max_type[index],max_type2[index] = \
            self.SetMaxTypeByij(self.m_board[i][j], \
                i,j,-1,1,max_type[index],max_type2[index])

    def GetChessType3(self,all_poss,count):
        max_type = [CHESS_TYPE_NONE,CHESS_TYPE_NONE]
        max_type2 = [CHESS_TYPE_NONE,CHESS_TYPE_NONE]
        for k in range(count):
            i = all_poss[k].x
            j = all_poss[k].y
            self.GetChessType3SubFunc(max_type,max_type2,i,j,all_poss,count)
        return max_type[0],max_type2[0],max_type[1],max_type2[1]

    def GetScore3(self,all_poss,count):
        wt1,wt2,bt1,bt2 = self.GetChessType3(all_poss,count)
        return m_chess_type_score[wt1] + m_chess_type_score[wt2] \
                - m_chess_type_score[bt1] - m_chess_type_score[bt2]


if __name__ == "__main__":
    b = Board()
    b.m_board[5][0] = 1
    b.m_board[1][4] = 1
    b.m_board[2][3] = 1
    b.m_board[3][2] = 1
    # b.m_board[4][1] = 1
    print(b)
    print(b.GetScore2())
    # print(b.IsOver())
    # next_pos = np.zeros([BOARD_SIZE,BOARD_SIZE])
    # b.GetNextPos(next_pos)
    # print(next_pos)
    # b1 = [CHESS_TYPE_NONE,CHESS_TYPE_NONE]
    # b.SetMaxTypeByij(b1)
    # print(b1)