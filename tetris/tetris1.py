#!/usr/bin/env python2
# -*- coding: utf-8 -*-

# NOTE FOR WINDOWS USERS:
# You can download a "exefied" version of this game at:
# http://kch42.de/progs/tetris_py_exefied.zip
# If a DLL is missing or something like this, write an E-Mail (kevin@kch42.de)
# or leave a comment on this gist.

# Very simple tetris implementation
# 
# Control keys:
#       Down - Drop stone faster
# Left/Right - Move stone
#         Up - Rotate Stone clockwise
#     Escape - Quit game
#          P - Pause game
#     Return - Instant drop
#
# Have fun!

# Copyright (c) 2010 "Kevin Chabowski"<kevin@kch42.de>
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.


from random import randrange as rand

import pygame, sys, os
from pygame import surfarray as sa


from random import randint as rint
from key_poller import KeyPoller
import Queue
import time
import threading
try:
   import cPickle as pickle
except:
   import pickle
import numpy as np




# The configuration
cell_size = 18
cols = 10
rows = 22
maxfps = 30

colors = [
    (0, 0, 0),
    (255, 85, 85),
    (100, 200, 115),
    (120, 108, 245),
    (255, 140, 50),
    (50, 120, 52),
    (146, 202, 73),
    (150, 161, 218),
    (35, 35, 35)  # Helper color for background grid
]

# Define the shapes of the single parts
tetris_shapes = [
    [[1, 1, 1],
     [0, 1, 0]],

    [[0, 2, 2],
     [2, 2, 0]],

    [[3, 3, 0],
     [0, 3, 3]],

    [[4, 0, 0],
     [4, 4, 4]],

    [[0, 0, 5],
     [5, 5, 5]],

    [[6, 6, 6, 6]],

    [[7, 7],
     [7, 7]]
]


def rotate_clockwise(shape):
    return [[shape[y][x]
             for y in xrange(len(shape))]
            for x in xrange(len(shape[0]) - 1, -1, -1)]


def check_collision(board, shape, offset):
    off_x, off_y = offset
    for cy, row in enumerate(shape):
        for cx, cell in enumerate(row):
            try:
                if cell and board[cy + off_y][cx + off_x]:
                    return True
            except IndexError:
                return True
    return False


def remove_row(board, row):
    del board[row]
    return [[0 for i in xrange(cols)]] + board


def join_matrixes(mat1, mat2, mat2_off):
    off_x, off_y = mat2_off
    for cy, row in enumerate(mat2):
        for cx, val in enumerate(row):
            mat1[cy + off_y - 1][cx + off_x] += val
    return mat1


def new_board():
    board = [[0 for x in xrange(cols)]
             for y in xrange(rows)]
    board += [[1 for x in xrange(cols)]]
    return board


class TetrisApp(object):
    def __init__(self):
        pygame.init()
        ### pygame.key.set_repeat(250, 25)
        self.width = cell_size * (cols + 6)
        self.height = cell_size * rows
        self.rlim = cell_size * cols
        self.bground_grid = [[8 if x % 2 == y % 2 else 0 for x in xrange(cols)] for y in xrange(rows)]
        self.max_move_action = 3;
        self.default_font = pygame.font.Font(pygame.font.get_default_font(), 12)

        self.screen = pygame.display.set_mode((self.width, self.height))
        ### pygame.event.set_blocked(pygame.MOUSEMOTION)  # We do not need
        # mouse movement
        # events, so we
        # block them.
        self.next_stone = tetris_shapes[rand(len(tetris_shapes))]
        self.init_game()

    def new_stone(self):
        self.stone = self.next_stone[:]
        self.next_stone = tetris_shapes[rand(len(tetris_shapes))]
        self.stone_x = int(cols / 2 - len(self.stone[0]) / 2)
        self.stone_y = 0

        if check_collision(self.board,self.stone,(self.stone_x, self.stone_y)):
            self.gameover = True
            return -100
        return 0


    def init_game(self):
        self.board = new_board()
        self.new_stone()
        self.level = 1
        self.score = 0
        self.lines = 0
        self.gameover=False
        self.move_action = 0
        ### pygame.time.set_timer(pygame.USEREVENT + 1, 1000)

    def disp_msg(self, msg, topleft):
        x, y = topleft
        for line in msg.splitlines():
            self.screen.blit(
                self.default_font.render(
                    line,
                    False,
                    (255, 255, 255),
                    (0, 0, 0)),
                (x, y))
            y += 14

    def center_msg(self, msg):
        for i, line in enumerate(msg.splitlines()):
            msg_image = self.default_font.render(line, False,
                                                 (255, 255, 255), (0, 0, 0))

            msgim_center_x, msgim_center_y = msg_image.get_size()
            msgim_center_x //= 2
            msgim_center_y //= 2

            self.screen.blit(msg_image, (
                self.width // 2 - msgim_center_x,
                self.height // 2 - msgim_center_y + i * 22))

    def draw_matrix(self, matrix, offset):
        off_x, off_y = offset
        for y, row in enumerate(matrix):
            for x, val in enumerate(row):
                if val:
                    pygame.draw.rect(
                        self.screen,
                        colors[val],
                        pygame.Rect(
                            (off_x + x) *
                            cell_size,
                            (off_y + y) *
                            cell_size,
                            cell_size,
                            cell_size), 0)

    def add_cl_lines(self, n):
        linescores = [0, 40, 100, 300, 1200]
        self.lines += n
        self.score += linescores[n] * self.level
        return linescores[min(n,4)]

    def move(self, delta_x):
        if not self.gameover:
            new_x = self.stone_x + delta_x
            if new_x < 0:
                new_x = 0
            if new_x > cols - len(self.stone[0]):
                new_x = cols - len(self.stone[0])
            if not check_collision(self.board,
                                   self.stone,
                                   (new_x, self.stone_y)):
                self.stone_x = new_x

    def quit(self):
        self.gameover = True
        self.center_msg("Exiting...")
        pygame.display.update()

    def drop(self):
        if not self.gameover:
            sc = 1
            self.stone_y += 1
            if check_collision(self.board,
                               self.stone,
                               (self.stone_x, self.stone_y)):
                self.board = join_matrixes(
                    self.board,
                    self.stone,
                    (self.stone_x, self.stone_y))
                sc = sc + self.new_stone()
                cleared_rows = 0
                while True:
                    for i, row in enumerate(self.board[:-1]):
                        if 0 not in row:
                            self.board = remove_row(self.board, i)
                            cleared_rows += 1
                            break
                    else:
                        break
                sc = sc + self.add_cl_lines(cleared_rows)
            return sc
        return 0


    def rotate_stone(self):
        if not self.gameover:
            new_stone = rotate_clockwise(self.stone)
            if not check_collision(self.board,
                                   new_stone,
                                   (self.stone_x, self.stone_y)):
                self.stone = new_stone

    def step(self,action):
        if self.gameover:
            return None

        action_map = {
            0: self.quit,
            1: self.drop,
            2: lambda: self.move(-1),
            3: lambda: self.move(+1),
            4: self.rotate_stone,
        }
        reward = 0
        if action == 1 or action not in action_map.keys() or self.move_action >= self.max_move_action:
            reward = self.drop()
            self.move_action = 0
        elif 2 <= action <= 4:
            action_map[action]()
            self.move_action = self.move_action + 1
        else:
            action_map[action]()



        self.update_screen()
        return self.prepro(),reward,self.gameover,{}

    def rgb2gray(self,rgb):
        r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray.astype(np.int)

    def prepro(self):
        """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
        I = sa.array3d(self.screen)[0:180]  # crop
        I[I==35] = 0
        I = self.rgb2gray(I)
        I = I[::9, ::9]  # downsample by factor of 9
        I[I != 0] = 1
        return I

    def reset(self):
        if self.gameover:
            self.init_game()

    def update_screen(self):
        self.screen.fill((0, 0, 0))
        if self.gameover:
                self.center_msg("""Game Over!\nYour score: %d Press space to continue""" % self.score)
        else:
            pygame.draw.line(self.screen,(255, 255, 255),(self.rlim + 1, 0),(self.rlim + 1, self.height - 1))
            self.disp_msg("Next:", (self.rlim + cell_size,2))
            self.disp_msg("Score: %d\n\nLevel: %d\nLines: %d" % (self.score, self.level, self.lines),
                          (self.rlim + cell_size, cell_size * 5))
            self.draw_matrix(self.bground_grid, (0, 0))
            self.draw_matrix(self.board, (0, 0))
            self.draw_matrix(self.stone,(self.stone_x, self.stone_y))
            self.draw_matrix(self.next_stone, (cols + 1, 2))

    def render(self):
        pygame.display.update()


def key_handler(q):
    with KeyPoller() as keyPoller:
        while True:
            key = keyPoller.poll()
            if key is not None:
                q.put_nowait(key)
                if key == 'q' or key == 'p':
                    break

            time.sleep(0.001)


def player(q, test=False):
    up = '\x1b[A'
    down = '\x1b[B'
    right = '\x1b[C'
    left = '\x1b[D'
    quit = 'q'
    quit_no_save = 'p'
    start = ' '

    action_map = {
        quit: 0,
        quit_no_save: 0,
        down : 1,
        left: 2,
        right : 3,
        up : 4,
    }

    user = os.environ.get('USER')
    uid = str(int(round(time.time())))
    dir = "./test/" if test else "./train/"
    f = open(dir + "_".join([user, uid, "tetris.p"] ) , 'wb')
    done = False
    episode = 1
    done = False
    obj = []
    obs = None
    App = TetrisApp()
    dump_now = 0
    dump_threshold = 100
    step  = 0

    while True:
        key = None
        try:
            key = q.get_nowait()
            print key[0]
        except Queue.Empty:
            pass

        if done and key == start:
            done = False

        if done :
            continue

        action = 1 if key is None or key == down or key not in action_map.keys() else action_map[key]
        t = 0.002 if key == down else 0.2

        if obs is not None:
            step  = step + 1
            dump_now = dump_now + 1

            obj.append({"obs":obs, "action": action, "reward":reward, "step" : step})


        if dump_now > dump_threshold:
            print "dumping to file"
            pickle.dump({"ep":episode, "step":step, "obj": obj},f)
            dump_now = 0
            obj = []

        obs, reward, done, info = App.step(action)

        if done:
            if key == quit_no_save : break
            pickle.dump({"ep":episode, "step":step, "obj": obj},f)
            if key == quit : break
            dump_now = 0
            obj = []
            step = 0
            episode = episode + 1
            obs = None
            App.reset()


        print action,reward
        App.render()
        time.sleep(t)

    f.close()


if __name__ == '__main__':
    q = Queue.Queue()
    q.empty()
    test = len(sys.argv)>1 and sys.argv[1].lower() == "test"

    p = threading.Thread(target=player, name='player',args=(q,test,))
    p.setDaemon(True)

    k = threading.Thread(target=key_handler, name='key_handler',args=(q,))
    k.setDaemon(True)

    p.start()
    k.start()

    p.join()
    k.join()

    # for i in range(1000):
    #     obs, reward, done, info = App.step(rint(1,4))
    #     print reward, done
    #     if done: break
    #     App.render()
    #     time.sleep(t)
