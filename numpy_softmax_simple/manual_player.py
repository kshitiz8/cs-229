import threading
import logging
import time
import Queue
import gym
import numpy

logging.basicConfig(level=logging.DEBUG,format='[%(levelname)s] (%(threadName)-10s) %(message)s')
global isWindows

isWindows = False
try:
    from win32api import STD_INPUT_HANDLE
    from win32console import GetStdHandle, KEY_EVENT, ENABLE_ECHO_INPUT, ENABLE_LINE_INPUT, ENABLE_PROCESSED_INPUT
    isWindows = True
except ImportError as e:
    import sys
    import select
    import termios


class KeyPoller(threading.Thread):
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs=None, verbose=None):

        threading.Thread.__init__(self, group=group, target=target, name=name,
                                  verbose=verbose)
        global isWindows
        if isWindows:
            self.readHandle = GetStdHandle(STD_INPUT_HANDLE)
            self.readHandle.SetConsoleMode(ENABLE_LINE_INPUT|ENABLE_ECHO_INPUT|ENABLE_PROCESSED_INPUT)

            self.curEventLength = 0
            self.curKeysLength = 0

            self.capturedChars = []
        else:
            # Save the terminal settings
            self.fd = sys.stdin.fileno()
            self.new_term = termios.tcgetattr(self.fd)
            self.old_term = termios.tcgetattr(self.fd)

            # New terminal setting unbuffered
            self.new_term[3] = (self.new_term[3] & ~termios.ICANON & ~termios.ECHO)
            termios.tcsetattr(self.fd, termios.TCSAFLUSH, self.new_term)

        self.q = kwargs['q']

    def exit(self):
        if isWindows:
            pass
        else:
            termios.tcsetattr(self.fd, termios.TCSAFLUSH, self.old_term)

    def run(self):
        while True:
            c = self.poll()
            if c is not None:

                self.q.put_nowait(c)

                if c == 'q':
                    break
                print(c)
                time.sleep(0.001)

    def poll(self):
        if isWindows:
            if not len(self.capturedChars) == 0:
                return self.capturedChars.pop(0)

            eventsPeek = self.readHandle.PeekConsoleInput(10000)

            if len(eventsPeek) == 0:
                return None

            if not len(eventsPeek) == self.curEventLength:
                for curEvent in eventsPeek[self.curEventLength:]:
                    if curEvent.EventType == KEY_EVENT:
                        if ord(curEvent.Char) == 0 or not curEvent.KeyDown:
                            pass
                        else:
                            curChar = str(curEvent.Char)
                            self.capturedChars.append(curChar)
                self.curEventLength = len(eventsPeek)

            if not len(self.capturedChars) == 0:
                return self.capturedChars.pop(0)
            else:
                return None
        else:
            dr,dw,de = select.select([sys.stdin], [], [], 0)
            if not dr == []:
                return sys.stdin.read(1)
            return None


class GameStep(threading.Thread):
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs=None, verbose=None):
        threading.Thread.__init__(self, group=group, target=target, name=name,
                                  verbose=verbose)
        self.q = kwargs['q']
        self.env = kwargs['env']
        self.done = True

    def run(self):
        while True:
            c = "empty"
            try:
                key = self.q.get_nowait()
                if key == 'q':
                    break
                if key =='s' and self.done:
                    self.env.reset()
                    self.done = False

                if not self.done:
                    self.env.render()
                    observation, reward, done, info = self.env.step()
            except Queue.Empty:
                pass
            print('game_step '+c)
            time.sleep(0.1)


q = Queue.Queue()


g = GameStep(name='daemon', kwargs={'q': q, 'env': gym.make("Breakout-v0")})
g.setDaemon(True)


t = KeyPoller(name="keyPoller", kwargs={'q': q})
t.setDaemon(True)

g.start()
t.start()

g.join()
t.join()

t.exit()