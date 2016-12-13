try:
   import cPickle as pickle
except:
   import pickle
import matplotlib.pyplot as plt
import time
import numpy as np

f = open("tripathi_1481558046_tetris.p" , 'r')

while 1:
    try:
        obj = pickle.load(f)
        print obj['ep']
        break
    except EOFError:
        print("jkj")
        break
print (obj['obj'][100].keys())
plt.imshow(obj['obj'][100]['obs'])
plt.show()
time.sleep(5)
f.close()