import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 10, 0.1)  #0부터 10까지 0.1 씩 증가
y = np.sin(x)              #0부터 0.1씩 sin함수에 들어간다. 

plt.plot(x, y)

plt.show