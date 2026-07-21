import numpy as np
import matplotlib.pyplot as plt


x = np.array([1, 2, 3, 4, 5, 6])
#x = np.asarray( [1,2,3,4,5,6])
#x = np.asarray(range(1,7))
print('x=', x)

y = [4, 9, 2, 4, 5, 6]
z = [1, 2, 3, 4, 5, 6]
k = [11, 12, 13, 4, 5, 6]

ax = plt.subplot(111)
ax.bar(x-0.2, y, width=0.2, color='b', align='center')
ax.bar(x, z, width=0.2, color='g', align='center')
ax.bar(x+0.2, k, width=0.2, color='r', align='center')
ax.set(xlabel='# of PCs')
ax.set(ylabel='$c$')

plt.show()