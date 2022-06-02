import numpy as np 
import matplotlib.pyplot as plt 

x = np.arange(50)

y = np.arange(1, 0, -0.02)
# plt.figure(figsize=(10, 10))
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

ax1.plot(x, y)
ax2.plot(x, y)

plt.show()