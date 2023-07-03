import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0,2*np.pi,5000)

square_wave = np.ones_like(x)
square_wave[int(x.size/2):]=-1

N = 30
fsq = np.zeros_like(x)
for i in range(N):
    n = 2*i + 1
    fsq += np.sin(n  *x) / n
fsq *= 4 / np.pi

fig, ax = plt.subplots()
ax.plot(x, square_wave, lw=5, alpha=0.5)
ax.plot(x, fsq, 'r')
ax.set_ylim(-1.2,1.2)
plt.title('N='+str(N))

ax.set_xticks([0,1,2,3,4,5,6,7])
ax.set_xticks([0.5,1.5,2.5,3.5,4.5,5.5,6.5], minor=True)
ax.set_yticks([-1, 0, 1])
ax.set_yticks(np.arange(-1.2,1.2,0.2), minor=True)

ax.grid(b=True, c='k', lw=1, ls='--', which='major')
ax.grid(b=True, c='0.4', lw=0.5, ls=':', which='minor')

plt.show()

