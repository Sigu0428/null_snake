import matplotlib.pyplot as plt
import numpy as np


ob = [np.array([-0.2, 0.65]), np.array([0.45, -0.6])]
vec = np.zeros((40, 40, 2))
pnt = np.zeros((40, 40, 2))
mag = np.ones((40, 40))

t = 0.3
k = 0.1
phi = 10

for i, x in enumerate(np.arange(-2, 2, 0.1)):
    for j, y in enumerate(np.arange(-2, 2, 0.1)):
        p = np.array([x, y])
        pnt[i, j, :] = p
        
        dist = np.linalg.norm(ob[0] - p)
        dir = (ob[0] - p)/dist
        m = (k/(dist-0.3))*(np.tanh(-phi*((dist-0.3)-t))+1)/2
        v1 = -dir*m

        dist = np.linalg.norm(ob[1] - p)
        dir = (ob[1] - p)/dist
        m = (k/(dist-0.3))*(np.tanh(-phi*((dist-0.3)-t))+1)/2
        v2 = -dir*m

        vec[i, j, :] = (v1 + v2) / np.linalg.norm(v1 + v2)
        mag[i, j] = np.log(np.linalg.norm(v1 + v2))
        if np.linalg.norm(v1 + v2) < 0.001:
            vec[i, j, :] = np.array([0, 0])
            mag[i, j] = 0

        #vec[i, j, :] = dir
        if (np.linalg.norm(ob[1] - p) <= 0.3) or (np.linalg.norm(ob[0] - p) <= 0.3):
            vec[i, j, :] *= np.zeros((2))

#(magnetude/x)*(np.tanh(-10*(x-decayrate))+1)/2
#((o - pli)/np.linalg.norm(o - pli))

#fig3, ax3 = plt.subplots(1, 2)
ax3 = plt.gca()
ax3.set_title("Potential field")
Q = ax3.quiver(pnt[:, :, 0], pnt[:, :, 1], vec[:, :, 0], vec[:, :, 1], mag, units='x', pivot='tail', width=0.012,
               scale=10, cmap=plt.cm.Greys)
ax3.set_xlabel("x")
ax3.set_ylabel("y")
cbar = plt.colorbar(Q, cmap=plt.cm.Greys, label=r'log magnetude')
cbar.set_ticks([])
#cbar.set_ticklabels(["1", "2", "3", "4", "5", "aylmao"])
#qk = ax3.quiverkey(Q, 0.9, 0.9, 1, r'$1 \frac{m}{s}$', labelpos='E',
#                   coordinates='figure')
#ax3.scatter(pnt[:, :, 0], pnt[:, :, 1], color='0.5', s=1)
ax3.add_patch(plt.Circle((ob[0][0], ob[0][1]), 0.3+t, color='black', linestyle="dotted", fill=False, linewidth=2))
ax3.text(ob[0][0]+0.15+t, ob[0][1]+0.15+t, r'$t_{0.5}$', dict(size=12))
ax3.add_patch(plt.Circle((ob[1][0], ob[1][1]), 0.3+t, color='black', linestyle="dotted", fill=False, linewidth=2))
ax3.text(ob[1][0]+0.15+t, ob[1][1]+0.15+t, r'$t_{0.5}$', dict(size=12))
img = plt.imread("robot_topview.png")
ax3.imshow(img, extent=[-2, 2, -2, 2])

plt.show()

plt.clf()

x = np.arange(0.00001, 1, 0.01)
y0 = (k/x)*(np.tanh(-phi*(x-t))+1)/2
y1 = (np.tanh(-phi*(x-t))+1)/2
y2 = (k/x)

ax1 = plt.gca()
ax1.set_title("Gain functions")
ax1.plot(x, y0, c='r')
ax1.plot(x, y1, c='g')
ax1.plot(x, y2, c='b')
ax1.set_xlabel("distance from object")
ax1.set_ylabel("gain")
ax1.axvline(x=t, c='black', linestyle='dotted')
ax1.legend([r'$\frac{k}{x} \; \frac{\text{tanh}(-\phi(x-t_{0.5}))+1}{2}$', r'$\frac{\text{tanh}(-\phi(x-t_{0.5}))+1}{2}$', r'$\frac{k}{x}$', r'$t_{0.5}$'])
ax1.set_ylim(ymax=2, ymin = 0)
ax1.set_xlim(xmin=0, xmax=1)
ax1.text(t-0.02, -0.1, r'$t_{0.5}$', dict(size=12), clip_on=False)

plt.show()
