

import roboticstoolbox as rtb
import numpy as np
import spatialmath as sm
np.set_printoptions(precision=6, suppress=True)
robot_base = sm.SE3.Trans(0,0,0)


pl1 = np.array([0,      -0.02561,   0.00193])
pl2 = np.array([0.2125, 0,          0.11336])
pl3 = np.array([0.15,   0.0,        0.0265])
pl4 = np.array([0,      -0.0018,    0.01634])

pl5 = np.array([0,      -0.02561,   0.00193])
pl6 = np.array([0.2125, 0,          0.11336])
pl7 = np.array([0.15,   0.0,        0.0265])
pl8 = np.array([0,      -0.0018,    0.01634])
pl9 = np.array([0,      0.0018,     0.01634])
pl10 = np.array([0,      0,          -0.001159])


i1 = np.array([[0.0084, 0., 0.],    [0., 0.0064, 0.],   [0., 0., 0.0064]])
i2 = np.array([[0.0078, 0., 0.],    [0., 0.21, 0.],   [0., 0., 0.21]])
i3 = np.array([[0.0016, 0., 0.],    [0., 0.0462, 0.],   [0., 0., 0.0462]])
i4 = np.array([[0.0, 0.0, 0.0],    [0.0, 0.0, 0.0],   [0.0, 0.0, 0.0]])

i5 = np.array([[0.0084, 0., 0.],    [0., 0.0064, 0.],   [0., 0., 0.0064]])
i6 = np.array([[0.0078, 0., 0.],    [0., 0.21, 0.],   [0., 0., 0.21]])
i7 = np.array([[0.0016, 0., 0.],    [0., 0.0462, 0.],   [0., 0., 0.0462]])
i8 = np.array([[0.0, 0.0, 0.0],    [0.0, 0.0, 0.0],   [0.0, 0.0, 0.0]])
i9 = np.array([[0.0, 0.0, 0.0],    [0.0, 0.0, 0.0],   [0.0, 0.0, 0.0]])
i10 = np.array([[0.0001, 0.0, 0.0],    [0.0, 0.0001, 0.0],   [0.0, 0.0, 0.0001]])

m1 = 3.761
m2 = 8.058
m3 = 2.846
m4 = 1.37

m5 = 3.761
m6 = 8.058
m7 = 2.846
m8 = 1.37
m9 = 1.3
m10 = 0.365



robot = rtb.DHRobot( 
    [ 
        #first UR
        rtb.RevoluteDH(d=0.1625, alpha = np.pi/2, m=m1, r=pl1, I=i1), #theta=np.pi either here or in XML using ref=""
        rtb.RevoluteDH(a=-0.425, m=m2, r=pl2, I=i2),
        rtb.RevoluteDH(a=-0.3922, m=m3, r=pl3, I=i3),
        rtb.RevoluteDH(d=0.1333, alpha=np.pi/2, m=m4, r=pl4, I=i4),
        #second UR
        rtb.RevoluteDH(d=0.1625, alpha = np.pi/2, m=m5, r=pl5, I=i5), #theta=pi
        rtb.RevoluteDH(a=-0.425, m=m6, r=pl6, I=i6),
        rtb.RevoluteDH(a=-0.3922, m=m7, r=pl7, I=i7),
        rtb.RevoluteDH(d=0.1333, alpha=np.pi/2, m=m8, r=pl8, I=i8),
        rtb.RevoluteDH(d=0.0997, alpha=-np.pi/2, m=m9, r=pl9, I=i9),
        rtb.RevoluteDH(d=0.0996, m=m10, r=pl10, I=i10)
    ], name="UR5e",
    base=robot_base,
    #add tool frame if used: tool=tool_matrix


    )

print(robot.gravload([1.0000,   1.0472,    1.0472, 1.0472, 1.0472, 1.0472, 1.0000,   1.0472,    1.0472, 1.0472]))

print(robot.inertia([1.0000,   1.0472,    1.0472, 1.0472, 1.0472, 1.0472, 1.0000,   1.0472,    1.0472, 1.0472]))

