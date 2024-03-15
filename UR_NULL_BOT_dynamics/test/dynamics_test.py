#!/usr/bin/env python3
import numpy as np
np.set_printoptions(precision=6, suppress=True)

import ur_robot


N = 10

q =     np.array([1.0000,   1.0472,    1.0472, 1.0472, 1.0472, 1.0472, 1.0000,   1.0472,    1.0472, 1.0472]).T
dq =    np.array([1.0000,   1.0472,    1.0472, 1.0472, 1.0472, 1.0472, 1.0000,   1.0472,    1.0472, 1.0472]).T
ddq =   np.array([1.0000,   1.0472,    1.0472, 1.0472, 1.0472, 1.0472, 1.0000,   1.0472,    1.0472, 1.0472]).T

print(q)
print(dq)
print(ddq)

robot = ur_robot.URRobot(ur_robot.RobotType.NullBot)
print(robot)

grav = robot.gravity(q).reshape((N,1))
print("grav\n", grav)

jac = robot.jacobian(q)
print("Jacobian\n", jac)

jacDot = robot.jacobianDot(q, dq)
print("Jacobian dot\n", jacDot)

inertia = robot.inertia(q)
print("Inertia matrix\n", inertia)

coriolis = robot.coriolis(q, dq)
print("Coriolis matrix\n", coriolis)

print("Velocity product\n", coriolis @ dq)

tau = inertia @ ddq + coriolis @ dq + grav
print("Joint torque\n", tau)

print("Jac @ dq")
print(jac @ dq)

# ##
# robot = ur_robot.URRobot(ur_robot.RobotType.UR10e)
# print(robot)
#
# grav = robot.gravity(q).reshape((6,1))
# print("grav\n", grav)
#
# jac = robot.jacobian(q)
# print("Jacobian\n", jac)
#
# jacDot = robot.jacobianDot(q, dq)
# print("Jacobian dot\n", jacDot)
#
# inertia = robot.inertia(q)
# print("Inertia matrix\n", inertia)
#
# coriolis = robot.coriolis(q, dq)
# print("Coriolis matrix\n", coriolis)
#
# print("Velocity product\n", coriolis @ dq)
#
# tau = inertia @ ddq + coriolis @ dq + grav
# print("Joint torque\n", tau)
