from sim import simulation
from Controllers import *
import numpy as np
import time
import scipy.io
import sys


class PrintArray:

    def __init__(self, **kwargs):
        self._kwargs = kwargs

    def __repr__(self):
        rpr = ('PrintArray(' +
               ', '.join([f'{name}={value}' for name, value in self._kwargs.items()]) +
               ')')
        return rpr

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if ufunc != np.floor_divide:
            return NotImplemented
        a = inputs[0]
        with np.printoptions(**self._kwargs):
            print(a)


def follow_trajectory(sim, traj, T=20, steps=500):
    '''
    This function passes a trajectory to the simulator
    
    ARGS:
        sim: the simulator object
        traj: the trajectory to follow
        T: time to follow trajectory
        steps: number of steps to follow trajectory
    '''
    
    
    for i in range(len(traj)):
        sim.Tref=traj[i]
        time.sleep(T/steps)




if __name__ == "__main__":
    sim=simulation()


    # ----------------- Defining controllers for the simulator -----------------
    OP_inverse_controller = OP_Space_inverse_controller(kp=20, kp_ori=20, kd=200)
    OP_inverse_ZYZ_controller = OP_Space_inverse_ZYZ_controller(kp=10, kd=200)
    g = grav_compensation_controller()
    joint_space_PDG = Joint_space_PDG_controller(kp=150, kd=50)
    SDD_control = SDD_controller(k=10)
    



    # ----------------- Adding controllers to the simulator -----------------
    sim.controllers.append(OP_inverse_controller)
    sim.controllers.append(SDD_control)
    # sim.controllers.append(g)
    # sim.start() 



    # ----------------- Trajectory Generation -----------------
    time.sleep(2)
    print("joint states: \n", sim.q0)
    sim.dq0 = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

    M = sim.robot.inertia(sim.q0)
    C = sim.robot.coriolis(sim.q0, sim.dq0)

    C_product = C@sim.dq0
    print("joints: ", sim.q0)
    print("joints speed: ", sim.dq0)


    p = PrintArray(precision=4, linewidth=150, suppress=True)
    print("\n")
    C_product // p
    print("\n")

    time.sleep(500)
    steps = 300

    q_goal   = [np.pi/2 , -np.pi/2.4, np.pi/2.4, -np.pi/2.2, np.pi,-np.pi/1.7,np.pi/1.7 , np.pi/2, -np.pi/2,0] 
    q_goal_2 =  [-np.pi/8 , -np.pi/2.4, np.pi/2.4, -np.pi/2.2, np.pi,-np.pi/1.7,np.pi/1.7 , np.pi/2, -np.pi/2,0]
    Tgoal=sim.robot.fkine(q_goal)
    Tgoal2=sim.robot.fkine(q_goal_2)


    Trj=rtb.ctraj(sim.Tref,Tgoal,steps)
    Trj_inv=rtb.ctraj(Tgoal,sim.Tref,steps)

    Trj_goal_2=rtb.ctraj(Tgoal,Tgoal2, steps)


    follow_trajectory(sim, traj=Trj, steps=steps)
    time.sleep(3)
    follow_trajectory(sim, traj=Trj_goal_2, steps=steps)
    time.sleep(3)
    follow_trajectory(sim, traj=Trj, steps=steps)
    time.sleep(3)
    follow_trajectory(sim, traj=Trj_goal_2, steps=steps)

    time.sleep(1000)