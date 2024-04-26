from sim import simulation
from Controllers import *
import numpy as np
import time

def follow_trajectory(sim, traj, T=10, steps=500):
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


    # Setting controllers
    OP_inverse_controller = OP_Space_inverse_controller(kp=10, kd=200)
    OP_inverse_ZYZ_controller = OP_Space_inverse_ZYZ_controller(kp=10, kd=200)
    g = grav_compensation_controller()
    joint_space_PDG = Joint_space_PDG_controller(kp=150, kd=50)
    SDD_control = SDD_controller(k=10)
    
    sim.controllers.append(g)
    sim.controllers.append(SDD_control)
    # sim.nullspace_controllers.append(SDD_control)

    sim.start() 



    steps = 100
    q_goal = [np.pi/2 , -np.pi/2.4, np.pi/2.4, -np.pi/2.2, np.pi,-np.pi/1.7,np.pi/1.7 , np.pi/2, -np.pi/2,0] 
    Tgoal=sim.robot.fkine(q_goal)

    Trj=rtb.ctraj(sim.Tref,Tgoal,steps)
    Trj_inv=rtb.ctraj(Tgoal,sim.Tref,steps)


    follow_trajectory(sim, traj=Trj, steps=steps)
    time.sleep(5)
    follow_trajectory(sim, traj=Trj_inv, steps=steps)



    time.sleep(1000)