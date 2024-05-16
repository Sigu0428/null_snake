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
    
    npTrj=np.zeros((7,len(traj)))

    for t in range(len(Trj)):
        Tr=np.array(Trj[t])
        npTrj[:3,t]=Tr[:3,3]
        npTrj[3:,t]=UnitQuaternion(sm.SE3(Tr)).vec
        #xyz quat velocity and acceleration from gradient MAYBE DOES NOT WORK FOR QUATS?
        Tvel=np.gradient(npTrj,T/steps,axis=1) #time to specify spacing
        Tacc=np.gradient(Tvel,T/steps,axis=1)
        #print(Tvel[:,:5])
        for i in range(steps):
            sim.xref=npTrj[:,i]
            sim.dxref=Tvel[:,i]
            sim.ddxref=Tacc[:,i]


        time.sleep(T/steps)




if __name__ == "__main__":
    sim=simulation()

    
    #fast=jitter. Jacdot is off. 

    # ----------------- Defining controllers for the simulator -----------------
    OP_inverse_controller = OP_Space_controller(kd=50, kp_trans=1000,kp_ori=1000)
    #OP_inverse_ZYZ_controller = OP_Space_inverse_ZYZ_controller(kp=10, kd=200)
    #g = grav_compensation_controller()
    #joint_space_PDG = Joint_space_PDG_controller(kp=150, kd=50)
    SDD_control = SDD_controller(k=1)
    



    # ----------------- Adding controllers to the simulator -----------------
    sim.controllers.append(OP_inverse_controller)
    sim.controllers.append(SDD_control)
    #sim.controllers.append(g)
    sim.start() 



    # ----------------- Trajectory Generation -----------------


    q_goal = [np.pi/2 , -np.pi/2.4, np.pi/2.4, -np.pi/2.2, np.pi,-np.pi/1.7,np.pi/1.7 , np.pi/2, -np.pi/2,0] 
    q_viapoint = [-np.pi/4, -np.pi/2.4, np.pi/2.4, -np.pi/2.2, np.pi,-np.pi/1.7,np.pi/1.7 , np.pi/2, -np.pi/2,0] 
    q_lmao=np.copy(sim.q0)
    q_lmao[0]+=-np.pi/10
    steps=[200,200,200]
    T=[8,8,5,3,3]
    viapoints=[sim.robot.fkine(q_goal)]
    #viapoints=[sim.robot.fkine(sim.q0)*sm.SE3.RPY(0,0,np.pi/2)] #zyx rot order    
    #viapoints.append(sim.robot.fkine(q_viapoint))

    viapoints.append(sim.robot.fkine(q_lmao))
    

    time.sleep(2)

    for j in range(len(viapoints)):
        if j==0:
            T_start=sim.robot.fkine(sim.getJointAngles())
        else:
            T_start=viapoints[j-1]
        Trj=rtb.ctraj(T_start,viapoints[j],steps[j]) #trapezoidal velocity profile
        npTrj=np.zeros((7,len(Trj)))
        #get orientation and translation
        for t in range(len(Trj)):
            Tr=np.array(Trj[t])
            npTrj[:3,t]=Tr[:3,3]
            npTrj[3:,t]=UnitQuaternion(sm.SE3(Tr)).vec
        #xyz quat velocity and acceleration from gradient
        Tvel=np.gradient(npTrj,T[j]/steps[j],axis=1) #time to specify spacing
        #print(np.gradient(Trj[0:2]))
        #exit(0)
        #print(npTrj[:,0:3])
        #print(Tvel[:,0:3])
        Tacc=np.gradient(Tvel,T[j]/steps[j],axis=1)
        starttime=time.time()
        currenttime=time.time()
        while(currenttime<(starttime+T[j])):
            currenttime=time.time()
            i=(((currenttime-starttime))/T[j])*steps[j]
      
            i=int(min(i,steps[j]-1))
            with sim.refLock:
                sim.xref=npTrj[:,i]
                sim.dxref=Tvel[:,i]
                sim.ddxref=Tacc[:,i]
        
            #print(sim.xref[:3])
            time.sleep(T[j]/steps[j])
        with sim.refLock:
            sim.dxref=np.zeros((7))
            sim.ddxref=np.zeros((7))

        time.sleep(5)

    time.sleep(1000)