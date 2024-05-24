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
    sim=simulation("dense")


    if sim.map=="dense":
        sim.obstacles=["sphere1","sphere2","sphere3","sphere4","cyl1","cyl2","cyl3","cyl4"]
    
    # ----------------- Defining controllers for the simulator -----------------
    OP_inverse_controller = OP_Space_controller(kd_trans=80,kd_ori=80, kp_trans=500,kp_ori=500,lambdaTraj=0.5)  #DEMO OP_Space_controller(kd_trans=80,kd_ori=80, kp_trans=500,kp_ori=500,lambdaTraj=0.5) 
    OP_vel_controller = OP_Space_Velocity_controller(kd_trans=40,kd_ori=40, kp_trans=280,kp_ori=300,Kv=50,lambdaTraj=0.5,lambdaAvoid=0.05) #DEMO: (kd_trans=40,kd_ori=40, kp_trans=280,kp_ori=300,Kv=50,lambdaTraj=0.5,lambdaAvoid=0.05) 
    #OP_inverse_ZYZ_controller = OP_Space_inverse_ZYZ_controller(kp=10, kd=200)
   # g = grav_compensation_controller()
    #joint_space_PDG = Joint_space_PDG_controller(kp=150, kd=50)
    SDD_control = SDD_controller(k=20,k_ground=15,threshold=0.25) #DEMO SDD_control = SDD_controller(k=20,k_ground=15,threshold=0.35)
    



    # ----------------- Adding controllers to the simulator -----------------
    sim.controllers.append(OP_vel_controller)
    #sim.controllers.append(OP_inverse_controller)
    #sim.controllers.append(SDD_control)
    #sim.controllers.append(g)
    sim.start() 

    #time.sleep(10000)
    # ----------------- Trajectory Generation -----------------

    q_snabel=[-np.pi*0.765, -np.pi/2.4, np.pi/2.4, -np.pi/2.2, np.pi,-np.pi/1.7,np.pi/1.7 , np.pi/2, -np.pi/2,0]
    T_snabel=sim.robot.fkine(q_snabel)*sm.SE3.Tz(0.5)
    q_goal = [np.pi/2 , -np.pi/2.4, np.pi/2.4, -np.pi/2.2, np.pi,-np.pi/1.7,np.pi/1.7 , np.pi/2, -np.pi/2,0] 
    q_goal2=  [np.pi/4 , -np.pi/2.4, np.pi/2.4, -np.pi/2.2, np.pi,-np.pi/1.7,np.pi/1.7 , np.pi/2, -np.pi/2,0] 
    q_viapoint = [-np.pi/4, -np.pi/2.4, np.pi/2.4, -np.pi/2.2, np.pi,-np.pi/1.7,np.pi/1.7 , np.pi/2, -np.pi/2,0] 
    q_back=np.copy(sim.q0)
    q_back[0]+=np.pi
    steps=[400,400,400,400,400]
    T=[10,8,12,10,12,12] #[8,8,8,5,3,4]
    #viapoints=[sim.robot.fkine(sim.q0)*sm.SE3.RPY(0,0,np.pi/2)] #zyx rot order
    #viapoints.append(viapoints[0]*sm.SE3.RPY(0,np.pi/2,0)) #zyx rot order
    #viapoints.append(viapoints[1]*sm.SE3.RPY(np.pi/2,0,0)) #zyx rot order
    #viapoints=[T_snabel]
    if sim.map=="dense":
        viapoints=[sim.robot.fkine(q_goal)]
        viapoints.append(sim.robot.fkine(q_goal2)*sm.SE3.Ty(-0.2))
        viapoints.append(sim.robot.fkine(q_viapoint))
        viapoints.append(T_snabel*sm.SE3.Tz(-1)*sm.SE3.Ty(-0.3)*sm.SE3.Tx(-0.2)*sm.SE3.Tz(-0.1))
        viapoints.append(T_snabel*sm.SE3.Ty(-0.1))
    else:
        viapoints=[sim.robot.fkine(q_goal)]
        viapoints.append(sim.robot.fkine(q_goal2))
        #viapoints.append(sim.robot.fkine(q_goal)*sm.SE3.Tz(0.5)*sm.SE3.RPY(np.pi/2,0,0))
        viapoints.append(sim.robot.fkine(q_viapoint))
    #viapoints.append(sim.robot.fkine(q_viapoint)*sm.SE3.RPY(0,0,np.pi/2)) #zyx rot order

    time.sleep(3)
    
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
        #xyz velocoity and acceleration from gradient
        Tvel=np.gradient(npTrj,T[j]/steps[j],axis=1) #time to specify spacing
        Tacc=np.gradient(Tvel,T[j]/steps[j],axis=1)
        #slerp velocity and acceleration for quats:
        q0=UnitQuaternion(npTrj[3:,0])
        q1=UnitQuaternion(npTrj[3:,-1])

        htrap=rtb.trapezoidal_func(0,1,T[j]) #scalar trapezoidal profile which slerp follows

        #Extract angle from quat:
        #u0hat=q0.v/np.linalg.norm(q0.v)
        #theta0=np.arccos(q0.s)
        #logarithm:
        #print((0,u0hat*theta0))

        #theta1=np.arccos(q1.s)
        #Dtheta=theta1-theta0


     
        #Make orientational derivatives
        if not q1==q0:
            for step in range(len(Trj)): 
                # 0<=t<=1
                #npTrj[3:,t]=sim.quatpower(q1*q0.inv(),t_mod[step+1])*q0  #SLERP
                if step==0:
                    sim.dxref=np.zeros(7)
                    sim.ddxref=np.zeros(7)
                else:
                    #q0*sim.quatpower((q0.inv()*q1),t_mod[step]) === petercorke slerp quat
                    
                    ht,htd,htdd=htrap((step/steps[j])*T[j])
                    ht=ht[0];htd=htd[0];htdd=htdd[0];
                    
                    slerpquat=q0*sim.quatpower((q0.conj()*q1),ht)
                    logquat=UnitQuaternion(q0.inv()*q1).log() 
                    Tvel[3:,step]=slerpquat*logquat*htd
                    Tacc[3:,step]=slerpquat*logquat*htdd+slerpquat*logquat*logquat*(htd**2)
                    # check if unit quat matters ( prob not)
        #print(Tvel[3:,:])
        starttime=sim.simtime

        while sim.simtime < (starttime+T[j]):
       
            i= min(int(((sim.simtime-starttime)/T[j])*steps[j]),steps[j]-1) #read sim time

            with sim.refLock:
                sim.xref=npTrj[:,i]
                sim.dxref=Tvel[:,i]
                sim.ddxref=Tacc[:,i]



            time.sleep((T[j]/steps[j]))
  

        with sim.refLock:

            sim.dxref=np.zeros(7)
            sim.ddxref=np.zeros(7)
    
        time.sleep(4)


    time.sleep(1000)