import numpy as np
import roboticstoolbox as rtb
import spatialmath as sm
from spatialmath import UnitQuaternion
from spatialmath.base import q2r, r2x, rotx, roty, rotz,tr2eul,tr2rt
import robot_matrices as rm
from numpy.linalg import norm, pinv, inv, det


'''
This python file handles all the controllers which will be used by the simulator.
'''


class OP_Space_inverse_controller:
    '''
    This class implements a OP_SPACE_INVERSE controller
    Remember to set the Kp and Kd gains
    '''
    def __init__(self, kd=1, kp=1,kp_ori=1):
        self.Kp = kp
        self.kp_ori = kp_ori
        self.Kd = kd
    

        

    def get_u(self, sim):
        '''
        This function sets u based on a OP_SPACE_INVERSE controller
            ARGS:
                sim: the simulator object
        '''

        dim_analytical=7
        Kp=np.eye(dim_analytical)
        Kp[:3,:3]=np.eye(int(np.floor(dim_analytical/2)))*self.Kp #translational gain
        Kp[3:,3:]=np.eye(int(np.ceil(dim_analytical/2)))*self.kp_ori#orientational gain
        Kd=np.eye(dim_analytical)*self.Kd

        # get tool orientation quaternion and analytical jacobian

        T_ee=np.array(sim.getObjFrame(sim.tool_name))

        # xtilde
        x_tilde=np.zeros(dim_analytical)
        # relative translation
        x_tilde[:3]=np.array(sim.Tref)[:3,3]-T_ee[:3,3] #translational error
        
        # relative orientation by quaternions:
        JA,q_ee,JAdot=sim.getAnalyticalJacobian()
        q_ref=UnitQuaternion(sim.Tref)
        x_tilde[3:]=q_ref.vec-q_ee #works with difference, 

        #xtilde dot: if trajectory is linear interp, this is 0
        x_tilde_dot=np.zeros(dim_analytical) #velocity error in operational space

        x_desired_ddot=np.zeros(dim_analytical) #desired acceleration

        #control signal
        q=sim.getJointAngles()
        dq = np.array(sim.getJointVelocities())
        
        gq= sim.robot.gravload(q)
        B = sim.robot.inertia(q)
        C = sim.robot.coriolis(q,dq)

        y = np.linalg.pinv(JA)@(x_desired_ddot+Kd@x_tilde_dot+Kp@x_tilde-JAdot@dq)

        u = gq+B@y + C@dq
    
        # print((np.linalg.norm(x_tilde[:3]),np.linalg.norm(x_tilde[3:]))) 
        return u


class OP_Space_controller:
    '''
    Follows trajectory better, maybe small probability of instability
    '''

    def __init__(self, kd=1, kp_trans=1,kp_ori=1):
        self.kp_trans = kp_trans
        self.kp_ori = kp_ori
        self.kd = kd


    def get_u(self, sim):

        dim_analytical=7
        Kp=np.eye(dim_analytical)
        Kp[:3,:3]=np.eye(int(np.floor(dim_analytical/2)))*self.kp_trans #translational gain
        Kp[3:,3:]=np.eye(int(np.ceil(dim_analytical/2)))*self.kp_ori #orientational gain

        Kd=np.eye(dim_analytical)*self.kd #velocity gain

        #get references
        while not sim.refmutex: pass
        sim.refmutex=0
        xref=sim.xref
        dxref=sim.dxref
        ddxref=sim.ddxref
        sim.refmutex=1

        dq = np.array(sim.getJointVelocities())

        # get tool orientation quaternion and analytical jacobian

        T_ee=np.array(sim.getObjFrame(sim.tool_name))

        Je,Je_dot=sim.getGeometricJacs()

        #print(sim.robot.jacob0_dot(q,dq))
        #print("....................")
        #print(Je_dot)        
    
        # xtilde
        x_tilde=np.zeros(dim_analytical)
        # relative translation

        x_tilde[:3]=xref[:3]-T_ee[:3,3] #translational error
        
        # relative orientation by quaternions:
        JA,q_ee,JAdot=sim.getAnalyticalJacobian(Je,Je_dot)
        q_ref=xref[3:]
        
        x_tilde[3:]= q_ref-q_ee

        # velocity error 

        x_tilde_dot=np.zeros(dim_analytical) 

        
        x_dot= JA@dq #dx=JA*dq
        
        x_tilde_dot= dxref-x_dot

        x_desired_ddot=ddxref #desired acceleration

        #control signal

        #gq= self.robot.gravload(q)
        B = sim.getM()
        #C = self.robot.coriolis(q,dq)
        n=sim.getBiasForces()
        #print((x_desired_ddot+Kd@x_tilde_dot+Kp@x_tilde-JAdot@dq))
    
        #clamp pseudoinverse if manip low
        #if np.sqrt(np.linalg.det(JA@JA.T)) >= 1e-2:
        #    JA_inv = np.linalg.pinv(JA)
        #else:
        #    JA_inv = np.linalg.pinv(JA, rcond=1e-2)
        JA_inv=np.linalg.pinv(JA)
        y = JA_inv@(x_desired_ddot+Kd@x_tilde_dot+Kp@x_tilde-JAdot@dq)

        u = B@y + n

        return u
    



class OP_Space_inverse_ZYZ_controller:
  
    def __init__(self, kd=1, kp=1):
        self.Kp = np.eye(6)*kp
        self.Kd = np.eye(6)*kd


    def get_u(self, sim):
        '''
        This function sets u based on a OP_SPACE_INVERSE controller
            ARGS:
                sim: the simulator object. Is used to gain all informations needed to calculate the control signal
        '''


        dim_analytical=6
        # get tool orientation quaternion and analytical jacobian

        T_ee=np.array(sim.getObjFrame(sim.tool_name))

        # xtilde
        x_tilde=np.zeros(dim_analytical)
        # relative translation
        x_tilde[:3]=np.array(sim.Tref)[:3,3]-T_ee[:3,3] #translational error
        

        JA=sim.robot.jacob0_analytical(sim.getJointAngles(),representation="eul")
        JAdot=sim.robot.jacob0_dot(sim.getJointAngles(),sim.getJointVelocities(),representation="eul")
        #rotation error as zyz euler

        obj_q = sim.d.body(sim.tool_name).xquat
        q_ee=UnitQuaternion(obj_q).vec   
        x_tilde[3:]=sim.Tref.eul()-r2x(q2r(q_ee),"eul")

        #xtilde dot: if trajectory is linear interp, this is 0
        x_tilde_dot=np.zeros(dim_analytical) #velocity error in operational space

        x_desired_ddot=np.zeros(dim_analytical) #desired acceleration

        #control signal
        q=sim.getJointAngles()
        dq = np.array(sim.getJointVelocities())
        
        gq= sim.robot.gravload(q)
        B = sim.robot.inertia(q)
        C = sim.robot.coriolis(q,dq)

        y = np.linalg.pinv(JA)@(x_desired_ddot+self.Kd@x_tilde_dot+self.Kp@x_tilde-JAdot@dq)

        u = B@y + C@dq + gq

        #print((np.linalg.norm(x_tilde[:3]),np.linalg.norm(x_tilde[3:]))) 
        return u
    

class grav_compensation_controller:
    def __init__(self) -> None:
        pass
   

    def get_u(self, sim):
        '''
        This function sets u based on a gravity compensation controller
            ARGS:
                sim: the simulator object
        '''
        #control signal
        gq= sim.robot.gravload(sim.getJointAngles())
        u=gq
        return u


class Joint_space_PDG_controller:

    def __init__(self, kp, kd):
        self.Kp = np.eye(10)*kp
        self.Kd = np.eye(10)*kd

        '''
        Kp = 150
        Kd = 50
        '''


    def get_u(self, sim):
        '''
        This function is used to set the control signal based on a PD controller in joint space
        ARGS:
            sim: the simulator object
        '''
        # PD controller with gravity compensation

        q = np.array(sim.getJointAngles())
        dq = np.array(sim.getJointVelocities())

        q_tilde = np.array(sim.qref) - q
        dq_tilde = np.array(sim.dqref) - dq

        grav = sim.robot.gravload(q) #gravity compensation

        u = self.Kp*q_tilde + self.Kd*dq_tilde + grav

        return u




class SDD_controller:
    ''' 
    This class uses a gravity compensation inspired controller for the robot, where SDD is Sum of directional derivatives.
    '''

    def __init__(self, k):
        self.k = k


    def get_u(self,sim):
        '''
        This function makes all joints move away for the object:
        note: The function is not multiplied with null space projection.
        ARGS:
            sim: the simulator object
        '''

        q = sim.getJointAngles()
        u = np.zeros((sim.n))

        for ob in sim.obstacles:
            gravs = np.zeros((10, 3))
            o = sim.getObjState(ob)
            for i in range(sim.n):
                pli = sim.getObjState(sim.robot_link_names[i])
                dir = ((o - pli)/np.linalg.norm(o - pli))
                dist = sim.raycastAfterRobotGeometry(pli, dir)
                if dist > 0:
                    gravs[i, :] = dir * sim.repulsion_force_func(dist, 20, 0.5) * self.k
                    gravs[i: 0:2] = 0
            u += rm.dynamicGrav(gravs, q)
            gravs = np.zeros((10, 3))

        dir = np.array([0, 0, -1])
        for i in range(sim.n):
            pli = sim.getObjState(sim.robot_link_names[i])
            dist = sim.raycast(pli, dir)
            if dist > 0:
                gravs[i, :] = dir * sim.repulsion_force_func(dist, 5, 0.5) * self.k
                gravs[i: 0:2] = 0
        u += rm.dynamicGrav(gravs, q)
        u = sim.getNullProjMat(q)@u
        return u
            


# ARCHIVED controllers


class ARCHIVED:

  def repulsion_force_func(x, Fmax_param, d_param, Fd_param): 
    #       │                                
    # F_max-+-....                x: input distance to obstacle                      
    #       │     ....            F_max: force at zero distance                      
    #       │         ..          F_d: force at distance d                       
    #       │           ..                               
    #       │             ..                            
    #       │               ..  <--- f(x)=1/exp(x^2)    (gaussian)                 
    #       │                 ...                       
    # F_d  -+-                   ....                          
    #       │                        .....                     
    #       │                             ........       
    #       └─────────────────────+───────────────      
    #       0                     d                                                                                           
    a = (np.log(1) - np.log(Fd_param/Fmax_param)) / (d_param**2)
    F = Fmax_param / np.exp(a*(x**2))
    return F
        
  def artificial_repulsion_field_controller(self, q):
    u = np.zeros((10)).T


    J = self.robot.jacob0(q)

    null_projection = self.getNullProjMat(q)

    u = J.T@np.array([0, 0, 0, 0, 0, -50]).T


    u_proj = null_projection@u



    #print(u_proj)
    return u_proj
  

  def nullSpacePDControl(self):
    Kp = np.eye(3)*50
    Kd = np.eye(3)*10

    repulsion_target=5
    link_name = self.robot_link_names[repulsion_target]
    
    desired_dist= 1.5

    q=self.getJointAngles()

    J = self.getJacobRevol(repulsion_target,q ) #jacob for 5th joint frame
    JA = J[:3,:] #3x10 

    self.robot.name
    

    dq=np.array(self.getJointVelocities())
    
    
    t_o=self.getObjState(self.obstacle)

    t_elbow=self.getObjState(link_name)

    #T_elbow=np.array(Ts[repulsion_target])
    obj_dist=np.linalg.norm(t_o-t_elbow)

    xtilde_dir=(t_o-t_elbow)/np.linalg.norm(t_o-t_elbow)

    x_tilde=xtilde_dir*(obj_dist-desired_dist)

    # print(obj_dist-desired_dist)

    #print((x_d,t_elbow))
    #print(obj_dist)
    #xtilde_dir=(t_o-t_elbow)/np.linalg.norm(t_o-t_elbow)
    #x_tilde=xtilde_dir*(obj_dist-desired_dist)
    #u += JA.T@Kp@x_tilde - JA.T@Kd@JA@dq

    #null space proj
    nullp=self.getNullProjMat(q)

    u = nullp@u

    # print(np.linalg.norm(u_proj))
    #print((x_tilde,np.linalg.norm(u_proj)))
    return u# u_proj


  def GravCompensationControlLoop(self):
    #control signal
    gq= self.robot.gravload(self.getJointAngles())
    u=gq
    return u


class OP_Space_PDG_Controller:
    '''
    ARCHIVED controller!
    '''
    def __init__(self):
        pass
        

    def get_u(self, sim):
        '''
        This function sets u based on a OP_SPACE_PDG controller

        ARGS:
            sim: the simulator object
        '''
        mode="quat"


        if mode=="zyz":
            dim_analytical=6 
        elif mode=="quat":
            dim_analytical=7

        Kp=np.eye(dim_analytical)
        Kp[:3,:3]=np.eye(int(np.floor(dim_analytical/2)))*2000 #translational gain
        Kp[3:,3:]=np.eye(int(np.ceil(dim_analytical/2)))*800#orientational gain
        Kd=np.eye(dim_analytical)*1

        # get tool orientation quaternion and analytical jacobian

        T_ee=np.array(sim.getObjFrame(sim.tool_name))

        # xtilde
        x_tilde=np.zeros(dim_analytical)
        # relative translation
        x_tilde[:3]=np.array(sim.Tref)[:3,3]-T_ee[:3,3] #translational error
        
        if mode=="quat":
            # relative orientation by quaternions:
            JA,q_ee,JAdot=sim.getAnalyticalJacobian()
            q_ref=UnitQuaternion(sim.Tref)
            #q_rel=q_ee.conj()*q_ref
            x_tilde[3:]=q_ref.vec-q_ee #works with difference, not relative transform

        elif mode=="zyz":
            #relative orientation by zyz euler and petercorke analytical jac
            JA=sim.robot.jacob0_analytical(sim.getJointAngles(),representation="eul")
            #rotation error as zyz euler
            eul_ee=tr2eul(T_ee)

        obj_q = sim.d.body(sim.tool_name).xquat
        q_ee=UnitQuaternion(obj_q).vec   
        x_tilde[3:]=sim.Tref.eul()-r2x(q2r(q_ee),"eul")

        #control signal
        gq= sim.robot.gravload(sim.getJointAngles())

        dq = np.array(sim.getJointVelocities())

        u=gq+JA.T@Kp@x_tilde-JA.T@Kd@JA@dq

        #self.robot.X
        #self.setJointTorques(u)
        #print(gq[:5])
        #print(gq[4:9])
        #print((np.linalg.norm(x_tilde[:3]),np.linalg.norm(x_tilde[3:]))) 
        #print()
        return u
