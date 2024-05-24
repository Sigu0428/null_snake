import numpy as np
import roboticstoolbox as rtb
import spatialmath as sm
from spatialmath import UnitQuaternion,Quaternion
from spatialmath.base import q2r, r2x, rotx, roty, rotz,tr2eul,tr2rt
import robot_matrices as rm
from numpy.linalg import norm, pinv, inv, det
import time
import math

'''
This python file handles all the controllers which will be used by the simulator.
'''

class MaciejewskiEtAl_controller:
    def __init__(self, Kd, Kp): # Kp=np.eye(10)*100, Kd=np.eye(3)
        # quirky trajectory placeholder 
        a = 5
        b = 1.5
        c = 0.9
        self.x = lambda t: np.array([(np.sin(t/a + np.pi) + c)/b, (np.sin(t/a) + c)/b, 0.5])
        self.dx = lambda t: np.array([-np.cos(t/a)/(a*b), np.cos(t/a)/(a*b), 0])
        self.ddx = lambda t: np.array([np.sin(t/a)/(a*a*b), -np.sin(t/a)/(a*a*b), 0])
        self.Kd = Kd
        self.Kp = Kp
    
    def get_u(self, sim):
        Je, _ = sim.getGeometricJacs()
        Je = Je[0:3, :]

        dq = np.array(sim.getJointVelocities())
        q = np.array(sim.getJointAngles())
        t = time.time() - sim.start_time
        
        dxe = self.dx(t) + self.Kd@(self.x(t) - sim.getObjState("ee_link2"))
        
        dxo = None
        d = math.inf
        Jo = None
        for ob in ["blockL01", "blockL02"]:
            o = sim.getObjState(ob)
            for joint in ["wrist_1_link", "shoulder_link2", "forearm_link2", "wrist_2_link2"]:
                pli = sim.getObjState(joint)
                dir = ((o - pli)/np.linalg.norm(o - pli))
                dist = sim.raycastAfterRobotGeometry(pli, dir)
                if dist < 0: dist = math.inf
                if dist < d:
                    d = dist
                    dxo = -dir
                    Jo = sim.getJointJacob(joint)
                    Jo = Jo[0:3, :]
        
        an = lambda d: np.tanh(-10*(d-0.3))
        ao = lambda d: 0.2*np.exp(-10*d)/d

        dtheta = pinv(Je)@dxe + an(d)*pinv(Jo@(np.eye(10) - pinv(Je)@Je))@(ao(d)*dxo - Jo@pinv(Je)@dxe)
        u = sim.robot.gravload(q) + self.Kp@(dtheta - dq)
        return u


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

    def __init__(self, kd_trans=1, kd_ori=1, kp_trans=1,kp_ori=1,lambdaTraj=0.7):
        self.kp_trans = kp_trans
        self.kp_ori = kp_ori
        self.kd_trans = kd_trans
        self.kd_ori = kd_ori
        self.lambdaTraj=lambdaTraj

    def get_u(self, sim):

        dim_analytical=6
        Kp=np.eye(dim_analytical)
        Kp[:3,:3]=np.eye(int(np.floor(dim_analytical/2)))*self.kp_trans #translational gain
        Kp[3:,3:]=np.eye(int(np.ceil(dim_analytical/2)))*self.kp_ori #orientational gain

        Kd=np.eye(dim_analytical)
        Kd[:3,:3]=np.eye(int(np.floor(dim_analytical/2)))*self.kd_trans #translational gain
        Kd[3:,3:]=np.eye(int(np.ceil(dim_analytical/2)))*self.kd_ori #orientational gain
        #get references
        with sim.refLock:
            xref=np.copy(sim.xref)
            dxref=np.copy(sim.dxref)
            ddxref=np.copy(sim.ddxref)
        q=np.array(sim.getJointAngles())
        dq = np.array(sim.getJointVelocities())

        # get tool orientation quaternion and analytical jacobian

        T_ee=np.array(sim.getObjFrame(sim.tool_name))

        Je,Je_dot=sim.getGeometricJacs()

        x_tilde=np.zeros(6)

        x_tilde[:3]=xref[:3]-T_ee[:3,3] #translational error

        #AS IN PAPER  
        R_d=q2r(xref[3:])
        R_e=sm.SO3(T_ee[:3,:3])
        R_ed=np.transpose(np.array(R_e))@np.array(R_d) #(24)
        q_de=UnitQuaternion(R_ed) #"extracted"
  
  
        epsi_e_de=q_de.v
        epsi_de=np.array(R_e)@epsi_e_de #(34)
  
        x_tilde[3:]= epsi_de

        # velocity error 
        x_tilde_dot=np.zeros(dim_analytical) 

        x_dot= Je@dq #dx=JA*dq
        omega_e=x_dot[3:] 


        omega_d=(2*Quaternion(dxref[3:])*UnitQuaternion(xref[3:]).conj()).v
        #omega_d=np.array(R_e).T@omega_d
        #print(np.linalg.norm(omega_d),dxref[3:])
        Delta_omega_e_d=omega_d-omega_e
        x_tilde_dot[3:]=Delta_omega_e_d # (33)
        
        

        #homebrewed acceleration via product rule
        x_desired_ddot=ddxref[:6]
        x_desired_ddot[3:]= 2*(Quaternion(ddxref[3:])*UnitQuaternion(xref[3:]).conj()-(Quaternion(dxref[3:])*UnitQuaternion(xref[3:]).conj())**2).v   #jia (the goat (30))

        #control signal

        B = sim.getM()

        n=sim.getBiasForces()
    
        JE_inv=Je.T@np.linalg.inv(Je@Je.T+(self.lambdaTraj**2)*np.eye(6)) #DLS inverse


        y = JE_inv@(x_desired_ddot+Kd@x_tilde_dot+Kp@x_tilde-Je_dot@dq)

        u = B@y + n

        return u
    
class OP_Space_Velocity_controller:
    '''
    Follows trajectory better, maybe small probability of instability
    '''

    def __init__(self, kd_trans=1, kd_ori=1, kp_trans=1,kp_ori=1,Kv=1,lambdaTraj=0.7,lambdaAvoid=0.05):
        self.kp_trans = kp_trans
        self.kp_ori = kp_ori
        self.kd_ori = kd_ori
        self.kd_trans = kd_trans
        self.kv=Kv
        self.lambdaTraj=lambdaTraj
        self.lambdaAvoid=lambdaAvoid

    def get_u(self, sim):

        dim_analytical=6
        Kp=np.eye(dim_analytical)
        Kp[:3,:3]=np.eye(int(np.floor(dim_analytical/2)))*self.kp_trans #translational gain
        Kp[3:,3:]=np.eye(int(np.ceil(dim_analytical/2)))*self.kp_ori #orientational gain

        Kd=np.eye(dim_analytical)
        Kd[:3,:3]=np.eye(int(np.floor(dim_analytical/2)))*self.kd_trans #translational gain
        Kd[3:,3:]=np.eye(int(np.ceil(dim_analytical/2)))*self.kd_ori #orientational gain
        #get references
        with sim.refLock:
            xref=np.copy(sim.xref)
            dxref=np.copy(sim.dxref)
            ddxref=np.copy(sim.ddxref)
        q=np.array(sim.getJointAngles())
        dq = np.array(sim.getJointVelocities())

        # get tool orientation quaternion and analytical jacobian

        T_ee=np.array(sim.getObjFrame(sim.tool_name))

        Je,Je_dot=sim.getGeometricJacs()

        x_tilde=np.zeros(6)

        x_tilde[:3]=xref[:3]-T_ee[:3,3] #translational error

        #AS IN PAPER  
        R_d=q2r(xref[3:])
        R_e=sm.SO3(T_ee[:3,:3])
        R_ed=np.transpose(np.array(R_e))@np.array(R_d) #(24)
        q_de=UnitQuaternion(R_ed) #"extracted"
        epsi_e_de=q_de.v
        epsi_de=np.array(R_e)@epsi_e_de #(34)

        x_tilde[3:]= epsi_de

        # velocity error 
        x_tilde_dot=np.zeros(dim_analytical) 

        x_dot= Je@dq #dx=JA*dq
        omega_e=x_dot[3:] 

        
        omega_d=(2*Quaternion(dxref[3:])*UnitQuaternion(xref[3:]).conj()).v
        #omega_d=np.array(R_e).T@omega_d
        #print(np.linalg.norm(omega_d),dxref[3:])
        Delta_omega_e_d=omega_d-omega_e
        x_tilde_dot[3:]=Delta_omega_e_d # (33)
        
        

        #homebrewed acceleration via product rule
        x_desired_ddot=ddxref[:6]
        x_desired_ddot[3:]= 2*(Quaternion(ddxref[3:])*UnitQuaternion(xref[3:]).conj()-(Quaternion(dxref[3:])*UnitQuaternion(xref[3:]).conj())**2).v   #jia (the goat (30))



        #control signal

        B = sim.getM()

        n=sim.getBiasForces()
    
        JE_inv=Je.T@np.linalg.inv(Je@Je.T+(self.lambdaTraj**2)*np.eye(6)) #DLS inverse

        q_e_ddot=JE_inv@(x_desired_ddot+Kd@x_tilde_dot+Kp@x_tilde-Je_dot@dq) #

        q_e_dot=dq+q_e_ddot*sim.m.opt.timestep

        #Machenviseskesr7rt controller
        dxo = None
        d = math.inf
        Jo = None
        smallets = None
        links = sim.robot_link_names
        targets = [links[3],links[7],links[5],links[10],"wrist_1_link2", "shoulder_link2", "forearm_link2"]  #[links[3],links[7],links[5],links[10]] 

        for ob in sim.obstacles:
            for joint in targets: #
                o = sim.getObjState(ob)
                pli = sim.getObjState(joint)
                dir = ((o - pli)/np.linalg.norm(o - pli))
                dist = sim.raycastAfterRobotGeometry(pli, dir)
                if dist < 0: dist = math.inf
                if dist < d:
                    d = dist
                    dxo = -dir
                    Jo = sim.getJointJacob(joint)
                    smallest = (ob, joint, d)
        #print(smallest)
        
        for joint in targets: #[ "wrist_1_link", "shoulder_link2", "forearm_link2","wrist_2_link2"]:#["shoulder_link2"]:# "wrist_1_link", "shoulder_link2", "forearm_link2"]:
            pli = sim.getObjState(joint)
            dir = np.array([0,0,-1])
            dist = sim.raycastAfterRobotGeometry(pli, dir)
            if dist < 0: dist = math.inf
            if dist < d:
                d = dist
                dxo = -dir
                Jo = sim.getJointJacob(joint)
        
        thresh=0.15 #0.15, simple: 0.35
        smoothing=10 #10
    
        an = lambda d: (np.tanh(smoothing*(d-thresh))+1)/2
        


        thresh_OA=0.20 #0.15, simple: 0.35
        an_OA = lambda d: (np.tanh(-smoothing*(d-thresh_OA))+1)/2
        ao = lambda d:  0.3/d

        Jo_temp = Jo
        Je_temp = Je
        Je = Jo_temp
        Jo = Je_temp


        dampen=self.lambdaAvoid
        Je_inv=Je.T@np.linalg.inv(Je@Je.T+(dampen**2)*np.eye(6))

        Jo_Je_inv_Je = Jo@(np.eye(10) - Je_inv@Je)
        JoJe_inv=Jo_Je_inv_Je.T@np.linalg.inv(Jo_Je_inv_Je@Jo_Je_inv_Je.T+(dampen**2)*np.eye(6))

        #ao_capped = ao(d)
        #dtheta = q_e_dot+ an(d)*JoJe_inv@(ao(d)*dxo - Jo@q_e_dot)
        
        # Add zeros in orientation part

        print("an: ", an(d), "ao: ", ao(d))

        dxo_temp = np.zeros(6)
        dxo_temp[:3] = dxo

        dxe = dxo_temp
        dxo = Je@q_e_dot

        dtheta = an_OA(d)*ao(d)*Je_inv@dxe + an(d)*JoJe_inv@(dxo - an_OA(d)*ao(d)*Jo@Je_inv@dxe)
        #dtheta = q_e_dot+ an(d)*np.linalg.pinv((Jo@(np.eye(10) - np.linalg.pinv(Je)@Je)))@(ao(d)*dxo - Jo@q_e_dot)
        u = np.eye(10)*self.kv@(dtheta - dq) + n

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
        gq= sim.getBiasForces()
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

    def __init__(self, k, k_ground, threshold):
        self.k = k
        self.thresh = threshold
        self.k_g = k_ground

    def get_u(self,sim):
        '''
        This function makes all joints move away for the object:
        note: The function is not multiplied with null space projection.
        ARGS:
            sim: the simulator object
        '''

        q = sim.getJointAngles()
        u = np.zeros((sim.n))

        smallest_dist = math.inf
        smallest = None

        gravs = np.zeros((10, 3))
        for ob in sim.obstacles:
            o = sim.getObjState(ob)
            for i in [3, 5, 7, 9]:
                pli = sim.getObjState(sim.robot_link_names[i])
                dir = ((o - pli)/np.linalg.norm(o - pli))
                dist = sim.raycastAfterRobotGeometry(pli, dir)
                if dist < smallest_dist:
                    smallest_dist = dist
                    smallest = (ob,o, sim.robot_link_names[i], pli, dist, dir)
                gravs[i, :] += dir * sim.repulsion_force_func(dist, self.k, self.thresh)

        dir = np.array([0, 0, -1])
        for i in range(sim.n):
            pli = sim.getObjState(sim.robot_link_names[i])
            dist = sim.raycastAfterRobotGeometry(pli, dir)
            gravs[i, :] += dir * sim.repulsion_force_func(dist, self.k_g, self.thresh) + np.random.randn(3)

        #print(smallest)
        u = rm.dynamicGrav(gravs, q)
        u = sim.getNullProjMat(q)@u
        #print(norm(u))
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
