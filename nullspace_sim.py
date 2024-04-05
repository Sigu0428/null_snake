
import time
from threading import Thread, Lock
import mujoco
import mujoco.viewer
import numpy as np
import roboticstoolbox as rtb
import time
import spatialmath as sm
import numpy as np
import matplotlib.pyplot as plt
from spatialmath import UnitQuaternion
from spatialmath.base import q2r, r2x, rotx, roty, rotz,tr2eul,tr2rt
from scipy.spatial.transform import  Rotation

class simulation:

  def __init__(self):
    self.m = mujoco.MjModel.from_xml_path('./Ur5_robot/Robot_scene.xml')
    self.d = mujoco.MjData(self.m)
    self.jointTorques = [0 ,0,0,0,0,0,0,0,0,0] #simulation reads these and sends to motors at every time step
    self.dt = 1/100 #control loop update rate
  
    # Universal Robot UR5e kiematics parameters
    tool_matrix = sm.SE3.Trans(0., 0., 0.18) #adds tool offset to fkine automatically!!
    robot_base = sm.SE3.Trans(0,0,0)

    self.q0=[0 , -np.pi/4, 0, 0, 0,np.pi,0 , 0, 0,0] #home pose
    self.q00=[0 , 0, 0, 0, 0,0,0, 0, 0,0]


    # UR5e kinematics parameters
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



    self.robot = rtb.DHRobot( 
        [ 
            #first UR
            rtb.RevoluteDH(d=0.1625, alpha = np.pi/2,offset=np.pi, m=m1, r=pl1, I=i1), #theta=np.pi either here or in XML using ref=""
            rtb.RevoluteDH(a=-0.425, m=m2, r=pl2, I=i2),
            rtb.RevoluteDH(a=-0.3922, m=m3, r=pl3, I=i3),
            rtb.RevoluteDH(d=0.1333, alpha=np.pi/2, m=m4, r=pl4, I=i4),
            #second UR
            rtb.RevoluteDH(d=0.1625, alpha = np.pi/2, m=m5,offset=np.pi, r=pl5, I=i5), #theta=pi
            rtb.RevoluteDH(a=-0.425, m=m6, r=pl6, I=i6),
            rtb.RevoluteDH(a=-0.3922, m=m7, r=pl7, I=i7),
            rtb.RevoluteDH(d=0.1333, alpha=np.pi/2, m=m8, r=pl8, I=i8),
            rtb.RevoluteDH(d=0.0997, alpha=-np.pi/2, m=m9, r=pl9, I=i9),
            rtb.RevoluteDH(d=0.0996, m=m10, r=pl10, I=i10)
        ], name="UR5e",
        base=robot_base,
        #add tool frame if used: tool=tool_matrix

        )
        
    self.n=self.robot.n
    self.q0=self.q0[:self.n]
    self.T_EE_TCP=sm.SE3.Trans(0.0823,0,0)

    #shared data for control thread
    self.control_enabled=1
    self.qref=self.q0
    self.dqref=self.q00
    self.tool_name="ee_link2"
    self.xref=np.zeros(6)
    self.Tref=self.robot.fkine(self.qref)

  def launch_mujoco(self):
    with mujoco.viewer.launch_passive(self.m, self.d) as viewer:
      # Close the viewer automatically after 30 wall-seconds.

      #initialize joint values to home before running sim
      for i in range(0, self.n):
        self.d.joint(f"joint{i+1}").qpos=self.q0[i]

      

      start = time.time()
      while viewer.is_running(): #simulation loop !
        step_start = time.time()
        

        #joint torque application loop
        with self.jointLock: #moved to before mjstep to fix snap
          if self.sendPositions: #assume this is mutex
            for i in range(0, self.n):
              self.d.actuator(f"actuator{i+1}").ctrl = self.jointTorques[i]
            self.sendPositions = False
    
        # mj_step can be replaced with code that also evaluates
        # a policy and applies a control signal before stepping the physics.
        mujoco.mj_step(self.m, self.d)


        viewer.sync()

        # Rudimentary time keeping, will drift relative to wall clock.
        time_until_next_step = self.m.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
          time.sleep(time_until_next_step)

  def launch_mujoco_with_control(self):
    with mujoco.viewer.launch_passive(self.m, self.d) as viewer:
      # Close the viewer automatically after 30 wall-seconds.

      #initialize joint values to home before running sim
      for i in range(0, self.n):
        if i==8:
           self.d.joint(f"joint{i+1}").qpos=self.q0[i]#+1.2 uncomment this to add a small offset to verify  rotation works
        else:
          self.d.joint(f"joint{i+1}").qpos=self.q0[i]

      control_thrd = Thread(target=self.control_loop,daemon=True) #control loop for commanding torques
      control_thrd.start()

      start = time.time()
      while viewer.is_running(): #simulation loop !
        step_start = time.time()
        

        #joint torque application loop
        with self.jointLock: #moved to before mjstep to fix snap
          if self.sendPositions: #assume this is mutex
            for i in range(0, self.n):
              self.d.actuator(f"actuator{i+1}").ctrl = self.jointTorques[i]
            self.sendPositions = False
    
        # mj_step can be replaced with code that also evaluates
        # a policy and applies a control signal before stepping the physics.
        mujoco.mj_step(self.m, self.d)


        viewer.sync()

        # Rudimentary time keeping, will drift relative to wall clock.
        time_until_next_step = self.m.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
          time.sleep(time_until_next_step)
    
  def setJointTorques(self,torques): #set joint torque vector which is applied to simulation from next time step
    with self.jointLock:
      for i in range(0, self.n):
        self.jointTorques[i] = torques[i]
      self.sendPositions = True
  
  def control_loop(self, debug=False):

    while True:
      time.sleep(self.dt)
      if self.control_enabled:
        #CONTROLLER GOES HERE!
        self.opSpacePDGControlLoop()

        
  def jointSpacePDGControlLoop(self):
    # PD controller with gravity compensation
    Kp = 1500
    Kd = 50

    q = np.array(self.getJointAngles())
    dq = np.array(self.getJointVelocities())

    q_tilde = np.array(self.qref) - q
    dq_tilde = np.array(self.dqref) - dq

    grav = self.robot.gravload(q) #gravity compensation

    u = Kp*q_tilde + Kd*dq_tilde + grav

    self.setJointTorques(u)

        
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
    J = self.jacob_revol(9, q)
    u = J.T@np.array([5000, 5000, 5000, 0, 0, 0]).T
    u_proj = self.getNullProjMat(q)@u
    return u_proj

  def getJacobRevol(self, j, q): # compute jacobian for arbitrary joint j in configuration q (only works for revolute joints)
    T_0_l = self.robot.A(j, q).A
    rj = self.robot[j].r[..., np.newaxis]
    rj = np.row_stack((rj, 1))
    p_0_lj = T_0_l@rj
    p_0_lj = p_0_lj[0:3] / p_0_lj[3]
    
    Jp = np.zeros((3, self.n))
    Jo = np.zeros((3, self.n))
    for i in range(j):
      T_0_i = self.robot.A(i, q).A
      zi = T_0_i[0:3, 3]
      ri = self.robot[j].r[..., np.newaxis]
      ri = np.row_stack((ri, 1))
      p_0_li = T_0_i@ri
      p_0_li = p_0_li[0:3] / p_0_li[3]
      Jo[:, i] = zi
      Jp[:, i] = np.cross(zi, np.squeeze(p_0_lj - p_0_li))
    return np.row_stack((Jo, Jp))

  def getNullProjMat(self, q): # dynamic projection matrix N, such that tau = tau_main + N@tau_second
    Je = self.robot.jacobe(q)
    Je_inv = np.linalg.pinv(Je)
    M = self.robot.inertia(q) # need to calculate this
    N = M@(np.eye(self.n) - Je_inv@Je)@np.linalg.inv(M)
    return N




  def opSpacePDGControlLoop(self):
    mode="quat"


    if mode=="zyz":
      dim_analytical=6 
    elif mode=="quat":
      dim_analytical=7

    Kp=np.eye(dim_analytical)
    Kp[:3,:3]=np.eye(int(np.floor(dim_analytical/2)))*2000 #translational gain
    Kp[3:,3:]=np.eye(int(np.ceil(dim_analytical/2)))*1000#orientational gain
    Kd=np.eye(dim_analytical)*1

    # get tool orientation quaternion and analytical jacobian

    T_ee=np.array(self.getObjFrame(self.tool_name))

    # xtilde
    x_tilde=np.zeros(dim_analytical)
    # relative translation
    x_tilde[:3]=np.array(self.Tref)[:3,3]-T_ee[:3,3] #translational error
    
    if mode=="quat":
      # relative orientation by quaternions:
      JA,q_ee=self.getAnalyticalJacobian()
      q_ref=UnitQuaternion(self.Tref)
      #q_rel=q_ee.conj()*q_ref
      x_tilde[3:]=q_ref.vec-q_ee #works with difference, not relative transform

    elif mode=="zyz":
      #relative orientation by zyz euler and petercorke analytical jac
      JA=self.robot.jacob0_analytical(self.getJointAngles(),representation="eul")
      #rotation error as zyz euler
      eul_ee=tr2eul(T_ee)

      obj_q = self.d.body(self.tool_name).xquat
      q_ee=UnitQuaternion(obj_q).vec   
      x_tilde[3:]=self.Tref.eul()-r2x(q2r(q_ee),"eul")

    #control signal
    gq= self.robot.gravload(self.getJointAngles())

    dq = np.array(self.getJointVelocities())

    u=gq+JA.T@Kp@x_tilde-JA.T@Kd@JA@dq

    #self.robot.X
    self.setJointTorques(u)
    #print(gq[:5])
    #print(gq[4:9])
    print((np.linalg.norm(x_tilde[:3]),np.linalg.norm(x_tilde[3:]))) 
    #print()
   

  def getJointAngles(self):
    ## State of the simulater robot 
    qState=[]    
    for i in range(0, self.n):
      qState.append(float(self.d.joint(f"joint{i+1}").qpos[0])) 
      
    return qState
  
  def getJointVelocities(self):
    ## State of the simulater robot 
    qState=[]    
    for i in range(0, self.n):
      qState.append(float(self.d.joint(f"joint{i+1}").qvel[0])) 
      
    return qState

  
  def getObjState(self, name):
    ## State of the simulater robot    
    Objstate = self.d.body(name).xpos
    return Objstate
  
  def getObjFrame(self, name):  
    obj_t = self.d.body(name).xpos
    obj_q = self.d.body(name).xquat
    R=q2r(UnitQuaternion(obj_q).vec)
    T=np.eye(4)
    T[:3,:3]=R
    T[:3,3]=obj_t
    return sm.SE3(T)
  
  def getObjDistance(self, name1,name2):
    ## State of the simulater robot    
    dist=np.linalg.norm(self.getObjState(name2)-self.getObjState(name1))
    return dist
  
  def getAnalyticalJacobian(self):

    #analytical jac transform for zyz euler angles, petercorke equivalent
    #T_ee=sim.robot.fkine(sim.q0)
    #eul_ee=tr2eul(np.array(T_ee))
    #z1=eul_ee[0]; y=eul_ee[1]; z2=eul_ee[2]
    
    #Einv=np.array([[(-np.cos(y)*np.cos(z1))/np.sin(y),(-np.cos(y)*np.sin(z1))/np.sin(y),1],
    #              [-np.sin(z1),np.cos(z1),0],
    #              [np.cos(z1)/np.sin(y),np.sin(z1)/np.sin(y),0]])
                    

    #TAinv=np.eye(6) #maps from geometric
    #TAinv[3:,3:]=Einv
    
    #JA=TAinv@sim.robot.jacob0(sim.q0)

    #get ee frame orientation as quaternion
    obj_q = self.d.body(self.tool_name).xquat
    q_ee=UnitQuaternion(obj_q).vec

    #analytical jac transform
    xi0=q_ee[0]; xi1=q_ee[1];xi2=q_ee[2];xi3=q_ee[3]

    H=np.array([[-xi1,xi0,-xi3,xi2],
                [-xi2,xi3,xi0,-xi1],
                [-xi3,-xi2,xi1,xi0]])
    
    TA_inv=np.zeros((7,6)) #maybe we have to use np.inv here 
    TA_inv[3:,3:]=0.5*H.T
    TA_inv[:3,:3]=np.eye(3)

    #get ee jacobian in world frame (error is defined in world frame)
    Je = self.robot.jacob0(self.getJointAngles())
    #transform to analytical
    Ja = TA_inv@Je
    return Ja,q_ee
  

  def start(self):
    #launch simulation thread
    self.jointLock = Lock()
    self.sendPositions = False
    mujoco_thrd = Thread(target=self.launch_mujoco_with_control, daemon=True)
    mujoco_thrd.start()
    #control_thrd = Thread(target=self.control_loop,daemon=True) #control loop for commanding torques
    #control_thrd.start()



if __name__ == "__main__":
  sim=simulation()
  sim.start() 




    #get ee frame orientation as quaternion
  
  


  while True:
    #print(sim.robot.fkine(sim.getJointAngles()))
    #print(sim.getObjFrame("ee_link2"))
    #print("-----------------------")
    time.sleep(2)
  
    


  
  #check fkine vs mujoco EE frames
  #print(sim.robot.fkine(sim.getJointAngles()))
  
  #print(sim.getObjFrame("ee_link2"))



  while True: pass

  #kinematic trajectory example--------------------
  #target="blockL06"
  #print("target frame")
  #print(self.getObjFrame(target))

  # conctruct the cartesian trajectory from current pose to block
  #print("current EE")
  #T_world_EE=self.robot.fkine(self.getJointAngles())
  #print(T_world_EE)    

  #time.sleep(5)
  #T_goal=self.getObjFrame(target)
  #print("target")
  #T_goal=T_goal*self.T_EE_TCP.inv() #account for tool (can be done in fkine instead if toolframe is added)
  #print(T_goal)
  #create interpolated trajectory with 100 steps
  #Trj=rtb.ctraj(T_world_EE,T_goal,100)

  #qik = self.robot.ikine_LM(Trj, q0=self.getJointAngles()) #get joint trajectory
  
  #for i in qik.q:
  #  self.sendJoint(i)
  #  time.sleep(0.1)


  #print("final EE frame")
  #print(self.robot.fkine(self.getJointAngles()))
  #kinematic trajectory example--------------------
 