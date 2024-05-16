
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
from pprint import pprint
import pickle
from plot_robot_log import robot_plot
import Controllers
from mujoco import _functions
from ctypes import c_int, addressof

class simulation:

  def __init__(self):
    '''
    This initizalizes the simulation object. 
    The initialization consist of setting the following parameters:
    - The robot model based on mujoco
      - self.m: the mujoco model
      - self.d: the mujoco data
      - self.jointTorques: the joint torques to be applied to the robot
      - self.dt: the time step of the simulation 
      - self.robot_link_names: the names of the robot links
      - self.q0: the initial joint angles
      - self.dq0: the initial joint velocities
      

    - The robot model based on peter corkes toolbox

    
    - The shared data for the control thread
      - self.control_enabled: a flag to enable the control loop
      - self.qref: the reference joint angles
      - self.dqref: the reference joint velocities
      - self.tool_name: the name of the tool used in the simulation
      - self.Tref: the reference end effector position
      - self.obstacle: the name of the obstacle in the simulation
    
      
    - The logging of data
      - self.log_data_enabled: a flag to enable logging of data
      - self.ee_desired_data: the list of desired end effector positions
      - self.ee_position_data: the list of actual end effector positions
      - self.robot_data_plot: the object to plot the data
    
      
    - The controllers used in the simulation
      - self.controllers: the list of all controllers for the main task
      - self.nullspace_controllers: the list of all controllers for the null space task
    '''



    self.m = mujoco.MjModel.from_xml_path('./Ur5_robot/Robot_scene.xml')
    self.d = mujoco.MjData(self.m)
    self.jointTorques = [0 ,0,0,0,0,0,0,0,0,0] #simulation reads these and sends to motors at every time step
    self.dt = 1/40 #control loop update rate
    self.robot_link_names = ["shoulder_link", "upper_arm_link", "forearm_link", "wrist_1_link", "ee_link1", "shoulder_link2", "upper_arm_link2", "forearm_link2", "wrist_1_link2", "wrist_2_link2", "wrist_3_link2", "ee_link2"]
    self.q0=  [0 , -np.pi/2.4, np.pi/2.4, -np.pi/2.2, np.pi,-np.pi/1.7,np.pi/1.7 , np.pi/2, -np.pi/2,0]  # 0, -3*np.pi/4, np.pi/3, np.pi, 0, 0, np.pi/3 , 0, 0,0] #home pose
    self.dq0= [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    self.mojo_internal_mutex = Lock()
    self.refLock = Lock()


    # Peter corke robot model initialization
    self.initialize_robot()
  


    #shared data for control thread
    self.control_enabled=1
    self.qref=self.q0
    self.dqref=self.dq0
    self.tool_name="ee_link2"
    self.Tref=self.robot.fkine(self.qref)
    self.obstacle="blockL01"
    self.obstacles = ['blockL01', 'blockL02']
    self.refmutex=1
    self.xref=np.zeros(7)#xyz wv1v2v3
    self.dxref=np.zeros(7)#xyz wv1v2v3 
    self.ddxref=np.zeros(7) #xyz wv1v2v3 
    self.Mpty=np.zeros((self.m.nv, self.m.nv))
    self.DLS_lambda=0.5


    # logging of data for plotting:
    self.log_data_enabled = 1
    self.ee_desired_data = []
    self.ee_position_data = []
    self.log_u = []
    self.latest_u = np.zeros((10)).T

    # The list of controllers used
    self.controllers = [] # List of all controllers to be used in the simulation


  def initialize_robot(self):
    '''
    This function initializes the peter corke robot model, using DH parameters
    To setup the DH parameters then the following variables are set.
    - robot_base: the base of the robot

    - pl1-pl10: the position of the link center of masses

    - i1-i10: the inertia of the links

    - m1-m10: the mass of the links
    '''

    # Universal Robot UR5e kiematics parameters
    robot_base = sm.SE3.Trans(0,0,0)


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


    i1 = np.array([[0.0, 0., 0.],    [0., 0.0, 0.],   [0., 0., 0.00000001]])
    i2 = np.array([[0.0, 0., 0.],    [0., 0.0, 0.],   [0., 0., 0.00000001]])
    i3 = np.array([[0.0, 0., 0.],    [0., 0.0, 0.],   [0., 0., 0.00000001]])
    i4 = np.array([[0.0, 0.0, 0.0],    [0.0, 0.0, 0.0],   [0.0, 0.0, 0.00000001]])

    i5 = np.array([[0.0, 0., 0.],    [0., 0.0, 0.],   [0., 0., 0.00000001]])
    i6 = np.array([[0.0, 0., 0.],    [0., 0.0, 0.],   [0., 0., 0.00000001]])
    i7 = np.array([[0.0, 0., 0.],    [0., 0.0, 0.],   [0., 0., 0.00000001]])
    i8 = np.array([[0.0, 0.0, 0.0],    [0.0, 0.0, 0.0],   [0.0, 0.0, 0.00000001]])
    i9 = np.array([[0.0, 0.0, 0.0],    [0.0, 0.0, 0.0],   [0.0, 0.0, 0.00000001]])
    i10 = np.array([[0.0, 0.0, 0.0],    [0.0, 0.0, 0.0],   [0.0, 0.0, 0.00000001]])

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



  def getBiasForces(self):
    qf=[]
    for i in range(0, self.n):
      qf.append(float(self.d.joint(f"joint{i+1}").qfrc_bias)) 
    return qf


  def launch_mujoco(self):
    '''
    This function launches mujoco without the control loop and the logging of data
    '''
    with mujoco.viewer.launch_passive(self.m, self.d) as viewer:
      # Close the viewer automatically after 30 wall-seconds.

      #initialize joint values to home before running sim
      for i in range(0, self.n):
        self.d.joint(f"joint{i+1}").qpos=self.q0[i]

      

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
    '''
    This function launches mujoco together with the control loop and the logging of data
    '''


    with mujoco.viewer.launch_passive(self.m, self.d) as viewer:
      # Close the viewer automatically after 30 wall-seconds.

      #initialize joint values to home before running sim
      for i in range(0, self.n):

          self.d.joint(f"joint{i+1}").qpos=self.q0[i]


      #control_thrd = Thread(target=self.control_loop,daemon=True) #control loop for commanding torques
      #control_thrd.start()


      log_data_thrd = Thread(target=self.data_log_loop,daemon=True) #control loop for commanding torques
      log_data_thrd.start()


      self.start_time = time.time()
        
      T_ref=self.robot.fkine(self.getJointAngles())
      q_ref=UnitQuaternion(T_ref)
      T_ref=np.array(T_ref)
      self.xref[:3]=T_ref[:3,3]
      self.xref[3:]=q_ref.vec
      while viewer.is_running(): #simulation loop !
        step_start = time.time()
        
        self.mojo_internal_mutex.acquire()
        mujoco.mj_step1(self.m, self.d)
        self.mojo_internal_mutex.release()

        self.control_loop()

        #joint torque application loop
        with self.jointLock: #moved to before mjstep to fix snap
          if self.sendPositions: #assume this is mutex
            for i in range(0, self.n):
              self.d.actuator(f"actuator{i+1}").ctrl = self.jointTorques[i]
            self.sendPositions = False
    
        # mj_step can be replaced with code that also evaluates
        # a policy and applies a control signal before stepping the physics.
        self.mojo_internal_mutex.acquire()
        mujoco.mj_step2(self.m, self.d)
        self.mojo_internal_mutex.release()

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
  


  def getM(self):
    L = mujoco.mj_fullM(self.m,self.Mpty , self.d.qM)
    return np.copy(self.Mpty[-10:,-10:]) #CHANGES IF DIFFERENT BODIES ADDED

  def getGeometricJac(self):
      self.mojo_internal_mutex.acquire()
      
      jac=np.zeros((6,self.m.nv))
      id=self.m.body("ee_link2").id
      mujoco.mj_jacBody(self.m, self.d, jac[:3], jac[3:], id)
      Je=jac[:,-10:] #CHANGES IF DIFFERENT BODIES ADDED
      self.mojo_internal_mutex.release()
      return Je

  def getAllJacs(self):

      h=1e-8
      #get geometric jacobian from mujoco
      jac=np.zeros((6,self.m.nv))
      id=self.m.body("ee_link2").id
      mujoco.mj_jacBody(self.m, self.d, jac[:3], jac[3:], id)
      Je=jac[:,-10:] #CHANGES IF DIFFERENT BODIES ADDED

      #get TA
      obj_q = self.d.body(self.tool_name).xquat
      q_ee=UnitQuaternion(obj_q).vec #s,v1,v2,v3

      #analytical jac transform
      xi0=q_ee[0]; xi1=q_ee[1];xi2=q_ee[2];xi3=q_ee[3] #xi0 = s, ...

      H=np.array([[-xi1,xi0,-xi3,xi2],
                  [-xi2,xi3,xi0,-xi1],
                  [-xi3,-xi2,xi1,xi0]])
    
    
      TA_inv_pre=np.zeros((7,6)) #maybe we have to use np.inv here 
      TA_inv_pre[3:,3:]=0.5*H.T
      TA_inv_pre[:3,:3]=np.eye(3)

      #integrate joint angles for small timestep h
      q=np.copy(self.d.qpos)
      dq=np.copy(self.d.qvel)
      q_init=np.copy(q)
      mujoco.mj_integratePos(self.m, q, dq, h)
      
      #update qpos with small step
      self.d.qpos=q

      #update internal model (kinematics etc) with new vals
      mujoco.mj_kinematics(self.m,self.d)
      mujoco.mj_comPos(self.m,self.d)

      #get next TA
      obj_q = self.d.body(self.tool_name).xquat
      q_ee=UnitQuaternion(obj_q).vec #s,v1,v2,v3

      #analytical jac transform
      xi0=q_ee[0]; xi1=q_ee[1];xi2=q_ee[2];xi3=q_ee[3] #xi0 = s, ...

      H=np.array([[-xi1,xi0,-xi3,xi2],
                  [-xi2,xi3,xi0,-xi1],
                  [-xi3,-xi2,xi1,xi0]])
    
    
      TA_inv_post=np.zeros((7,6)) #maybe we have to use np.inv here 
      TA_inv_post[3:,3:]=0.5*H.T
      TA_inv_post[:3,:3]=np.eye(3)

      #get next jacobian
      jach=np.zeros((6,self.m.nv))
      mujoco.mj_jacBody(self.m, self.d, jach[:3], jach[3:], id)
      Jeh=jach[:,-10:] #CHANGES IF DIFFERENT BODIES ADDEDs

      #finite differences
      Je_dot=(Jeh-Je)/h #why does this shit work
      TA_inv_dot=(TA_inv_post-TA_inv_pre)/h

      #reset q back to beginning
      self.d.qpos=q_init
      #update internal model (kinematics etc) with new vals
      mujoco.mj_kinematics(self.m,self.d)
      mujoco.mj_comPos(self.m,self.d)

      JA=TA_inv_pre@Je
      JA_dot = TA_inv_pre@Je_dot+TA_inv_dot@Je #product rule!
      return Je,Je_dot,JA,JA_dot


  
  def control_loop(self, debug=False):
    '''
    This function sets the torque values u based on all appended main and secondary task controllers.
    For each of the controller tasks, it is expected that it has the function get_u, which returns the control signal.
    '''


    time_elapsed_list = []

    start_time = time.time()

    if self.control_enabled:


      u = np.zeros(len(self.q0))
      for controller in self.controllers:
        u += controller.get_u(self)


      '''
      for controller in self.nullspace_controllers:
        u += self.getNullProjMat(self.getJointAngles())@controller.get_u(self)   
      '''

      self.setJointTorques(u)



    time_elapsed = time.time() - start_time
    sleep_adjust_time = max(0, self.dt - time_elapsed) # Adjusted for time the control loop takes

    
    
    time_elapsed = time.time() - start_time
    time_elapsed_list.append(time_elapsed)

    #if len(time_elapsed_list) % 1000 > 0:
     # print(f"Average time per 1000 steps: {np.mean(np.asarray(time_elapsed_list))}  --- adjusted sleep time {sleep_adjust_time}")

      


  def data_log_loop(self):
    '''
    This function logs the data for the robot positions and the desired robot positions.
    The data is logged into the object self.robot_data_plot.
    '''

    time_start = time.time()
    while True:
      time_elapsed = time.time() - time_start
      time.sleep(self.dt)

      if self.log_data_enabled:
        self.log_robot_positions()
        self.log_desired_position()
        self.log_u.append(self.latest_u)

      if time_elapsed > 1:
        self.save_data()
        time_start = time.time()

  def log_robot_positions(self):
    '''
    This function is for logging the robot positions for both the end effector and the null space controller.
    The position are used to calculate the velocities of the given instances. The values are logged in the following variables:
    self.ee_position_data = []
    self.null_position_data = []
    ''' 
    # get tool orientation quaternion and analytical jacobian

    T_ee=np.array(self.getObjFrame(self.tool_name))
    x_ee=np.zeros(7)
    x_ee[:3]=T_ee[:3,3]
    x_ee[3:]=UnitQuaternion(sm.SE3(T_ee)).vec
    self.ee_position_data.append(x_ee)



  def log_desired_position(self):
    '''
    This function logs the desired position of the end effector in the world frame
    given as sim.Tref
    '''
    # get tool orientation quaternion and analytical jacobian

    self.ee_desired_data.append(self.xref)




  def save_data(self):
    '''
    This function will be used to save the data from the robot. given from the log loop
    '''
    with open('robot_end_effector_position.txt', 'wb') as f:
      pickle.dump(self.ee_position_data, f)
    
    with open('robot_end_effector_position_desired.txt', 'wb') as f:
      pickle.dump(self.ee_desired_data, f)
    
    with open('robot_joint_torques.txt', 'wb') as f:
      pickle.dump(self.log_u, f)



  def get_trans(self, T_ee):
    '''
    This function takes a homogenous transformation matrix and returns the translational part.
    ''' 
    return np.array(T_ee)[:3,3]

    

  def distToCubeSurface(self, pos, cube_frame):
    # position in world frame and cube frame in world frame, both numpy arrays
    # transform position to cubeframe, such that cube is effectively "axis-aligned"
    pos = np.linalg.inv(cube_frame)@np.concatenate((pos, np.array([1])))
    pos = pos/pos[3]
    # then follow this approach for distance to axis aligned cubes:
    # https://math.stackexchange.com/questions/2133217/minimal-distance-to-a-cube-in-2d-and-3d-from-a-point-lying-outside
    dist = np.sqrt(max(0, abs(pos[0])-1)**2 + max(0, abs(pos[1])-1)**2 + max(0, abs(pos[2])-1)**2)
    return dist




  def getNullProjMat(self, q): # dynamic projection matrix N, such that tau = tau_main + N@tau_second
    #Je = self.robot.jacob0(q)
    Je = self.getGeometricJac()
    Je_inv = np.linalg.pinv(Je)
    M = self.robot.inertia(q)

    N = (np.eye(self.n) - Je_inv@Je) # from book


    return N


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
  
  def getAnalyticalJacobian(self,Je,Jedot):

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
    q_ee=UnitQuaternion(obj_q).vec #s,v1,v2,v3

    #analytical jac transform
    xi0=q_ee[0]; xi1=q_ee[1];xi2=q_ee[2];xi3=q_ee[3] #xi0 = s, ...

    H=np.array([[-xi1,xi0,-xi3,xi2],
                [-xi2,xi3,xi0,-xi1],
                [-xi3,-xi2,xi1,xi0]])
    
    
    TA_inv=np.zeros((7,6)) #maybe we have to use np.inv here 
    TA_inv[3:,3:]=0.5*H.T
    TA_inv[:3,:3]=np.eye(3)


    
    #get ee jacobian in world frame (error is defined in world frame)
    q=self.getJointAngles()
    dq=self.getJointVelocities()

    #transform to analytical
    Ja = TA_inv@Je

    #time derivative - get ee velocity as quat
    dx=Ja@dq

    #analytical jac transform
    xi0=dx[3]; xi1=dx[4];xi2=dx[5];xi3=dx[6] #xi0 = s, ...

    Hd=np.array([[-xi1,xi0,-xi3,xi2],
                [-xi2,xi3,xi0,-xi1],
                [-xi3,-xi2,xi1,xi0]])
    
    
    TA_d_inv=np.zeros((7,6)) #maybe we have to use np.inv here 
    TA_d_inv[3:,3:]=0.5*Hd.T
    TA_d_inv[:3,:3]=np.eye(3)


    Jadot = TA_inv@Jedot+TA_d_inv@Je #product rule!
    return Ja,q_ee, Jadot
  


  def get_trans_position(self, T_ee):
    '''
    This function is for logging the robot positions for both the end effector and the null space controller.
    The position are used to calculate the velocities of the given instances. The values are logged in the following variables:
    self.ee_position_data = []
    self.null_position_data = []
    ''' 
    # get tool orientation quaternion and analytical jacobian

    ee_position = np.zeros((3,4))
    # relative translation
    ee_position[:3,3]=np.array(T_ee)[:3,3] #translational error
    
    # relative orientation by quaternions:
    q_ref=UnitQuaternion(T_ee)
    #q_rel=q_ee.conj()*q_ref
    ee_position[3:]=q_ref.vec #works with difference, not relative transform

  

  def start(self):
    #launch simulation thread
    self.jointLock = Lock()
    self.sendPositions = False


    mujoco_thrd = Thread(target=self.launch_mujoco_with_control, daemon=True)


    
    mujoco_thrd.start()
    

  def raycast(self, pos, dir, mask = np.array([1, 1, 0, 0, 0, 0]).astype(np.uint8)):
    # source xd: https://github.com/openai/mujoco-worldgen/blob/master/mujoco_worldgen/util/geometry.py
    # and https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html
    c_arr = (c_int*1)(0)
    dist = _functions.mj_ray(
      self.m, #mujoco model
      self.d, #mujoco data
      pos, # starting point of ray np array(3, 1)
      dir, # direction to cast ray np array(3, 1)
      mask, #falgs for enabling collisions with geom groups 0=ground, 1=obstacles, 2=robot vizualization geom, 3=robot collision geom
      1, #flag that enables collision for static geometry
      -1, # id of body to exclude. -1 to include all bodies
      np.array([c_arr]) # output array for id of geometry the ray collided with
    ) 
    #collision_geom = c_arr[0] if c_arr[0] != -1 else None
    return dist
  
  def raycastAfterRobotGeometry(self, pos, dir):
    dist = self.raycast(pos, dir, mask = np.array([0, 0, 1, 0, 0, 0]).astype(np.uint8))
    if dist < 0:
      return dist
    dist = self.raycast(pos + dir*dist, dir, mask = np.array([1, 1, 0, 0, 0, 0]).astype(np.uint8))
    return dist

  def repulsion_force_func(self, x, magnetude, decayrate):
    return (magnetude/x)*np.exp(-decayrate*x)