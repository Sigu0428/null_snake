
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
from spatialmath.base import q2r, r2x, rotx, roty, rotz

class simulation:

  def __init__(self):
    self.m = mujoco.MjModel.from_xml_path('./Ur5_robot/Robot_scene.xml')
    self.d = mujoco.MjData(self.m)
    self.jointTorques = [0 , 0,0,0,0,0,0,0,0,0] #simulation reads these and sends to motors at every time step
    self.dt = 1/100 #control loop update rate
  
    # Universal Robot UR5e kiematics parameters
    tool_matrix = sm.SE3.Trans(0., 0., 0.18) #adds tool offset to fkine automatically!!
    robot_base = sm.SE3.Trans(0,0,0)

    self.q0=[0 , -np.pi/2, -np.pi/2, -np.pi/2, np.pi/2,0,0 , -np.pi/2, -np.pi/2,0] #home pose
    self.q00=[0 , 0, 0, 0, 0,0,0, 0, 0,0]
    self.robot = rtb.DHRobot( 
        [ 
            #first UR
            rtb.RevoluteDH(d=0.1625, alpha = np.pi/2,offset=np.pi), #theta=np.pi either here or in XML using ref=""
            rtb.RevoluteDH(a=-0.425),
            rtb.RevoluteDH(a=-0.3922),
            rtb.RevoluteDH(d=0.1333, alpha=np.pi/2),
            #second UR
            rtb.RevoluteDH(d=0.1625, alpha = np.pi/2,offset=np.pi), #theta=pi
            rtb.RevoluteDH(a=-0.425),
            rtb.RevoluteDH(a=-0.3922),
            rtb.RevoluteDH(d=0.1333, alpha=np.pi/2),
            rtb.RevoluteDH(d=0.0997, alpha=-np.pi/2),
            rtb.RevoluteDH(d=0.0996)
        ], name="UR5e",
        base=robot_base,
        #add tool frame if used: tool=tool_matrix
        )
    self.n=self.robot.n
    self.q0=self.q0[:self.n]
    self.T_EE_TCP=sm.SE3.Trans(0.0823,0,0)

    #shared data for control thread
    self.control_enabled=1
    self.qref=self.q00
    self.dqref=self.q00
    


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
   
  def setJointTorques(self,torques): #set joint torque vector which is applied to simulation from next time step
    with self.jointLock:
      for i in range(0, self.n):
        self.jointTorques[i] = torques[i]
      self.sendPositions = True
  
  def control_loop(self):
    while True:
      time.sleep(self.dt)
      if self.control_enabled:
        #CONTROLLER GOES HERE!
        
        self.setJointTorques([self.qref[0]+1000,1000,0,0,0,0,0,0,0,0])
        
  def force_mag_func(x, Fmax_param, d_param, Fd_param): 
    # function f(x) = 1 / (e^a(x^2)) has gaussian like shape
    # x is the input distance between the objects
    # F_max is desired force at zero distance
    # F_d is the desired force at some distance d                                       
    #
    #       │                                
    # F_max-+-....                                      
    #       │     ....                                  
    #       │         ..                                
    #       │           ..                               
    #       │             ..                            
    #       │               ..                          
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

  def jacob_revol(self, l, q):
    T_0_l = self.robot.A(l, q).A
    rl = self.robot[l].r[..., np.newaxis]
    rl = np.row_stack((rl, 1))
    p_0_ll = T_0_l@rl
    p_0_ll = p_0_ll[0:3] / p_0_ll[3]
    
    Jp = np.zeros((3, self.n))
    Jo = np.zeros((3, self.n))
    for i in range(l):
      T_0_i = self.robot.A(i, q).A
      zi = T_0_i[0:3, 3]
      ri = self.robot[l].r[..., np.newaxis]
      ri = np.row_stack((ri, 1))
      p_0_li = T_0_i@ri
      p_0_li = p_0_li[0:3] / p_0_li[3]
      Jo[:, i] = zi
      Jp[:, i] = np.cross(zi, np.squeeze(p_0_ll - p_0_li))
    return np.row_stack((Jo, Jp))

  def getNullProjMat(self, q): # dynamic projection matrix N, such that tau = tau_main + N@tau_second
    Je = self.robot.jacobe(q)
    Je_inv = np.linalg.pinv(Je)
    M = self.robot.inertia(q) # need to calculate this
    N = M@(np.eye(self.n) - Je_inv@Je)@np.linalg.inv(M)
    return N

  def getJointAngles(self):
    ## State of the simulater robot 
    qState=[]    
    for i in range(0, self.n):
      qState.append(float(self.d.joint(f"joint{i+1}").qpos)) 
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

  def start(self):
    #launch simulation thread
    self.jointLock = Lock()
    self.sendPositions = False
    mujoco_thrd = Thread(target=self.launch_mujoco, daemon=True)
    mujoco_thrd.start()
    control_thrd = Thread(target=self.control_loop,daemon=True) #control loop for commanding torques
    control_thrd.start()

if __name__ == "__main__":
  sim=simulation()
  sim.start() 

  time.sleep(5)
  sim.control_enabled=0
  time.sleep(2)
  sim.setJointTorques([0,0,0,0,0,0,0,0,0,0]) #change torques for controller
  time.sleep(5)
  #check fkine vs mujoco EE frames
  print(sim.robot.fkine(sim.getJointAngles()))
  print(sim.getObjFrame("ee_link2"))



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
 