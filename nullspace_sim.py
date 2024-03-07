
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
    self.m = mujoco.MjModel.from_xml_path('Null space Project/Ur5_robot/Robot_scene.xml')
    self.d = mujoco.MjData(self.m)
    self.joints = [0 , -np.pi/2, -np.pi/2, -np.pi/2, np.pi/2,0,0 , -np.pi/2, -np.pi/2,0] #mujoco launch config
    self.dt = 1/100
    

    # Universal Robot UR5e kiematics parameters
    tool_matrix = sm.SE3.Trans(0., 0., 0.18) #adds tool offset to fkine automatically!!
    robot_base = sm.SE3.Trans(0,0,0)

    self.q0=[0 , -np.pi/2, -np.pi/2, -np.pi/2, np.pi/2,0,0 , -np.pi/2, -np.pi/2,0]
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
  
  def getState(self):
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

  def launch_mujoco(self):
    with mujoco.viewer.launch_passive(self.m, self.d) as viewer:
      # Close the viewer automatically after 30 wall-seconds.
      start = time.time()
      while viewer.is_running():
        step_start = time.time()

        # mj_step can be replaced with code that also evaluates
        # a policy and applies a control signal before stepping the physics.
        # with self.jointLock:
        mujoco.mj_step(self.m, self.d)

        with self.jointLock:
          if self.sendPositions:
            for i in range(0, self.n):
              self.d.actuator(f"actuator{i+1}").ctrl = self.joints[i]
            self.sendPositions = False
    
        viewer.sync()

        # Rudimentary time keeping, will drift relative to wall clock.
        time_until_next_step = self.m.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
          time.sleep(time_until_next_step)
  
  def sendJoint(self,join_values):
    with self.jointLock:
      for i in range(0, self.n):
        self.joints[i] = join_values[i]
      self.sendPositions = True
  
  def send2sim(self, trj):
    # send trajectory step by step to simulation
    for i in trj.q:
      self.sendJoint(i)
      time.sleep(self.dt)
    self.curST = trj.q[len(trj.q)-1]  
    return self.curST  
  
  def start(self):
    self.jointLock = Lock()
    self.sendPositions = False
    mujoco_thrd = Thread(target=self.launch_mujoco, daemon=True)
    mujoco_thrd.start()
    

    # send Robot to init congiguration
    self.sendJoint(self.q0)
    time.sleep(10)
    print(self.robot.fkine(self.q0))
    print(self.getObjFrame("ee_link2"))

    while True: pass
    print(self.getObjDistance("blockL04","ee_link2"))
    
    target="blockL04"

    print("target frame")
    print(self.getObjFrame(target))
    

     # conctruct the cartesian trajectory from current pose to block
    print("current EE")
    T_world_EE=self.robot.fkine(self.getState())
    print(T_world_EE)    



    time.sleep(5)
    T_goal=self.getObjFrame(target)#T_world_ee*T_ee_block
    print("target")
    #
    T_goal=T_goal*self.T_EE_TCP.inv() #account for tool
    print(T_goal)
    #create interpolated trajectory with 100 steps
    Trj=rtb.ctraj(T_world_EE,T_goal,100)

    qik = self.robot.ikine_LM(Trj, q0=self.getState()) #get joint trajectory

    
    for i in qik.q:
      self.sendJoint(i)
      time.sleep(0.1)

    #q_ik = self.robot.ikine_LM(T_goal, q0=self.getState())
    #self.sendJoint(q_ik.q)

    print("final EE frame")
    print(self.robot.fkine(self.getState()))

    #while True:
    #  time.sleep(2)
    #  print(self.getObjDistance("blockL01","wrist_3_link2")) #dist from block to lower wrist 3

    #self.sendJoint([0,0,0,-np.pi/2,0,0,0,-np.pi/2,0,0,0,0])
    print("Press any key to exit")
    input()
    

if __name__ == "__main__":
  simulation().start() 
  