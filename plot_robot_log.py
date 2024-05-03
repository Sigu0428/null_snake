
import matplotlib.pyplot as plt
import pickle
from spatialmath import UnitQuaternion
from spatialmath.base import q2r, r2x,tr2rt
import spatialmath as sm
import numpy as np

class robot_plot:
    def __init__(self, robot_name):
        self.robot_name = robot_name

    def get_data(self):
        self.ee_pos = pickle.load(open("robot_end_effector_position.txt", "rb"))
        self.ee_des = pickle.load(open("robot_end_effector_position_desired.txt", "rb"))

    
    def get_translational(self, data):
        '''
        This function gives the translational position of the end effector
        '''


        translational_pos = []

        for i in range(len(data)):
            
            t=tr2rt(np.array(data[i]))
            translational_pos.append(t[1])
        
        return translational_pos
    
    def get_orientational(self, data):
        '''
        This function gives the translational position of the end effector
        '''


        orientational_pos = []

        for i in range(len(data)):

            R=tr2rt(np.array(data[i]))

            rot=r2x(R[0])
            orientational_pos.append(rot)
        
        return orientational_pos
               
    def plot(self):
        trans_data = self.get_translational(self.ee_pos)
        trans_data_des = self.get_translational(self.ee_des)

        #print(trans_data[0].shape)


        rot_data = self.get_orientational(self.ee_pos)
        rot_data_des = self.get_orientational(self.ee_des)
        
        #normalize the torque data

        fig, (ax1) = plt.subplots(2, 1)
        ax1[0].plot(trans_data[:1000], label='translation')
        ax1[0].plot(trans_data_des[:1000], label='translation desired')
        ax1[0].set_title('translation')
        ax1[0].legend(["x","y","z","x_d","y_d","z_d"])
        ax1[1].plot(rot_data[:1000], label='orientation')
        ax1[1].plot(rot_data_des[:1000], label='orientation desired')
        ax1[1].set_title('orientation')
        ax1[1].legend(["roll","pitch","yaw","roll_d","pitch_d","yaw_d"])
        plt.show()







robot_data_plot = robot_plot("null snake")

robot_data_plot.get_data()
robot_data_plot.plot()

