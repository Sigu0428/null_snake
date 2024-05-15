
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
            
            
            translational_pos.append(data[i][:3])
        
        return translational_pos
    
    def get_orientational(self, data):
        '''
        This function gives the translational position of the end effector
        '''


        orientational_pos = []

        for i in range(len(data)):

            R=q2r(data[i][3:])

            rot=r2x(R)
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
        trans_data_np = np.asarray(trans_data)
        trans_data_des_np = np.asarray(trans_data_des)
        rot_data_np = np.asarray(rot_data)
        rot_data_des_np = np.asarray(rot_data_des)


        ax1[0].plot(trans_data_np[:,0], label='translation', color='red')
        ax1[0].plot(trans_data_des_np[:,0], label='translation desired',linestyle='--', linewidth=1.0, color='red')


        ax1[0].plot(trans_data_np[:,1], label='translation', color='green')
        ax1[0].plot(trans_data_des_np[:,1], label='translation desired',linestyle='--', linewidth=1.0, color='green')
        
        
        ax1[0].plot(trans_data_np[:,2], label='translation',  color='blue')
        ax1[0].plot(trans_data_des_np[:,2], label='translation desired',linestyle='--', linewidth=1.0, color='blue')

        ax1[0].set_title('translation')
        ax1[0].legend(["x","x_d","y","y_d", "z","z_d"])




        ax1[1].plot(rot_data_np[:,0], label='translation', color='red')
        ax1[1].plot(rot_data_des_np[:,0], label='translation desired',linestyle='--', linewidth=1.0, color='red')


        ax1[1].plot(rot_data_np[:,1], label='translation',  color='green')
        ax1[1].plot(rot_data_des_np[:,1], label='translation desired',linestyle='--', linewidth=1.0, color='green')
        
        
        ax1[1].plot(rot_data_np[:,2], label='translation',  color='blue')
        ax1[1].plot(rot_data_des_np[:,2], label='translation desired',linestyle='--', linewidth=1.0, color='blue')

        ax1[1].set_title('orientation')
        ax1[1].legend(["roll","roll_d", "pitch", "pitch_d", "yaw","yaw_d"])

        plt.show()




if __name__ == "__main__":

    robot_data_plot = robot_plot("null snake")

    robot_data_plot.get_data()
    robot_data_plot.plot()