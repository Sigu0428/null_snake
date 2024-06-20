
import matplotlib.pyplot as plt
import pickle
from spatialmath import UnitQuaternion
from spatialmath.base import q2r, r2x,tr2rt
import spatialmath as sm
import numpy as np
from sim import simulation

class robot_plot:
    def __init__(self, robot_name):
        self.robot_name = robot_name

    def get_data(self):
        self.ee_pos = pickle.load(open("robot_end_effector_position.txt", "rb"))
        self.ee_des = pickle.load(open("robot_end_effector_position_desired.txt", "rb"))
        self.distances = pickle.load(open("distances.txt", "rb"))
        self.contacts = pickle.load(open("contacts.txt", "rb"))
        self.times= pickle.load(open("timelist.txt","rb"))

    
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

            #R=q2r(data[i][3:])

            #rot=r2x(R)
            #orientational_pos.append(rot)
            orientational_pos.append(data[i][3:])
        
        return orientational_pos
               
    def plot(self):
        trans_data = self.get_translational(self.ee_pos)
        trans_data_des = self.get_translational(self.ee_des)

        #print(trans_data[0].shape)


        rot_data = self.get_orientational(self.ee_pos)
        rot_data_des = self.get_orientational(self.ee_des)
        
        #normalize the torque data

        fig, (ax1) = plt.subplots(3, 1)
        trans_data_np = np.asarray(trans_data)
        trans_data_des_np = np.asarray(trans_data_des)
        rot_data_np = np.asarray(rot_data)
        rot_data_des_np = np.asarray(rot_data_des)
        sim=simulation("aylmao")

        ax1[0].plot(self.times,trans_data_np[:,0], label='translation', color='red')
        ax1[0].plot(self.times,trans_data_des_np[:,0], label='translation desired',linestyle='--', linewidth=1.0, color='red')


        ax1[0].plot(self.times,trans_data_np[:,1], label='translation', color='green')
        ax1[0].plot(self.times,trans_data_des_np[:,1], label='translation desired',linestyle='--', linewidth=1.0, color='green')
        
        
        ax1[0].plot(self.times,trans_data_np[:,2], label='translation',  color='blue')
        ax1[0].plot(self.times,trans_data_des_np[:,2], label='translation desired',linestyle='--', linewidth=1.0, color='blue')

        ax1[0].set_ylabel("[m]")
        ax1[0].set_title('translation')
        ax1[0].legend(["x","x_d","y","y_d", "z","z_d"])




        ax1[1].plot(self.times,rot_data_np[:,0], label='translation', color='red')
        ax1[1].plot(self.times,rot_data_des_np[:,0], label='translation desired',linestyle='--', linewidth=1.0, color='red')


        ax1[1].plot(self.times,rot_data_np[:,1], label='translation',  color='green')
        ax1[1].plot(self.times,rot_data_des_np[:,1], label='translation desired',linestyle='--', linewidth=1.0, color='green')
        
        
        ax1[1].plot(self.times,rot_data_np[:,2], label='translation',  color='blue')
        ax1[1].plot(self.times,rot_data_des_np[:,2], label='translation desired',linestyle='--', linewidth=1.0, color='blue')

        ax1[1].plot(self.times,rot_data_np[:,3], label='translation',  color='blue')
        ax1[1].plot(self.times,rot_data_des_np[:,3], label='translation desired',linestyle='--', linewidth=1.0, color='purple')


        ax1[1].set_title('orientation')
        ax1[1].legend(["s","s_d", "v1", "v1_d", "v2","v2_d", "v3","v3_d"])
        
        # (timestep, link, obstacle)
        
        dist_mat = np.array(self.distances)
        #dist_mat = np.clip(dist_mat, a_min=0, a_max=None)
        dist_mat = np.min(dist_mat, axis=2)
        #print(dist_mat.shape)

        '''
        for i, val in enumerate(dist_mat[:,:]):
            print(val, " -- index", self.times[i])
        '''
            

        ax1[2].plot(self.times,np.min(dist_mat[:,[3,5,7,10]],axis=1), label='translation', color='red')
        dist_mat = np.min(dist_mat, axis=1)

    

        ax1[2].plot(self.times,dist_mat[:], label='translation', color='blue')
        ax1[2].axhline(0.15,color = "orange", linestyle = "--", linewidth=1.0)
        ax1[2].axhline(0.2, color = "purple", linestyle = "--", linewidth=1.0)

        #skibidi toilet
        ohio_rizz=np.reshape(np.array(self.contacts),dist_mat.shape)
        brrr = np.arange(dist_mat.shape[0])


        time_array=np.array(self.times)
        ax1[2].scatter(time_array[np.where(ohio_rizz>0)],dist_mat[np.where(ohio_rizz>0)],color="red")
        ax1[2].set_xlabel("time [s]")
        ax1[2].set_ylabel("[m]")
        ax1[2].set_title('distance to obstacle')
        ax1[2].legend(["Minimum distance (Velocity Controlled)", "Minimum distance (all joints)","Tanh Threshold","Collisions", "OA clamp"])
        
        plt.show()




if __name__ == "__main__":

    robot_data_plot = robot_plot("null snake")

    robot_data_plot.get_data()
    robot_data_plot.plot()