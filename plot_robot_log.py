
import matplotlib.pyplot as plt
import pickle


class robot_plot:
    def __init__(self, robot_name):
        self.robot_name = robot_name

    def get_data(self):
        self.ee_pos = pickle.load(open("robot_end_effector_position.txt", "rb"))
        self.torque_vals = pickle.load(open("robot_joint_torques.txt", "rb"))

        print(len(self.ee_pos))
        print(len(self.torque_vals))
    
    def get_translational_position(self):

        translational_pos = []

        for i in range(len(self.ee_pos)):
            translational_pos.append(self.ee_pos[i][:, 3])
        
        return translational_pos
        
    def plot(self):
        trans_data = self.get_translational_position()
        torque_data = self.torque_vals
        print(trans_data[0].shape)

        #normalize the torque data
        torque_data_normalized = [i/200 for i in torque_data]
        fig, (ax1) = plt.subplots(1, 1)
        ax1.plot(trans_data[:1000], label='Translational pos')

        ax1.plot(torque_data_normalized[:1000], label='torque')
        ax1.set_title('Plot 1')
        ax1.legend()

        plt.show()







robot_data_plot = robot_plot("null snake")

robot_data_plot.get_data()
robot_data_plot.plot()

