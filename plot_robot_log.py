
import matplotlib.pyplot as plt
import pickle


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
            translational_pos.append(data[i][:, 3])
        
        return translational_pos
        
    def plot(self):
        trans_data = self.get_translational(self.ee_pos)
        trans_data_des = self.get_translational(self.ee_des)

        print(trans_data[0].shape)

        #normalize the torque data

        fig, (ax1) = plt.subplots(1, 1)
        ax1.plot(trans_data[:1000], label='Translational pos')
        ax1.plot(trans_data_des[:1000], label='Translational pos desired')
        ax1.set_title('Plot 1')
        ax1.legend()

        plt.show()







robot_data_plot = robot_plot("null snake")

robot_data_plot.get_data()
robot_data_plot.plot()

