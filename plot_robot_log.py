
import matplotlib.pyplot as plt
import pickle
import numpy as np
from spatialmath import UnitQuaternion


class robot_plot:
    def __init__(self, robot_name, data_folder_name, plot_amounts):
        self.robot_name = robot_name
        self.data_folder_name = data_folder_name
        self.plot_amounts = plot_amounts

        if plot_amounts <= 0:
            print("WARNING: no plots are saved")
        else:
            self.list_plots = [[] for _ in range(plot_amounts)]



    def get_data(self):
        self.ee_pos = [[] for _ in range(self.plot_amounts)]


        for i in range(len(self.list_plots)):
            self.ee_pos[i] = pickle.load(open(self.data_folder_name + f"/data_{i}.txt", "rb"))

    
    def get_translational(self, data):
        '''
        This function gives the translational position of the end effector
        '''


        translational_pos = []

        for i in range(len(data)):
            translational_pos.append(data[i][:, 3])
        
        return translational_pos
        

    def plot(self, names):


        #normalize the torque data

        fig, (ax1) = plt.subplots(1, 1)
        for i in range(len(names)):
            ax1.plot(self.ee_pos[i], label=names[i])


        ax1.set_title('Plot 1')
        ax1.legend()

        plt.show()


    def put_data_in_list(self, data, plot_index):
        '''
        This function takes as input the data that must be saved, and saves it the correct place.
        '''

        self.list_plots[plot_index].append(data)


    def save_data(self):
        '''
        This function will be used to save the data from the robot. given from the log loop
        '''
        i = 0
        for data in self.list_plots:
            with open(f'plot_data/data_{i}.txt', 'wb') as f:
                pickle.dump(data, f)
            i += 1




if __name__ == "__main__":
  

    robot_data_plot = robot_plot("null snake", data_folder_name="plot_data", plot_amounts=2)

    robot_data_plot.get_data()
    robot_data_plot.plot(["end effector position", "desired effector position"])