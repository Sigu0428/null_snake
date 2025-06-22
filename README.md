# Advanced robot project: Null space control
[link to video demonstration](https://youtu.be/_rYatMV-v3g)

The purpose of this project is to make a null space controller so a robot can solve multiple tasks simultaneously.
We chose the problem of simultaneous trajectory following and obstacle avoidance with a highly redundant robot.
Null space control is particularly useful when doing dynamic-obstacle avoidance, where planning is not sufficient.

We created a robot with a lot of redundancy by attaching part of a UR robot to the end of a UR robot in simulation (MuJoCo):
![10_DOF_robot](https://github.com/user-attachments/assets/eb333b85-16ab-457d-af0c-85f9305ea466)

We designed two different environments in simulation to test the control schemes, a simple and dense environment respectively:
![simpleScenario](https://github.com/user-attachments/assets/6f0555e1-65d2-4be9-a87f-cbd7fd70e64c)
![dense environment](https://github.com/user-attachments/assets/5119d333-dbf8-46ff-bd3a-bd8042561734)

To represent the distance to obstances we use a potential field approach, shown here in top-down view for the simple environment:
![Screenshot From 2025-06-22 12-50-13](https://github.com/user-attachments/assets/5161670f-9061-4a1b-9c90-9d3ea24a80ab)

# Force based avoidance as secondary task.
For this control scheme, the potential field is used to generate forces on the links of the robot as a secondary task.
These forces are projected into joint torques using the transpose-Jacobian, specifically the positional part.
They are then projected into the null space of the end-effector Jacobian, so they won’t interfere with the main task of trajectory following.
![Screenshot From 2025-06-22 13-03-09](https://github.com/user-attachments/assets/6ef40721-4055-44f7-ab17-ae884887eb47)
![Screenshot From 2025-06-22 12-54-16](https://github.com/user-attachments/assets/2c464059-e237-40a8-8445-43dd1d7ad64d)

# Velocity based avoidance as primary task.
Anthony A. Maciejewski and Charles A. Klein, "Obstacle Avoidance for Kinematically Redundant Manipulators in Dynamically Varying Environments", The International Journal of Robotics Research.

The paper above presents a method of obstacle avoidance by prioritizing two Cartesian velocities differently.
One is a desired end effector velocity defined at the end effector.
The other is defined at the point closest to an obstacle and pointing directly away from the obstacle.
We implemented the method on our robot in combination with a resolved acceleration controller for operational space tracking.

![Screenshot From 2025-06-22 13-02-54](https://github.com/user-attachments/assets/9d0497b7-04b0-4fd7-8e9f-62795f86d987)
![Screenshot From 2025-06-22 12-54-38](https://github.com/user-attachments/assets/8efe1322-61fe-4f5b-b6e4-52f57679a60c)
