# UR Dynamics
This repository contains files for generating a `C++` class for calculating the  dynamics of UR robots.

The script `python/generate6LinkModelSymengine.py` calculates the dynamics using the Lagrangian method using the libraries `SymPy` and `SymEngine` and exports it to a `C++` class defined in `lib/ur_robot.h` and `src/ur_robot.cpp`.

By default, the constructor and the desctructor of the class are not defined.
This is up to you.
In the constructor you have to assign the kinematic and the dynamic parameters.
An example of this is given in `src/ur_robot_constructor.cpp`.

Currently the following functions are implemented:
* `Eigen::Matrix<double, 6, 1> gravity(Eigen::Matrix<double, 6, 1> q);`
* `Eigen::Matrix<double, 6, 6> jacobian(Eigen::Matrix<double, 6, 1> q);`
* `Eigen::Matrix<double, 6, 6> inertia(Eigen::Matrix<double, 6, 1> q);`
* `Eigen::Matrix<double, 6, 6> coriolis(Eigen::Matrix<double, 6, 1> q, Eigen::Matrix<double, 6, 1> dq);`

## Installation

    python3 -m pip install .
