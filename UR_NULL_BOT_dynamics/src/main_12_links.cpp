#include <iostream>
#include <chrono>
#include <vector>

#include "ur_robot.h"

int main(int, char**) 
{
    std::cout << "Hello, world!\n";
    
    URRobot robot(UR5eTIMES2);

    // double q[] = { 1.0000,   1.0472,    1.0472};
    // double dq[] = { 1.0000,   1.0472,    1.0472};
    // double ddq[] = { 1.0000,   1.0472,    1.0472};
    const int N = 12;

    Eigen::Matrix<double, N, 1> q, dq, ddq;



    q <<    1.0000,   1.0472,    1.0472, 1.0472, 1.0472, 1.0472, 1.0000,   1.0472,    1.0472, 1.0472, 1.0472, 1.0472;
    dq <<   1.0000,   1.0472,    1.0472, 1.0472, 1.0472, 1.0472, 1.0000,   1.0472,    1.0472, 1.0472, 1.0472, 1.0472;
    ddq <<  1.0000,   1.0472,    1.0472, 1.0472, 1.0472, 1.0472, 1.0000,   1.0472,    1.0472, 1.0472, 1.0472, 1.0472;

    std::cout << "q" << std::endl;
    for (double x : q)
        std::cout << x << ", ";
    std::cout << std::endl;

    std::cout << "dq" << std::endl;
    for (double x : dq)
        std::cout << x << ", ";
    std::cout << std::endl;

    std::cout << "ddq" << std::endl;
    for (double x : ddq)
        std::cout << x << ", ";
    std::cout << std::endl;



    Eigen::Matrix<double, N, 1> grav = robot.gravity(q);
    std::cout << "gravity vector" << std::endl;
    std::cout << grav << std::endl;

    Eigen::Matrix<double, 6, N> jac = robot.jacobian(q);
    std::cout << "Jacobian matrix" << std::endl;
    std::cout << jac << std::endl;

    Eigen::Matrix<double, 6, N> jacDot = robot.jacobianDot(q, dq);
    std::cout << "Jacobian dot matrix" << std::endl;
    std::cout << jacDot << std::endl;

    Eigen::Matrix<double, N, N> inertia = robot.inertia(q);
    std::cout << "Inertia matrix" << std::endl;
    std::cout << inertia << std::endl;

    Eigen::Matrix<double, N, N> Coriolis = robot.coriolis(q, dq);
    std::cout << "Coriolis matrix" << std::endl;
    std::cout << Coriolis << std::endl;

    using std::chrono::high_resolution_clock;
    using std::chrono::duration_cast;
    using std::chrono::duration;
    using std::chrono::milliseconds;

    double time_list[10000];

    for (size_t i = 0; i < 10000; i++)
    {
        for (size_t j = 0; j < N; j++)
        {
            q[j] = i / 10000.;
            dq[j] = i / 10000.;
        }

        auto t1 = high_resolution_clock::now();    
        Eigen::Matrix<double, N, N> coriolis = robot.coriolis(q, dq);
        coriolis.transpose();
        auto t2 = high_resolution_clock::now();

        duration<double, std::milli> ms_double = t2 - t1;
        time_list[i] = ms_double.count();
    }

    double sum_list = 0.;
    double max_val = 0.;
    double min_val = 1.;
    int above_limit = 0;
    for (size_t i = 0; i < 10000; i++)
    {
        sum_list += time_list[i];
        if (time_list[i] > max_val)
            max_val = time_list[i];
        if (time_list[i] < min_val)
            min_val = time_list[i];

        if (time_list[i] > 0.05)
            above_limit++;
    }

    double avg = sum_list / 10000.;
    std::cout << "avg " << avg << std::endl;
    std::cout << "max " << max_val << std::endl;
    std::cout << "min " << min_val << std::endl;
    std::cout << "count above limit " << above_limit << std::endl;

    // std::cout << "Coriolis matrix" << std::endl;
    // std::cout << coriolis << std::endl;

    // std::cout << "Coriolis matrix transpose" << std::endl;
    // std::cout << coriolis.transpose() << std::endl;

    // std::cout << "Velocity product" << std::endl;
    // std::cout << coriolis * dq << std::endl;

    // Eigen::Matrix<double, 6, 1> tau = inertia * ddq + coriolis * dq + grav;
    // std::cout << "Torque" << std::endl;
    // std::cout << tau << std::endl;

}

