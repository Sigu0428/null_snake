#include <pybind11/pybind11.h>
#include <pybind11/eigen/matrix.h>
#include <ur_robot.h>

namespace py = pybind11;

namespace ur_robot {
PYBIND11_MODULE(ur_robot, m)
{
    m.doc() = "UR Dynamics Interface";

    py::enum_<RobotType>(m, "RobotType")
        .value("UR3e", RobotType::UR3e)
        .value("UR5e", RobotType::UR5e);

    py::class_<URRobot>(m, "URRobot")
        .def(py::init<>())
        .def(py::init<RobotType>())
        .def("gravity", &URRobot::gravity, "Get the gravity vector.",
             py::arg("q"))
        .def("jacobian", &URRobot::jacobian, "Get the Jacobian matrix.",
             py::arg("q"))
        .def("jacobianDot", &URRobot::jacobianDot, "Get the Jacobian derivative w.r.t. time matrix.",
             py::arg("q"), py::arg("dq"))
        .def("inertia", &URRobot::inertia, "Get the inertia matrix.",
             py::arg("q"))
        .def("coriolis", &URRobot::coriolis, "Get the coriolis matrix",
             py::arg("q"), py::arg("dq"));
}
}
