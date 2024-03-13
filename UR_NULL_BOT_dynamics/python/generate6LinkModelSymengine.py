#!/usr/bin/env python3
import numpy as np
import sympy.utilities.codegen
# from symengine import *
# import sympy
import symengine as se
import sympy as sp
import time
import copy
import textwrap

# from sympy.utilities.codegen import codegen
# from symengine.printing import CCodePrinter

import print2ccode

N = 6

# Define symbols
q = se.Matrix(se.symbols(f'q[:{N}]'))
dq = se.Matrix(se.symbols(f'dq[:{N}]'))
ddq = se.Matrix(se.symbols(f'ddq[:{N}]'))

# DH parameters
a = se.Matrix(se.symbols(f'a1:{N + 1}_'))
a_vals = [0, se.nan, se.nan, 0, 0, 0]
for i in range(N):
    if a_vals[i] == se.nan:
        pass
    else:
        a[i] = a_vals[i]
    # a[i] = a_vals[i] if not np.isnan(a_vals[i]) else a[i]
# a[0] = 0    
# a[3] = 0
# a[4] = 0
# a[5] = 0
alpha = se.Matrix(se.symbols(f'alpha[:{N}]'))
alpha_vals = [se.S(se.pi)/2, 0, 0, se.S(se.pi)/2, -se.S(se.pi)/2, 0]
for i in range(N):
    # alpha[i] = alpha_vals[i] if not np.isnan(alpha_vals[i]) else alpha[i]
    if alpha_vals[i] == se.nan:
        pass
    else:
        alpha[i] = alpha_vals[i]

# alpha[0] = S(se.pi)/2
# alpha[1] = 0
# alpha[2] = 0
# alpha[3] = S(se.pi)/2
# alpha[4] = -S(se.pi)/2
# alpha[5] = 0
d = se.Matrix(se.symbols(f'd1:{N + 1}_'))
d_vals = [se.nan, 0, 0, se.nan, se.nan, se.nan]
for i in range(N):
    # d[i] = d_vals[i] if not np.isnan(d_vals[i]) else d[i]
    if d_vals[i] == se.nan:
        # print("se.nan")
        pass
    else:
        d[i] = d_vals[i]
# d[1] = 0
# d[2] = 0

# pl = Matrix(symbols(f'pl[:{N}][:{3}]')).reshape(N, 3)
pl = se.Matrix(se.symbols(f'com_[:{N}][:{3}]')).reshape(N, 3)

# Dynamic parameters
# m = Matrix(symbols(f'm[:{N}]')) # mass
m = se.Matrix(se.symbols(f'm_[:{N}]')) # mass

# InertiaMatrix = [Matrix(MatrixSymbol(f'I_{i}', 3, 3)) for i in range(N)]
# InertiaMatrix = [Matrix(symbols(f'I[{i}][:{3}][:{3}]')).reshape(3, 3) for i in range(N)]
InertiaMatrix = [se.Matrix(se.symbols(f'link_inertia_[{i}][:{3}][:{3}]')).reshape(3, 3) for i in range(N)]

for i in range(N):
    InertiaMatrix[i] = sp.Matrix(np.tril(InertiaMatrix[i].T, 0)) + sp.Matrix(np.tril(InertiaMatrix[i].T, -1).T)

# pprint(InertiaMatrix[0][:, :])
# print(InertiaMatrix)
# exit(0)
# gravity
g0 = se.Matrix(se.symbols(f'g[:{3}]'))
# g0[0] = 0.
# g0[1] = 0.
# g0[2] = -g0[2]
# pprint(g0)

## Forward kinematics
T_rel = [se.zeros(4, 4) for i in range(N)]
for i in range(N):
    ci = se.cos(q[i])
    si = se.sin(q[i])
    T_rel[i] = se.Matrix([[ci, -si * se.cos(alpha[i]),  si * se.sin(alpha[i]), a[i] * ci],
                   [si,  ci * se.cos(alpha[i]), -ci * se.sin(alpha[i]), a[i] * si],
                   [0,       se.sin(alpha[i]),       se.cos(alpha[i]),      d[i]],
                   [0,                   0,                   0,         1]])

    # pprint(T_rel[i])

T_0_0 = se.eye(4)

T_base = T_rel.copy()

for i in range(1, N):
    T_base[i] = T_base[i - 1] @ T_rel[i]

    # print("T_base", i)
    # # pprint(T_base[i])

## Coordinates for CoM for each link given in base frame
p_0_l0 = se.zeros(3, 1)
p_0_links = se.zeros(3, N)
III = se.Matrix(sp.eye(3, 4))

for i in range(N):
    tmp = se.zeros(4, 1)
    tmp[0:3] = pl[i, :].T
    tmp[3] = 1
    p_0_links[:, i] = III @ T_base[i] @ tmp

# pprint(p_0_links[:, 0])


## Compute the Jacobians given in Base frame
J_O_link = [se.Matrix(se.zeros(3, N)) for _ in range(N)]
# pprint(J_O_link[0])

J_O_link[0][:, 0] = T_0_0[0:3, 2]

for i in range(1, N):
    J_O_link[i][:, :] = J_O_link[i - 1][:, :]
    J_O_link[i][:, i] = T_base[i - 1][0:3, 2]
#     pprint(J_O_link[i])

    # J_O_link[i] = simplify(J_O_link[i])
    # J_O_link[i] = J_O_link[i].applyfunc(simplify)

# print(J_O_link)

J_P_link = [se.zeros(3, N) for _ in range(N)]
for i in range(N):
    for j in range(i + 1):
        # print(j)
        if j == 0: # TODO: fix this crap
            J_P_link[i][:, j] = T_0_0[0:3, 2].cross(p_0_links[:, i] - T_0_0[0:3, 3])
        else:
            J_P_link[i][:, j] = T_base[j - 1][0:3, 2].cross(p_0_links[:, i] - T_base[j - 1][0:3, 3])

    # J_P_link[i] = simplify(J_P_link[i])
    # J_P_link[i] = J_P_link[i].applyfunc(simplify)

## Compute the end effector Jacobian
J_ee = se.Matrix(se.symbols(f"J:{6}:{N}")).reshape(6, N)
J_ee[0:3, 0] = T_0_0[0:3, 2].cross(T_base[-1][0:3, 3] - T_0_0[0:3, 3])
J_ee[3:6, 0] = T_0_0[0:3, 2]

for i in range(1, N):
    J_ee[0:3, i] = T_base[i - 1][0:3, 2].cross(T_base[-1][0:3, 3] - T_base[i - 1][0:3, 3])  # translational
    J_ee[3:6, i] = T_base[i - 1][0:3, 2]  # oriental

# Time derivative of Jacobian
J_dot = se.Matrix(se.symbols(f"J_dot:{6}:{N}")).reshape(6, N)

t = se.symbols('t')
q_ = se.Matrix([se.function_symbol(f'q_[{i}]', t) for i in range(N)])
# dq_ = se.Matrix([se.function_symbol(f'dq_[{i}]', t) for i in range(N)])
# ddq_ = se.Matrix([se.function_symbol(f'ddq_[{i}]', t) for i in range(N)])

# print(q_)
# print(se.diff(q_, t))

J_ee_ = se.Subs(J_ee, q, q_)

for i in range(6):
    for j in range(N):
        J_dot[i, j] = se.diff(J_ee_[i, j], t)


# print(J_ee[0,0])
# print(J_dot[0,0])
J_dot = se.Subs(J_dot, se.diff(q_, t), dq)
J_dot = se.Subs(J_dot, q_, q)

# print(J_dot[0,0])
# exit(0)

# Codegen Jacobian
# export_function(J_ee)
# sp.codegen(('jacobian', J_ee),
#         language="c", to_files=True, prefix="SymPyCodeGen/jacobian_se")


# print(J_ee_code)

## Compute inertia for each link given in base frame
Inertia_link = [se.zeros(3, 3) for _ in range(N)]
for i in range(N):
    Inertia_link[i] = T_base[i][0:3, 0:3] * InertiaMatrix[i] * T_base[i][0:3, 0:3].T


## Compute gravity term
Epot = se.Matrix(se.zeros(1, 1))
for i in range(N):
    Epot = Epot - m[i] * g0.T * p_0_links[:, i]
# pprint(Epot)
# print(Epot)
# print(shape(Epot))

gravity = Epot.jacobian(q).T
# gravity = expand(gravity)
# print("gravity[0]")
print(gravity[0])
# print("gravity[1]")
# print(gravity[1])

# sympy.utilities.codegen.codegen(('gravity', gravity), language="octave", to_files=True, prefix="gravity")
        # global_vars=(a, alpha, d, pl, m, InertiaMatrix, ))

## Compute inertia matrix
print("Compute inertia matrix")
start = time.time()
B = se.zeros(N, N)
for i in range(N):
    B = B + m[i] * J_P_link[i].T * J_P_link[i] + J_O_link[i].T * Inertia_link[i] * J_O_link[i]

#

# B = simplify(B)
# B = factor_terms(B)
# B = collect(B, q)

# B.print_nonzero()
# Export inertia matrix


## Compute coriolis and centrifugal term
print("Compute coriolis and centrifugal term")

start = time.time()
C = se.zeros(N, N)
for i in range(N):
    for j in range(N):
        for k in range(N):
            C[i, j] = C[i, j] + (
                se.diff(B[i, j], q[k]) + \
                se.diff(B[i, k], q[j]) - \
                se.diff(B[j, k], q[i])
            ) * dq[k]
        # C[i, j] = expand(C[i, j])

        # C[j, i] = C[i, j].copy()

# C = C.doit()
C = 1/se.S(2) * C

# print(C)


print(f"elapsed = {time.time() - start}s")



# print("C1")
# print(C1)
#
# Chr = [zeros(N, N) for _ in range(N)]
# for i in range(N):
#     for j in range(N):
#         for k in range(j, N):
#             Chr[i][j, k] = 1/S(2) * (
#                 better_diff(B[i, j], q[k], evaluate=False) + \
#                 better_diff(B[i, k], q[j], evaluate=False) - \
#                 better_diff(B[j, k], q[i], evaluate=False)
#             )
#             Chr[i][k, j] = Chr[i][j, k]
#
# C2 = zeros(N, N)
# for i in range(N):
#     for j in range(N):
#         for k in range(N):
#             C2[i, j] = C2[i, j] + Chr[i][j, k] * dq[k]


# print("C2")
# print(C2)
# C.print_nonzero()

# J_ee_code = printer.doprint(J_ee, 'J')
J_ee_code = print2ccode.expr2ccode(J_ee, 'J', optimize=True)
# print(J_ee_code)
# TODO: use se.cse([element]) to refactor the function
J_dot_code = print2ccode.expr2ccode(J_dot, 'J_dot', optimize=True)

print("Export gravity")
start = time.time()
# grav_code = printer.doprint(gravity, "grav")
grav_code = print2ccode.expr2ccode(gravity, 'grav', optimize=True)
print(f"elapsed = {time.time() - start}s")
# print(grav_code)

print("Export inertia matrix")
# B_code = printer.doprint(B, 'B')
B_code = print2ccode.expr2ccode(B, 'B', optimize=True)
print(f"elapsed = {time.time() - start}s")

print("Exporting Coriolis to file")
start = time.time()
# C_code = printer.doprint(C, 'C')
C_code = print2ccode.expr2ccode(C, 'C', optimize=True)
print(f"elapsed = {time.time() - start}s")

"""
Eigen::Matrix<double, 6, 6> URModel::inertia_matrix(std::vector<double> q)
{
  //start_time = clock();
  Eigen::Matrix<double, 6, 6> inertia_matrix_;
  inertia_matrix_.fill(0);
  """

def export2file(fcnname : str, content : str, rtnname : str, arguments, rows, columns):
    # argument_str = [f"double * {arg}" for arg in arguments]
    argument_str = [f"Eigen::Matrix<double, {N}, 1> & {arg}" for arg in arguments]
    argument_str = ", ".join(argument_str)

    descriptor = f"Eigen::Matrix<double, {rows}, {columns}> {fcnname} ({argument_str})"
    descriptor_internal = f"Eigen::Matrix<double, {rows}, {columns}> URRobot::{fcnname}({argument_str})"

# for line in content:
#             new_line = "\n".join(line[n:n + lw] for n in range(0, len(line), lw))

    # lw = 50
    # new_content = ""
    # for line in content:
    #     new_line = "\n".join(line[n:n + lw] for n in range(0, len(line), lw))
    #     new_content += new_line + "\n"
    # new_content = '\n'.join(textwrap.wrap(content, 50))
    # new_content = '+\n'.join(content.split('+'))

    return_val = ""
    if columns == 1:
        return_val = f"""
    Eigen::Map<Eigen::Matrix<double, {rows}, {columns}>> out({rtnname}.data());
        """
    else:
        return_val = f"""
    Eigen::Map<Eigen::Matrix<double, {rows}, {columns}, Eigen::RowMajor>> out({rtnname}.data());
        """        

    fcn_content = f"""
{descriptor_internal}
{{
    std::vector<double> {rtnname}({rows * columns}, 0.0);
    {content}
    {return_val}
    return out;
}}
    """
    return descriptor, fcn_content

gravityH, gravityFile = export2file("gravity", grav_code, "grav", ["q"], *gravity.shape)
# print(gravityFile)

jacobianH, jacobianFile = export2file("jacobian", J_ee_code, "J", ["q"], *J_ee.shape)
# print(jacobianFile)

jacobianDotH, jacobianDotFile = export2file("jacobianDot", J_dot_code, "J_dot", ["q", "dq"], *J_dot.shape)

inertiaH, inertiaFile = export2file("inertia", B_code, "B", ["q"], *B.shape)
# print(inertiaFile)

coriolisH, coriolisFile = export2file("coriolis", C_code, "C", ["q", "dq"], *C.shape)
# print(coriolisFile)

descriptions = [gravityH, jacobianH, jacobianDotH, inertiaH, coriolisH]
contents = [gravityFile, jacobianFile, jacobianDotFile, inertiaFile, coriolisFile]

# Write to a file
with open("lib/ur_robot.h", "w") as f:
    f.write("#include <Eigen/Dense>\n\n")
    f.write("// This file has been generated by generate6LinkModelSymengine.py.\n")
    f.write("// Edit at your own risk.\n\n")

    f.write("enum RobotType\n")
    f.write("{\n")
    f.write("\tUR3e,\n")
    f.write("\tUR5e\n")
    f.write("};\n")
    f.write("\n\n")

    f.write("class URRobot\n")
    f.write("{\n")

    f.write("public:\n")
    f.write("\tURRobot();\n") # constructor
    f.write("\tURRobot(RobotType type);\n")  # constructor
    f.write("\t~URRobot();\n") # destructor
    f.write("\tint N;\n")

    for description in descriptions:
        f.write(f"\t{description};")
        f.write("\n\n")

    f.write("private: \n")
    a_args = "\n".join([f"\tdouble {str(x)};" for x in a if str(x) != '0'])
    d_args = "\n".join([f"\tdouble {str(x)};" for x in d if str(x) != '0'])
    m_args = f"\tdouble {str(m[0]).split('[')[0]}[{N}];"
    # pl_args = f"\tEigen::Matrix<double, {pl.shape[0]}, {pl.shape[1]}> {str(pl[0, 0]).split('[')[0]};"
    pl_args = f"\tdouble {str(pl[0, 0]).split('[')[0]}[{pl.shape[0]}][{pl.shape[1]}];"
    # I_args = f"\tEigen::Matrix<double, {N}, 3, 3> {str(InertiaMatrix[0][0, 0]).split('[')[0]};"
    I_args = f"\tdouble {str(InertiaMatrix[0][0, 0]).split('[')[0]}[{N}][3][3];"
    g_args = f"\tdouble {str(g0[0]).split('[')[0]}[3];"

    f.writelines(f"""
{a_args}
{d_args}
{m_args}
{pl_args}
{I_args}
{g_args}
""")
    f.write("};\n")

lw = 50
with open("src/ur_robot.cpp", "w") as f:
    f.write("#include <ur_robot.h>\n\n")
    f.write("// This file has been generated by generate6LinkModelSymengine.py.\n")
    f.write("// Edit at your own risk.\n\n")

    for content in contents:
        f.write(content)
        f.write("\n")

