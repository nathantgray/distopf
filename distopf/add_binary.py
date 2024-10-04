"""
q = u*q_nom*v
q_v = q_nom*v
q_v_max = q_nom*v_max
q_cap = u*q_v
z = A*x => q_cap = q_nom*v*u
A == q_nom*v
x == u
q_cap == q_nom*z_c
z_c == v*u
z_c <= v_max*u
z_c <= v
z_c >= v - (1 - u)*v_max
z_c >= 0

Qij - sum(Qjk) - (bus.cvr_q[j] / 2) * q_load*v + Qdg + q_nom*z_c = (1 - (bus.cvr_q[j] / 2)) * q_load

A_ub*x <= b_ub

1*z_c - v_max*u <= 0
1*z_c - 1*v <=0
-1*z_c + 1*v + v_max*u <= v_max
-1*z_c <= 0
"""
