import numpy as np
import distopf as opf
from distopf.lindist_mi_base import LinDistModel
from distopf.lindist_mi_q import LinDistModelQ

case = opf.DistOPFCase(
    data_path="ieee123_30der", gen_mult=1, load_mult=1, v_swing=1.0, v_max=1.05, v_min=0.95
)

model = opf.LinDistModelQ(
    branch_data=case.branch_data,
    bus_data=case.bus_data,
    gen_data=case.gen_data,
    cap_data=case.cap_data,
    reg_data=case.reg_data
)
new_model = LinDistModel(
    branch_data=case.branch_data,
    bus_data=case.bus_data,
    gen_data=case.gen_data,
    cap_data=case.cap_data,
    reg_data=case.reg_data
)
# Solve model using provided objective function
result = opf.cvxpy_solve(model, opf.cp_obj_loss)
new_result = opf.cvxpy_solve(new_model, opf.cp_obj_loss)
# result = cvxpy_solve(model, cp_obj_loss)
print(result.fun)
print(result.runtime)
print(new_result.fun)
print(new_result.runtime)
v = model.get_voltages(result.x)
s = model.get_apparent_power_flows(result.x)
dec_var = model.get_decision_variables(result.x)
new_v = new_model.get_voltages(new_result.x)
new_s = new_model.get_apparent_power_flows(new_result.x)
new_dec_var = new_model.get_decision_variables(new_result.x)
# opf.plot_network(model, v, s, dec_var, "Q").show(renderer="browser")
# opf.compare_voltages(v, new_v).show(renderer="browser")
# opf.compare_flows(s, new_s).show(renderer="browser")

