import numpy as np
import distopf as opf
from distopf.lindist import LinDistModel
from distopf.lindist_capacitor_mi import LinDistModelCapMI
from distopf.opf_solver import cvxpy_mi_solve
import pandas as pd
import cvxpy as cp

case = opf.DistOPFCase(
    data_path="ieee123_30der",
    gen_mult=1,
    load_mult=1,
    v_swing=1.0,
    v_max=1.05,
    v_min=0.95,
)
# reg_data = pd.concat([case.reg_data, pd.DataFrame({
# "fb": [128], "tb": [127], "phases": ["abc"], "tap_a": [15.0], "tap_b": [2.0], "tap_c": [5.0]})])
case.cap_data = pd.concat(
    [
        case.cap_data,
        pd.DataFrame(
            {
                "id": [14],
                "name": [632],
                "qa": [0.3],
                "qb": [0.3],
                "qc": [0.5],
                "phases": ["abc"],
            }
        ),
    ]
)
model = LinDistModelCapMI(
    branch_data=case.branch_data,
    bus_data=case.bus_data,
    gen_data=case.gen_data,
    cap_data=case.cap_data,
    reg_data=case.reg_data,
)
# Solve model using provided objective function
result = cvxpy_mi_solve(model, opf.cp_obj_loss)
# result = cvxpy_solve(model, cp_obj_loss)
print(result.fun)
print(result.runtime)
v = model.get_voltages(result.x)
s = model.get_apparent_power_flows(result.x)
dec_var = model.get_q_gens(result.x)
print(model.get_uc(result.x))
print(model.get_zc(result.x))
print(model.get_q_caps(result.x))
opf.plot_network(model, v, s, dec_var, "Q", show_reactive_power=True).show()

# opf.compare_voltages(v, new_v).show(renderer="browser")
# opf.compare_flows(s, new_s).show(renderer="browser")
