import numpy as np
import distopf as opf
from distopf.lindist_base_modular import LinDistModel
from distopf.opf_solver import cvxpy_mi_solve
import pandas as pd

from time import perf_counter

case = opf.DistOPFCase(
    data_path="ieee123_30der", gen_mult=1, load_mult=1, v_swing=1.0, v_max=1.05, v_min=0.95
)
# reg_data = pd.concat([case.reg_data, pd.DataFrame({
# "fb": [128], "tb": [127], "phases": ["abc"], "tap_a": [15.0], "tap_b": [2.0], "tap_c": [5.0]})])
case.cap_data = pd.concat([case.cap_data, pd.DataFrame({"id": [14],"name": [632],"qa": [0.3],"qb": [0.3],"qc": [0.5],"phases": ["abc"]})])
m = LinDistModel(
    branch_data=case.branch_data,
    bus_data=case.bus_data,
    gen_data=case.gen_data,
    cap_data=case.cap_data,
    reg_data=case.reg_data
)
repeat = 20
tic = perf_counter()
for i in range(repeat):
    a_eq, b_eq = m.create_model()
print(f"Original model creation time: {(perf_counter() - tic)/repeat:.4f} seconds.")