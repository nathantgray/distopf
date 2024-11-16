import pandas as pd

from distopf.multiperiod import LinDistModelQ
from distopf.multiperiod import cvxpy_solve, cp_obj_loss
from distopf import CASES_DIR

branch_data = pd.read_csv(f"{CASES_DIR}/csv/prev_123/branchdata.csv", header=0)
bus_data = pd.read_csv(f"{CASES_DIR}/csv/prev_123/bus_data.csv", header=0)
gen_data = pd.read_csv(f"{CASES_DIR}/csv/prev_123/gen_data.csv", header=0)
cap_data = pd.read_csv(f"{CASES_DIR}/csv/prev_123/cap_data.csv", header=0)
reg_data = pd.read_csv(f"{CASES_DIR}/csv/prev_123/reg_data.csv", header=0)
loadshape_data = pd.read_csv(
    f"{CASES_DIR}/csv/prev_123/default_loadshape.csv", header=0
)
pv_loadshape_data = pd.read_csv(f"{CASES_DIR}/csv/prev_123/pv_loadshape.csv", header=0)
bat_data = pd.read_csv(f"{CASES_DIR}/csv/prev_123/battery_data.csv", header=0)
# modify generator power ratings to be 5x larger (alternatively, could modify csv directly or create helper function)
gen_data.loc[:, ["pa", "pb", "pc"]] *= 3
gen_data.loc[:, ["sa_max", "sb_max", "sc_max"]] *= 3
# bat_data.loc[:,["Pb_max_a","Pb_max_b","Pb_max_c","hmax_a","hmax_b","hmax_c","bmin_a","bmin_b","bmin_c","bmax_a","bmax_b","bmax_c"]] *=4
bat_data.loc[:, ["Pb_max_a", "Pb_max_b", "Pb_max_c", "hmax_a", "hmax_b", "hmax_c"]] *= 1
bat_data.loc[:, ["bmin_a", "bmin_b", "bmin_c", "bmax_a", "bmax_b", "bmax_c"]] *= 4
# Initialize model (create matrices, bounds, indexing methods, etc.)
model = LinDistModelQ(
    branch_data,
    bus_data,
    gen_data,
    cap_data,
    None,
    loadshape_data,
    pv_loadshape_data,
    bat_data,
    der=True,
    battery=True,
)

print(5)
# Solve model using provided objective function
res = cvxpy_solve(model, cp_obj_loss)
# res=solve_lin(model,np.zeros(model.n_x))
print(res.fun)
v = model.get_voltages(res.x)
print(v)
power = model.get_apparent_power_flows(res.x)
print(power)
dec = model.get_decision_variables(res.x)
print(dec)
