import pandas as pd
from distopf.multiperiod.lindist_multi_all import LinDistModel
from distopf.multiperiod.lindist_q_multi import LinDistModelQ
from distopf.multiperiod.lindist_p_multi import LinDistModelP
import distopf as opf
import plotly.graph_objects as go
import plotly.express as px


branch_data = pd.read_csv("../distopf/cases/csv/ieee123_alternate/branch_data.csv")
bus_data = pd.read_csv("../distopf/cases/csv/ieee123_alternate/bus_data.csv")
gen_data = pd.read_csv("../distopf/cases/csv/ieee123_alternate/gen_data.csv")
reg_data = pd.read_csv("../distopf/cases/csv/ieee123_alternate/reg_data.csv")
cap_data = pd.read_csv("../distopf/cases/csv/ieee123_alternate/cap_data.csv")
battery_data = pd.read_csv("../distopf/cases/csv/ieee123_alternate/battery_data.csv")
pv_loadshape = pd.read_csv("../distopf/cases/csv/ieee123_alternate/pv_loadshape.csv")
default_loadshape = pd.read_csv("../distopf/cases/csv/ieee123_alternate/default_loadshape.csv")
# bus_data.v_min = 0
bus_data.v_max = 1.07
bus_data.v_a = 1.0
bus_data.v_b = 1.0
bus_data.v_c = 1.0
m = LinDistModelQ(
    branch_data=branch_data,
    bus_data=bus_data,
    gen_data=gen_data,
    reg_data=reg_data,
    cap_data=cap_data,
    loadshape_data=default_loadshape,
    pv_loadshape_data=pv_loadshape,
    bat_data=battery_data,
    n=24,
    battery=True,
)
result = opf.multiperiod.opf_solver_multi.cvxpy_solve(m, opf.multiperiod.cp_obj_loss_batt)
v = m.get_voltages(result.x)
v_df = m.get_voltages_new(result.x)
v_df.to_csv("mp_output/v.csv", index=False)
# v_df = v_df.melt(
#     id_vars=["id", "name", "t"],
#     value_vars=["a", "b", "c"],
#     value_name="value",
#     var_name="phase",
# )
# px.line(v_df, x="t", y="value", facet_col="phase", color="name").write_html(
#     "mp_output/v.html"
# )
print(v_df.head())
s = m.get_apparent_power_flows(result.x)
s_df = m.get_apparent_power_flows_new(result.x)
s_df.to_csv("mp_output/s.csv", index=False)
# s_df = s_df.melt(
#     id_vars=["fb", "tb", "t"],
#     value_vars=["a", "b", "c"],
#     value_name="value",
#     var_name="phase",
# )
# px.line(s_df, x="t", y="value", facet_col="phase", color="name").write_html(
#     "mp_output/s.html"
# )
q_der, batt_charge, batt_discharge, batt_soc = m.get_decision_variables(result.x)
q_der_list = []
for t in q_der.keys():
    q_der[t]["t"] = t
    q_der[t]["id"] = q_der[t].index
    q_der_list.append(q_der[t])
batt_charge_list = []
for t in batt_charge.keys():
    batt_charge[t]["t"] = t
    batt_charge[t]["id"] = batt_charge[t].index
    batt_charge_list.append(batt_charge[t])
batt_discharge_list = []
for t in batt_discharge.keys():
    batt_discharge[t]["t"] = t
    batt_discharge[t]["id"] = batt_discharge[t].index
    batt_discharge_list.append(batt_discharge[t])
batt_soc_list = []
for t in batt_soc.keys():
    batt_soc[t]["t"] = t
    batt_soc[t]["id"] = batt_soc[t].index
    batt_soc_list.append(batt_soc[t])
q_der = pd.concat(q_der_list, axis=0).reset_index(drop=True)
batt_charge = pd.concat(batt_charge_list, axis=0).reset_index(drop=True)
batt_discharge = pd.concat(batt_discharge_list, axis=0).reset_index(drop=True)
batt_soc = pd.concat(batt_soc_list, axis=0).reset_index(drop=True)
q_der.to_csv("mp_output/q_der.csv", index=False)
batt_charge.to_csv("mp_output/batt_charge.csv", index=False)
batt_discharge.to_csv("mp_output/batt_discharge.csv", index=False)
batt_soc.to_csv("mp_output/batt_soc.csv", index=False)
# q_der = q_der.melt(
#     id_vars=["id", "name", "t"],
#     value_vars=["a", "b", "c"],
#     value_name="value",
#     var_name="phase",
# )
# batt_charge = batt_charge.melt(
#     id_vars=["id", "name", "t"],
#     value_vars=["a", "b", "c"],
#     value_name="value",
#     var_name="phase",
# )
# batt_discharge = batt_discharge.melt(
#     id_vars=["id", "name", "t"],
#     value_vars=["a", "b", "c"],
#     value_name="value",
#     var_name="phase",
# )
# batt_soc = batt_soc.melt(
#     id_vars=["id", "name", "t"],
#     value_vars=["a", "b", "c"],
#     value_name="value",
#     var_name="phase",
# )
# print(f"{q_der=}")
# print(f"{batt_charge=}")
# print(f"{batt_discharge=}")
# print(f"{batt_soc=}")
# px.line(q_der, x="t", y="value", facet_col="phase", color="name").write_html(
#     "mp_output/q_der.html"
# )
# px.line(batt_charge, x="t", y="value", facet_col="phase", color="name").write_html(
#     "mp_output/batt_charge.html"
# )
# px.line(batt_discharge, x="t", y="value", facet_col="phase", color="name").write_html(
#     "mp_output/batt_discharge.html"
# )
# px.line(batt_soc, x="t", y="value", facet_col="phase", color="name").write_html(
#     "mp_output/batt_soc.html"
# )


print(f"{q_der=}")
print(f"{batt_charge=}")
print(f"{batt_discharge=}")
print(f"{batt_soc=}")
print(f"{result.runtime}")
pass
