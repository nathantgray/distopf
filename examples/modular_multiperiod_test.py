import pandas as pd

# from distopf.multiperiod.lindist_multi_all import LinDistModel
# from distopf.multiperiod.lindist_q_multi import LinDistModelQ
# from distopf.multiperiod.lindist_p_multi import LinDistModelP
from distopf.multiperiod.lindist_base_modular_multi import LinDistModelModular
import distopf as opf
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

# dir_path = Path("../distopf/cases/csv/2Bus-1ph-batt").expanduser()
dir_path = Path("../distopf/cases/csv/ieee123_alternate").expanduser()

branch_data =       pd.read_csv(dir_path / "branch_data.csv")
bus_data =          pd.read_csv(dir_path / "bus_data.csv")
gen_data =          pd.read_csv(dir_path / "gen_data.csv")
reg_data =          pd.read_csv(dir_path / "reg_data.csv")
cap_data =          pd.read_csv(dir_path / "cap_data.csv")
battery_data =      pd.read_csv(dir_path / "battery_data.csv")
pv_loadshape =      pd.read_csv(dir_path / "pv_loadshape.csv")
default_loadshape = pd.read_csv(dir_path / "default_loadshape.csv")
# bus_data.v_min = 0.95
bus_data.v_max = 1.05
bus_data.v_a = 1.0
bus_data.v_b = 1.0
bus_data.v_c = 1.0
# bus_data.pl_a *= 0.25
# bus_data.pl_b *= 0.25
# bus_data.pl_c *= 0.25
# bus_data.ql_a *= 0.25
# bus_data.ql_b *= 0.25
# bus_data.ql_c *= 0.25
gen_data["a_mode"] = opf.CONSTANT_P
gen_data["b_mode"] = opf.CONSTANT_P
gen_data["c_mode"] = opf.CONSTANT_P
m = LinDistModelModular(
    branch_data=branch_data,
    bus_data=bus_data,
    gen_data=gen_data,
    reg_data=reg_data,
    cap_data=cap_data,
    loadshape_data=default_loadshape,
    pv_loadshape_data=pv_loadshape,
    bat_data=battery_data,
    n_steps=24,
)
result = opf.opf_solver.cvxpy_solve(m, opf.multiperiod.opf_solver_multi.cp_obj_loss)
v_df = m.get_voltages(result.x)
v_df.to_csv("mp_output_new/v.csv", index=False)
v_df = v_df.melt(
    id_vars=["id", "name", "t"],
    value_vars=["a", "b", "c"],
    value_name="value",
    var_name="phase",
)
px.line(v_df, x="t", y="value", facet_col="phase", color="name").write_html(
    "mp_output_new/v.html"
)
print(v_df.head())
s_df = m.get_apparent_power_flows(result.x)
s_df.to_csv("mp_output_new/s.csv", index=False)
# s_df = s_df.melt(
#     id_vars=["fb", "tb", "from_name", "to_name", "t"],
#     value_vars=["a", "b", "c"],
#     value_name="value",
#     var_name="phase",
# )
# px.line(s_df, x="t", y="value", facet_col="phase", color="to_name").write_html(
#     "mp_output_new/s.html"
# )
# q_der, batt_charge, batt_discharge, batt_soc = m.get_decision_variables(result.x)
p_der = m.get_p_gens(result.x)
q_der = m.get_q_gens(result.x)
batt_charge = m.get_p_charge(result.x)
batt_discharge = m.get_p_discharge(result.x)
batt_soc = m.get_soc(result.x)

p_der.to_csv("mp_output_new/p_der.csv")
q_der.to_csv("mp_output_new/q_der.csv", index=False)
batt_charge.to_csv("mp_output_new/batt_charge.csv", index=False)
batt_discharge.to_csv("mp_output_new/batt_discharge.csv", index=False)
batt_soc.to_csv("mp_output_new/batt_soc.csv", index=False)
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
#     "mp_output_new/q_der.html"
# )
# px.line(batt_charge, x="t", y="value", facet_col="phase", color="name").write_html(
#     "mp_output_new/batt_charge.html"
# )
# px.line(batt_discharge, x="t", y="value", facet_col="phase", color="name").write_html(
#     "mp_output_new/batt_discharge.html"
# )
# px.line(batt_soc, x="t", y="value", facet_col="phase", color="name").write_html(
#     "mp_output_new/batt_soc.html"
# )


print(f"{q_der=}")
print(f"{batt_charge=}")
print(f"{batt_discharge=}")
print(f"{batt_soc=}")
print(f"{result.runtime}")
pass
