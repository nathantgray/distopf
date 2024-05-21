import distopf as opf
import numpy as np


ieee123 = opf.CASES_DIR / "dss/ieee123_dss/Run_IEEE123Bus.DSS"
print(ieee123)
dss_parser = opf.DSSParser(ieee123, s_base=1e6, v_min=0, v_max=1.05)
dss_parser.dss.Text.Command(
    "New Generator.gen1a phases=1 Bus1=13.1 kV=2.4  kW=1200  kvar=0 model=7"
)
dss_parser.dss.Text.Command(
    "New Generator.gen1b phases=1 Bus1=13.2 kV=2.4  kW=1200  kvar=0 model=7"
)
dss_parser.dss.Text.Command(
    "New Generator.gen1c phases=1 Bus1=13.3 kV=2.4  kW=1200  kvar=0 model=7"
)
dss_parser.update()
model = opf.LinDistModelQ(
    dss_parser.branch_data,
    dss_parser.bus_data,
    gen_data=dss_parser.gen_data,
    cap_data=dss_parser.cap_data,
    reg_data=dss_parser.reg_data,
)

result = opf.cvxpy_solve(model, opf.cp_obj_loss, solver="CLARABEL")
# print(result.fun)
v_df = model.get_voltages(result.x)
s_df = model.get_apparent_power_flows(result.x)
dec_var_df = model.get_decision_variables(result.x)
dss_parser.update_gen_q(dec_var_df)
dss_parser.dss.Text.Command("Set Controlmode=OFF")
dss_parser.update()
s_loss = np.array(dss_parser.dss.Circuit.Losses()) / 1e6
s_total = -np.array(dss_parser.dss.Circuit.TotalPower()) * 1e3 / 1e6
loss_percent = s_loss / s_total * 100
print(loss_percent)
print("Maximum voltage error %:")
v_diff = v_df.copy()
v_diff.loc[:, ["a", "b", "c"]] = (
    v_df.loc[:, ["a", "b", "c"]] - dss_parser.v_solved.loc[:, ["a", "b", "c"]].abs()
)
v_rdiff = (
    v_diff.loc[:, ["a", "b", "c"]] / dss_parser.v_solved.loc[:, ["a", "b", "c"]].abs()
)
print(f"{v_rdiff.max().max()*100}%")

# plot.plot_voltages(v_df).show(renderer="browser")
# plot.plot_power_flows(s_df).show(renderer="browser")
# plot.compare_flows(s_df, dss_parser.s_solved).show(renderer="browser")
# plot.compare_voltages(v_df, dss_parser.v_solved).show(renderer="browser")
# plot.plot_ders(dec_var_df).show(renderer="browser")
fig = opf.plot_network(
    model,
    v=v_df,
    s=s_df,
    show_phases="abc",
    show_reactive_power=False,
)
fig.show(renderer="browser")
