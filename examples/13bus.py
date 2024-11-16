import numpy as np
import distopf as opf

case = opf.DistOPFCase(
    data_path="ieee13",
    output_dir="ieee13_output",
    gen_mult=1,
    load_mult=1,
    v_swing=1.0,
    v_max=1.05,
    v_min=0.95,
)

model = opf.create_model(
    control_variable="Q",
    branch_data=case.branch_data,
    bus_data=case.bus_data,
    gen_data=case.gen_data,
    cap_data=case.cap_data,
    reg_data=case.reg_data,
)
result = opf.lp_solve(model, np.zeros(model.n_x))
print(result.fun)
v = model.get_voltages(result.x)
s = model.get_apparent_power_flows(result.x)
dec_var = model.get_decision_variables(result.x)
opf.plot_network(model, v, s, dec_var, "Q").show(renderer="browser")
opf.plot_voltages(v).show(renderer="browser")
opf.plot_power_flows(s).show(renderer="browser")
