import numpy as np
import distopf as opf
from distopf.lindist import LinDistModel
case = opf.DistOPFCase(
    data_path="ieee123_30der", gen_mult=1, load_mult=1, v_swing=1.0, v_max=1.05, v_min=0.95
)

model = opf.LinDistModel(
    branch_data=case.branch_data,
    bus_data=case.bus_data,
    gen_data=case.gen_data,
    cap_data=case.cap_data,
    reg_data=case.reg_data,
)
# Solve model using provided objective function
result = opf.lp_solve(model, opf.gradient_load_min(model))
# result = cvxpy_solve(model, cp_obj_loss)
print(result.fun)
v = model.get_voltages(result.x)
s = model.get_apparent_power_flows(result.x)
p_gens = model.get_p_gens(result.x)
q_gens = model.get_q_gens(result.x)
# opf.plot_network(model, v=v, s=s, p_gen=None, q_gen=q_gens).show()
opf.plot_network(model).show()
# opf.plot_voltages(v).show()
# opf.plot_power_flows(s).show()

# fig.write_image("123_plot.pdf", format="pdf", width=1000, height=1000)
# import time

# time.sleep(1)
# fig.write_image("123_plot.pdf", format="pdf", width=1000, height=1000)
