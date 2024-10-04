import distopf as opf
import numpy as np


if __name__ == "__main__":

    case = opf.DistOPFCase(
        data_path="ieee123_30der",
        gen_mult=1,
        load_mult=1,
        v_swing=1.0,
        v_max=1.05,
        v_min=0.95,
    )
    # reg_data = pd.concat(
    #     [
    #         case.reg_data,
    #         pd.DataFrame(
    #             {
    #                 "fb": [128],
    #                 "tb": [127],
    #                 "phases": ["abc"],
    #                 "tap_a": [15.0],
    #                 "tap_b": [2.0],
    #                 "tap_c": [5.0],
    #             }
    #         ),
    #     ]
    # )
    # case.cap_data = pd.concat(
    #     [
    #         case.cap_data,
    #         pd.DataFrame(
    #             {
    #                 "id": [14],
    #                 "name": [632],
    #                 "qa": [0.3],
    #                 "qb": [0.3],
    #                 "qc": [0.5],
    #                 "phases": ["abc"],
    #             }
    #         ),
    #     ]
    # )
    case.gen_data.a_mode = "CONTROL_PQ"
    case.gen_data.b_mode = "CONTROL_PQ"
    case.gen_data.c_mode = "CONTROL_PQ"
    case.gen_data.sa_max = case.gen_data.sa_max / 1.2
    case.gen_data.sb_max = case.gen_data.sb_max / 1.2
    case.gen_data.sc_max = case.gen_data.sc_max / 1.2

    m = opf.LinDistModelCapacitorRegulatorMI(
        branch_data=case.branch_data,
        bus_data=case.bus_data,
        gen_data=case.gen_data,
        cap_data=case.cap_data,
        reg_data=case.reg_data,
    )
    result = m.solve(opf.cp_obj_loss)

    print(f" objective={result.fun}, time={result.runtime}")
    print(m.u_reg.value @ np.array([m.b_i]).T)
    v = m.get_voltages(result.x)
    s = m.get_apparent_power_flows(result.x)
    pg = m.get_p_gens(result.x)
    qg = m.get_q_gens(result.x)
    opf.plot_network(m, v, s, qg, "Q").show(renderer="browser")
    opf.plot_voltages(v).show(renderer="browser")
    opf.plot_power_flows(s).show(renderer="browser")
    opf.plot_ders(pg).show(renderer="browser")
    opf.plot_ders(qg).show(renderer="browser")
    opf.plot.plot_polar(pg, qg).show(renderer="browser")
