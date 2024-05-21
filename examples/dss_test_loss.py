import plotly.express as px
import pandas as pd
import numpy as np
import distopf as opf


test2x1 = opf.CASES_DIR / "dss/2Bus_1phase/2Bus1ph.DSS"
test2 = opf.CASES_DIR / "dss/2Bus/2Bus.DSS"
test2D = opf.CASES_DIR / "dss/2BusD/2Bus.DSS"
test3 = opf.CASES_DIR / "dss/3Bus/3Bus.DSS"
ieee4 = opf.CASES_DIR / "dss/4Bus-YY-Bal/4Bus-YY-Bal.DSS"
ieee4YD = opf.CASES_DIR / "dss/4Bus-YD-Bal/4Bus-YD-Bal.DSS"
ieee13 = opf.CASES_DIR / "dss/ieee13_dss/IEEE13Nodeckt.dss"
ieee34 = opf.CASES_DIR / "dss/34Bus/Run_IEEE34Mod2.dss"
ieee123 = opf.CASES_DIR / "dss/ieee123_dss/Run_IEEE123Bus.DSS"
rahul123 = opf.CASES_DIR / "dss/rahul123/ieee123master_base.dss"
dirs = [
    test2,
    test2D,
    test3,
    ieee4,
    ieee4YD,
    ieee13,
    ieee34,
    ieee123,
    rahul123,
]
# ~~~~~ Rahul's 123
loss_list = []
for _dir in dirs:
    print(_dir)
    for mult in np.linspace(0, 1, 6):
        dss = opf.DSSParser(_dir, s_base=1e6, v_min=0, v_max=2)
        dss.dss.Solution.LoadMult(mult)
        dss.dss.Solution.Solve()
        dss.update()
        model = opf.LinDistModel(
            dss.branch_data, dss.bus_data, cap_data=dss.cap_data, reg_data=dss.reg_data
        )
        s_loss = np.array(dss.dss.Circuit.Losses()) / 1e6
        s_total = -np.array(dss.dss.Circuit.TotalPower()) * 1e3 / 1e6
        loss_percent = s_loss / s_total * 100
        try:
            result = opf.lp_solve(model, np.zeros(model.n_x))
        except:
            continue
        v_df = model.get_voltages(result.x)
        s_df = model.get_apparent_power_flows(result.x)
        dec_var_df = model.get_decision_variables(result.x)
        v_diff = v_df.copy()
        v_diff.loc[:, ["a", "b", "c"]] = (
            v_df.loc[:, ["a", "b", "c"]] - dss.v_solved.loc[:, ["a", "b", "c"]].abs()
        )
        v_rdiff = (
            v_diff.loc[:, ["a", "b", "c"]] / dss.v_solved.loc[:, ["a", "b", "c"]].abs()
        )
        print(f"{mult:.1f}: V error %: {v_rdiff.max().max():.3e}")
        loss_list.append(
            {
                "name": _dir.parent.name,
                "mult": mult,
                "%Ploss": s_loss[0],
                "%Qloss": s_loss[1],
                "%V_err": v_rdiff.max().max() * 100,
            }
        )
loss_df = pd.DataFrame(loss_list)
fig = px.scatter(loss_df, x="%Ploss", y="%V_err", color="name")
fig.show(renderer="browser")
fig = px.scatter(loss_df, x="%Qloss", y="%V_err", color="name")
fig.show(renderer="browser")
