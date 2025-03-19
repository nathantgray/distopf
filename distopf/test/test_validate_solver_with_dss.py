import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from distopf import opf_solver, DSSParser
from distopf.lindist_p_gen import LinDistModelPGen
from distopf.lindist_q_gen import LinDistModelQGen
import distopf as opf
from distopf.test import TEST_DIR

# bus_data_path = TEST_DIR / Path("bus_data.csv")
gen_data_path = TEST_DIR / Path("gen_data_2.csv")
# cap_data_path = TEST_DIR / Path("cap_data.csv")
# reg_data_path = TEST_DIR / Path("reg_data.csv")
# assert bus_data_path.exists()


def max_flow_difference(s1: pd.DataFrame, s2: pd.DataFrame) -> float:
    """
    Visually compare voltages by plotting two different results.
    Parameters
    ----------
    s1 : pd.DataFrame
    s2 : pd.DataFrame

    Returns
    -------
    fig : Plotly figure object
    """
    p1 = s1.loc[:, ["tb", "to_name"]].copy()
    p1["id"] = p1.tb
    p1["name"] = p1.to_name
    p1 = p1.loc[:, ["id", "name"]]
    q1 = p1.copy()
    p2 = s2.loc[:, ["tb", "to_name"]].copy()
    p2["id"] = p2.tb
    p2["name"] = p2.to_name
    p2 = p2.loc[:, ["id", "name"]]
    q2 = p2.copy()
    p1.loc[:, ["a", "b", "c"]] = s1.loc[:, ["a", "b", "c"]].apply(np.real)
    q1.loc[:, ["a", "b", "c"]] = s1.loc[:, ["a", "b", "c"]].apply(np.imag)
    p2.loc[:, ["a", "b", "c"]] = s2.loc[:, ["a", "b", "c"]].apply(np.real)
    q2.loc[:, ["a", "b", "c"]] = s2.loc[:, ["a", "b", "c"]].apply(np.imag)
    p1 = p1.melt(
        ignore_index=True, var_name="phase", id_vars=["id", "name"], value_name="p1"
    )
    p2 = p2.melt(
        ignore_index=True, var_name="phase", id_vars=["id", "name"], value_name="p2"
    )
    p1["p1"] = p1["p1"].astype(float)
    p2["p2"] = p2["p2"].astype(float)
    p = pd.merge(p1, p2, on=["name", "phase"])
    p["d"] = np.abs(p.p1 - p.p2)
    p["r"] = p.d / ((p.p1.apply(np.abs) + p.p2.apply(np.abs)) / 2)

    q1 = q1.melt(
        ignore_index=True, var_name="phase", id_vars=["id", "name"], value_name="q1"
    )
    q2 = q2.melt(
        ignore_index=True, var_name="phase", id_vars=["id", "name"], value_name="q2"
    )
    q1["q1"] = q1["q1"].astype(float)
    q2["q2"] = q2["q2"].astype(float)
    q = pd.merge(q1, q2, on=["name", "phase"])
    q.loc[q.q1 < 1e-7, "q1"] = np.nan
    q.loc[q.q2 < 1e-7, "q2"] = np.nan
    q["d"] = q.q1 - q.q2
    q["r"] = q.d / q.q2
    return p.d.abs().max(), q.d.abs().max()


def max_voltage_difference(v1: pd.DataFrame, v2: pd.DataFrame) -> float:
    """
    Visually compare voltages by plotting two different results.
    Parameters
    ----------
    v1 : pd.DataFrame
    v2 : pd.DataFrame

    Returns
    -------
    fig : Plotly figure object
    """
    if "id" not in v1.columns:
        v1["id"] = v1.index
    if "name" not in v1.columns:
        v1["name"] = v1["id"]
    if "id" not in v2.columns:
        v2["id"] = v2.index
    if "name" not in v2.columns:
        v2["name"] = v2["id"]
    v1 = v1.melt(
        ignore_index=True, var_name="phase", id_vars=["id", "name"], value_name="v1"
    )
    v2 = v2.melt(
        ignore_index=True, var_name="phase", id_vars=["id", "name"], value_name="v2"
    )
    v1["v1"] = v1["v1"].astype(float)
    v2["v2"] = v2["v2"].astype(float)
    v = pd.merge(v1, v2, on=["name", "phase"])
    v["d"] = np.abs(v.v1 - v.v2)
    return v.d.abs().max()


def add_generators_from_gen_data(dss_parser: DSSParser, gen_data: pd.DataFrame):
    for i in gen_data.index:
        bus_id = gen_data.loc[i, "id"]
        bus_name = gen_data.loc[i, "name"]
        if isinstance(bus_name, float):
            bus_name = str(int(bus_name))
        s_base = gen_data.loc[i, "s_base"]
        pa = gen_data.loc[i, "pa"] * s_base / 1000
        pb = gen_data.loc[i, "pb"] * s_base / 1000
        pc = gen_data.loc[i, "pc"] * s_base / 1000
        qa = gen_data.loc[i, "qa"] * s_base / 1000
        qb = gen_data.loc[i, "qb"] * s_base / 1000
        qc = gen_data.loc[i, "qc"] * s_base / 1000
        if f"{bus_name}" not in dss_parser.bus_names:
            raise ValueError(
                f"{bus_name} is not a valid bus. Valid bus names include: {dss_parser.bus_names}"
            )
        dss_parser.dss.Text.Command(
            f"New Generator.gen{bus_name}a phases=1 Bus1={bus_name}.1 kV=2.4  kW={pa}  kvar={qa} model=7"
        )
        dss_parser.dss.Text.Command(
            f"New Generator.gen{bus_name}b phases=1 Bus1={bus_name}.2 kV=2.4  kW={pb}  kvar={qb} model=7"
        )
        dss_parser.dss.Text.Command(
            f"New Generator.gen{bus_name}c phases=1 Bus1={bus_name}.3 kV=2.4  kW={pc}  kvar={qc} model=7"
        )
    dss_parser.update()


def add_generators_from_results(
    dss_parser: DSSParser, p_gens: pd.DataFrame, q_gens: pd.DataFrame
):
    for i in p_gens.index:
        bus_id = int(p_gens.loc[i, "id"])
        bus_name = p_gens.loc[i, "name"]
        if isinstance(bus_name, float):
            bus_name = str(int(bus_name))
        pa = p_gens.loc[i, "a"] * 1e6 / 1000
        pb = p_gens.loc[i, "b"] * 1e6 / 1000
        pc = p_gens.loc[i, "c"] * 1e6 / 1000
        qa = q_gens.loc[i, "a"] * 1e6 / 1000
        qb = q_gens.loc[i, "b"] * 1e6 / 1000
        qc = q_gens.loc[i, "c"] * 1e6 / 1000
        if f"{bus_name}" not in dss_parser.bus_names:
            raise ValueError(
                f"{bus_name} is not a valid bus. Valid bus names include: {dss_parser.bus_names}"
            )
        dss_parser.dss.Text.Command(
            f"New Generator.gen{bus_name}a phases=1 Bus1={bus_name}.1 kV=2.4  kW={pa}  kvar={qa} model=7"
        )
        dss_parser.dss.Text.Command(
            f"New Generator.gen{bus_name}b phases=1 Bus1={bus_name}.2 kV=2.4  kW={pb}  kvar={qb} model=7"
        )
        dss_parser.dss.Text.Command(
            f"New Generator.gen{bus_name}c phases=1 Bus1={bus_name}.3 kV=2.4  kW={pc}  kvar={qc} model=7"
        )
    dss_parser.update()


def assert_results_equal(model_new, model_old, res_new, res_old):
    v_old = model_old.get_v_solved(res_old.x)
    v_new = model_new.get_voltages(res_new.x).loc[:, ["a", "b", "c"]]
    # compare_voltages(v_old, v_new).show()
    s_old = model_old.get_s_solved(res_old.x)
    s_new = model_new.get_apparent_power_flows(res_new.x)
    # compare_flows(s_old, s_new).show()
    p_old = np.real(s_old.loc[:, ["a", "b", "c"]])
    p_new = np.real(s_new.loc[:, ["a", "b", "c"]])
    q_old = np.imag(model_old.get_s_solved(res_old.x).loc[:, ["a", "b", "c"]])
    q_new = np.imag(
        model_new.get_apparent_power_flows(res_new.x).loc[:, ["a", "b", "c"]]
    )
    gen_old = model_old.get_dec_variables(res_old.x)
    try:
        gen_new = model_new.get_decision_variables(res_new.x)
    except AttributeError:
        try:
            gen_new = model_new.get_p_gens(res_new.x)
        except AttributeError:
            gen_new = model_new.get_q_gens(res_new.x)
    gen_old = pd.DataFrame(gen_old, columns=["a", "b", "c"])
    gen_old = gen_old.loc[(gen_new.id - 1).to_numpy(), :]
    gen_old.index = gen_old.index + 1
    gen_old["id"] = gen_old.index
    gen_old["name"] = gen_old.index
    assert np.allclose(
        gen_old.loc[:, ["a", "b", "c"]].astype(float).to_numpy(),
        gen_new.loc[:, ["a", "b", "c"]].astype(float).to_numpy(),
        rtol=1.0e-5,
        atol=1.0e-9,
        equal_nan=True,
    )
    assert abs(res_new.fun - res_old.fun) <= 1.0e-6
    assert np.allclose(
        v_old, v_new.astype(float), rtol=1.0e-5, atol=1.0e-9, equal_nan=True
    )
    assert np.allclose(
        p_old, p_new.astype(float), rtol=1.0e-5, atol=1.0e-3, equal_nan=True
    )
    assert np.allclose(
        q_old, q_new.astype(float), rtol=1.0e-5, atol=1.0e-3, equal_nan=True
    )


class TestDssValidation(unittest.TestCase):
    def test_loss(self):
        # bus_data = pd.read_csv(bus_data_path)
        gen_data = pd.read_csv(gen_data_path)
        p_rating_mult = 3
        load_mult = 1
        gen_data.loc[:, ["pa", "pb", "pc"]] *= p_rating_mult
        gen_data.loc[:, ["sa_max", "sb_max", "sc_max"]] *= p_rating_mult

        gen_data.control_variable = "PQ"
        ieee123 = opf.CASES_DIR / "dss/ieee123_dss/Run_IEEE123Bus.DSS"
        dss_parser = opf.DSSParser(ieee123, s_base=1e6, v_min=0, v_max=1.05)
        dss_parser.dss.Solution.LoadMult(load_mult)
        dss_parser.update()
        bus_data = dss_parser.get_bus_data()
        branch_data = dss_parser.get_branch_data()
        cap_data = dss_parser.get_cap_data()
        reg_data = dss_parser.get_reg_data()

        model = LinDistModelQGen(branch_data, bus_data, gen_data, cap_data, reg_data)
        res = opf_solver.cvxpy_solve(model, opf_solver.cp_obj_loss)
        p_gen = model.get_p_gens(res.x)
        q_gen = model.get_q_gens(res.x)
        add_generators_from_results(dss_parser, p_gen, q_gen)
        dss_parser.update()
        v_dss = dss_parser.v_solved
        v = model.get_voltages(res.x)
        s_dss = dss_parser.s_solved
        s = model.get_apparent_power_flows(res.x)
        vdiff = max_voltage_difference(v, v_dss)
        pdiff, qdiff = max_flow_difference(s, s_dss)
        print(f"vdiff = {vdiff}\tpdiff = {pdiff}\tqdiff = {qdiff}")
        assert vdiff < 0.005

    def test_cp_obj_target_q_3ph(self):
        gen_data = pd.read_csv(gen_data_path)
        p_rating_mult = 4
        load_mult = 1
        target_per_phase = 0.2
        gen_data.loc[:, ["pa", "pb", "pc"]] *= p_rating_mult
        gen_data.loc[:, ["sa_max", "sb_max", "sc_max"]] *= p_rating_mult

        gen_data.control_variable = "PQ"
        ieee123 = opf.CASES_DIR / "dss/ieee123_dss/Run_IEEE123Bus.DSS"
        dss_parser = opf.DSSParser(ieee123, s_base=1e6, v_min=0, v_max=1.05)
        dss_parser.dss.Solution.LoadMult(load_mult)
        dss_parser.update()
        bus_data = dss_parser.get_bus_data()
        branch_data = dss_parser.get_branch_data()
        cap_data = dss_parser.get_cap_data()
        reg_data = dss_parser.get_reg_data()

        model = LinDistModelQGen(branch_data, bus_data, gen_data, cap_data, reg_data)
        target = np.array([target_per_phase, target_per_phase, target_per_phase])
        res = opf_solver.cvxpy_solve(
            model,
            opf_solver.cp_obj_target_q_3ph,
            target=target,
            error_percent=np.array([0.1, 0.1, 0.1]),
        )
        p_gen = model.get_p_gens(res.x)
        q_gen = model.get_q_gens(res.x)
        add_generators_from_results(dss_parser, p_gen, q_gen)
        dss_parser.update()
        v_dss = dss_parser.v_solved
        v = model.get_voltages(res.x)
        s_dss = dss_parser.s_solved
        s = model.get_apparent_power_flows(res.x)
        vdiff = max_voltage_difference(v, v_dss)
        pdiff, qdiff = max_flow_difference(s, s_dss)
        print(f"vdiff = {vdiff}\tpdiff = {pdiff}\tqdiff = {qdiff}")
        assert vdiff < 0.005

    def test_cp_obj_target_q_total(self):
        gen_data = pd.read_csv(gen_data_path)
        p_rating_mult = 4
        load_mult = 1
        target_per_phase = 0
        gen_data.loc[:, ["pa", "pb", "pc"]] *= p_rating_mult
        gen_data.loc[:, ["sa_max", "sb_max", "sc_max"]] *= p_rating_mult

        gen_data.control_variable = "PQ"
        ieee123 = opf.CASES_DIR / "dss/ieee123_dss/Run_IEEE123Bus.DSS"
        dss_parser = opf.DSSParser(ieee123, s_base=1e6, v_min=0, v_max=1.05)
        dss_parser.dss.Solution.LoadMult(load_mult)
        dss_parser.update()
        bus_data = dss_parser.get_bus_data()
        branch_data = dss_parser.get_branch_data()
        cap_data = dss_parser.get_cap_data()
        reg_data = dss_parser.get_reg_data()

        model = LinDistModelQGen(branch_data, bus_data, gen_data, cap_data, reg_data)
        target = target_per_phase * 3
        res = opf_solver.cvxpy_solve(
            model,
            opf_solver.cp_obj_target_q_total,
            target=target,
            error_percent=np.array([0.1, 0.1, 0.1]),
        )
        p_gen = model.get_p_gens(res.x)
        q_gen = model.get_q_gens(res.x)
        add_generators_from_results(dss_parser, p_gen, q_gen)
        dss_parser.update()
        v_dss = dss_parser.v_solved
        v = model.get_voltages(res.x)
        s_dss = dss_parser.s_solved
        s = model.get_apparent_power_flows(res.x)
        vdiff = max_voltage_difference(v, v_dss)
        pdiff, qdiff = max_flow_difference(s, s_dss)
        print(f"vdiff = {vdiff}\tpdiff = {pdiff}\tqdiff = {qdiff}")
        assert vdiff < 0.005

    def test_cp_obj_target_p_3ph(self):
        area_dir = Path("./")
        assert area_dir.exists()
        gen_data = pd.read_csv(gen_data_path)
        p_rating_mult = 5
        load_mult = 0.5
        gen_data.loc[:, ["pa", "pb", "pc"]] *= p_rating_mult
        gen_data.loc[:, ["sa_max", "sb_max", "sc_max"]] *= p_rating_mult
        gen_data.control_variable = "PQ"
        ieee123 = opf.CASES_DIR / "dss/ieee123_dss/Run_IEEE123Bus.DSS"
        dss_parser = opf.DSSParser(ieee123, s_base=1e6, v_min=0, v_max=1.05)
        dss_parser.dss.Solution.LoadMult(load_mult)
        dss_parser.update()
        bus_data = dss_parser.get_bus_data()
        branch_data = dss_parser.get_branch_data()
        cap_data = dss_parser.get_cap_data()
        reg_data = dss_parser.get_reg_data()

        model = LinDistModelPGen(branch_data, bus_data, gen_data, cap_data, reg_data)
        res = opf_solver.cvxpy_solve(
            model,
            opf_solver.cp_obj_target_p_3ph,
            target=np.array([0.3, 0.3, 0.3]),
            error_percent=np.array([0.1, 0.1, 0.1]),
        )
        p_gen = model.get_p_gens(res.x)
        q_gen = model.get_q_gens(res.x)
        add_generators_from_results(dss_parser, p_gen, q_gen)
        dss_parser.update()
        v_dss = dss_parser.v_solved
        v = model.get_voltages(res.x)
        s_dss = dss_parser.s_solved
        s = model.get_apparent_power_flows(res.x)
        vdiff = max_voltage_difference(v, v_dss)
        pdiff, qdiff = max_flow_difference(s, s_dss)
        print(f"vdiff = {vdiff}\tpdiff = {pdiff}\tqdiff = {qdiff}")
        assert vdiff < 0.005

    def test_cp_obj_target_p_total(self):
        area_dir = Path("./")
        assert area_dir.exists()
        gen_data = pd.read_csv(gen_data_path)
        target = 0
        p_rating_mult = 5
        load_mult = 0.5
        gen_data.loc[:, ["pa", "pb", "pc"]] *= p_rating_mult
        gen_data.loc[:, ["sa_max", "sb_max", "sc_max"]] *= p_rating_mult
        gen_data.control_variable = "PQ"
        ieee123 = opf.CASES_DIR / "dss/ieee123_dss/Run_IEEE123Bus.DSS"
        dss_parser = opf.DSSParser(ieee123, s_base=1e6, v_min=0, v_max=1.05)
        dss_parser.dss.Solution.LoadMult(load_mult)
        dss_parser.update()
        bus_data = dss_parser.get_bus_data()
        branch_data = dss_parser.get_branch_data()
        cap_data = dss_parser.get_cap_data()
        reg_data = dss_parser.get_reg_data()
        model = LinDistModelPGen(branch_data, bus_data, gen_data, cap_data, reg_data)
        res = opf_solver.cvxpy_solve(
            model,
            opf_solver.cp_obj_target_p_total,
            target=target,
            error_percent=np.array([0.1, 0.1, 0.1]),
        )
        p_gen = model.get_p_gens(res.x)
        q_gen = model.get_q_gens(res.x)
        add_generators_from_results(dss_parser, p_gen, q_gen)
        dss_parser.update()
        v_dss = dss_parser.v_solved
        v = model.get_voltages(res.x)
        s_dss = dss_parser.s_solved
        s = model.get_apparent_power_flows(res.x)
        vdiff = max_voltage_difference(v, v_dss)
        pdiff, qdiff = max_flow_difference(s, s_dss)
        print(f"vdiff = {vdiff}\tpdiff = {pdiff}\tqdiff = {qdiff}")
        assert vdiff < 0.005

    def test_cp_obj_quadratic_curtail(self):
        gen_data = pd.read_csv(opf.CASES_DIR / "csv/ieee123_30der/gen_data.csv")
        gen_data.loc[:, ["pa", "pb", "pc"]] *= 10
        gen_data.loc[:, ["sa_max", "sb_max", "sc_max"]] *= 10
        gen_data.control_variable = "PQ"
        ieee123 = opf.CASES_DIR / "dss/ieee123_dss/Run_IEEE123Bus.DSS"
        dss_parser = opf.DSSParser(ieee123, s_base=1e6, v_min=0, v_max=1.05)
        dss_parser.dss.Solution.LoadMult(0.1)
        dss_parser.update()
        bus_data = dss_parser.get_bus_data()
        branch_data = dss_parser.get_branch_data()
        cap_data = dss_parser.get_cap_data()
        reg_data = dss_parser.get_reg_data()
        # add_generators_from_gen_data(dss_parser, gen_data)
        model = LinDistModelPGen(branch_data, bus_data, gen_data, cap_data, reg_data)
        res = opf_solver.cvxpy_solve(model, opf_solver.cp_obj_curtail_lp)
        p_gen = model.get_p_gens(res.x)
        q_gen = model.get_q_gens(res.x)
        add_generators_from_results(dss_parser, p_gen, q_gen)
        dss_parser.update()
        v_dss = dss_parser.v_solved
        v = model.get_voltages(res.x)
        s_dss = dss_parser.s_solved
        s = model.get_apparent_power_flows(res.x)
        vdiff = max_voltage_difference(v, v_dss)
        pdiff, qdiff = max_flow_difference(s, s_dss)
        print(f"vdiff = {vdiff}\tpdiff = {pdiff}\tqdiff = {qdiff}")
        assert vdiff < 0.005
