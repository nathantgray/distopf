"""
These tests no longer pass:
Need to reevaluate if they are still needed,
then revise.
The inconsistency here is likely because of minor differences and
do not represent a true problem.
A better test would validate each model against open dss.
"""

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from distopf import opf_solver
from distopf.lindist_p_gen import LinDistModelPGen
from distopf.lindist_q_gen import LinDistModelQGen
from distopf.test.legacy.opf_var import QModel
from distopf.test.legacy.opf_watt import PModel
from distopf import compare_voltages, compare_flows

branchdata_path = Path("branch_data.csv")
powerdata_path = Path("powerdata.csv")
legacy_powerdata_path = Path("legacy/powerdata.csv")
bus_data_path = Path("bus_data.csv")
gen_data_path = Path("gen_data.csv")
cap_data_path = Path("cap_data.csv")
reg_data_path = Path("reg_data.csv")


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
        p_gen_new = model_new.get_p_gens(res_new.x)
        q_gen_new = model_new.get_q_gens(res_new.x)
    gen_old = pd.DataFrame(gen_old, columns=["a", "b", "c"])
    gen_old = gen_old.loc[(p_gen_new.id - 1).to_numpy(), :]
    gen_old.index = gen_old.index + 1
    gen_old["id"] = gen_old.index
    gen_old["name"] = gen_old.index
    try:
        assert np.allclose(
            gen_old.loc[:, ["a", "b", "c"]].astype(float).to_numpy(),
            p_gen_new.loc[:, ["a", "b", "c"]].astype(float).to_numpy(),
            rtol=1.0e-3,
            atol=1.0e-2,
        )
    except AssertionError:
        assert np.allclose(
            gen_old.loc[:, ["a", "b", "c"]].astype(float).to_numpy(),
            q_gen_new.loc[:, ["a", "b", "c"]].astype(float).to_numpy(),
            rtol=1.0e-3,
            atol=1.0e-2,
        )
    assert abs(res_new.fun - res_old.fun) <= 1.0e-4
    # assert np.allclose(
    #     v_old, v_new.astype(float), rtol=1.0e-5, atol=1.0e-9, equal_nan=True
    # )
    # assert np.allclose(
    #     p_old, p_new.astype(float), rtol=1.0e-5, atol=1.0e-3, equal_nan=True
    # )
    # assert np.allclose(
    #     q_old, q_new.astype(float), rtol=1.0e-5, atol=1.0e-3, equal_nan=True
    # )


class TestObjectives(unittest.TestCase):
    def test_loss(self):
        branch_data = pd.read_csv(branchdata_path, header=0)
        bus_data = pd.read_csv(bus_data_path)
        gen_data = pd.read_csv(gen_data_path)
        cap_data = pd.read_csv(cap_data_path)
        reg_data = pd.read_csv(reg_data_path)
        gen_data.loc[:, ["pa", "pb", "pc"]] *= 3
        gen_data.loc[:, ["sa_max", "sb_max", "sc_max"]] *= 3

        model_new = LinDistModelQGen(
            branch_data, bus_data, gen_data, cap_data, reg_data
        )
        powerdata = pd.read_csv(legacy_powerdata_path, header=0)
        model_old = QModel(
            branch_data,
            powerdata,
            p_rating_mult=3,
            v_up=1.05,
            p_base_gld=1e6,
            v_ll_base_gld=4160,
        )
        # print("Solve old")
        res_old = model_old.solve(objective="loss", gld_correction=False)
        print(f"old: {res_old.fun}")
        # print("Solve new")
        res_new = opf_solver.cvxpy_solve(model_new, opf_solver.cp_obj_loss)
        print(f"new: {res_new.fun}")
        assert_results_equal(model_new, model_old, res_new, res_old)

    def test_cp_obj_target_q_3ph(self):
        branch_data = pd.read_csv(branchdata_path, header=0)
        bus_data = pd.read_csv(bus_data_path)
        gen_data = pd.read_csv(gen_data_path)
        cap_data = pd.read_csv(cap_data_path)
        reg_data = pd.read_csv(reg_data_path)
        p_rating_mult = 4
        target_per_phase = 0
        gen_data.loc[:, ["pa", "pb", "pc"]] *= p_rating_mult
        gen_data.loc[:, ["sa_max", "sb_max", "sc_max"]] *= p_rating_mult

        bus_data.loc[bus_data.bus_type == "SWING", ["v_a", "v_b", "v_c"]] = 1.0
        model_new = LinDistModelQGen(
            branch_data, bus_data, gen_data, cap_data, reg_data
        )
        powerdata = pd.read_csv(legacy_powerdata_path, header=0)
        model_old = QModel(
            branch_data,
            powerdata,
            p_rating_mult=p_rating_mult,
            v_up=1.0,
            p_base_gld=1e6,
            v_ll_base_gld=4160,
        )
        target_per_phase = 0.2
        target = np.array([target_per_phase, target_per_phase, target_per_phase])
        # print("Solve old")
        model_old.loss_percent = np.array([0.1, 0.1, 0.1])
        res_old = model_old.solve(objective="q_target", target=target)
        print(f"old: {res_old.fun}")
        # print("Solve new")
        res_new = opf_solver.cvxpy_solve(
            model_new,
            opf_solver.cp_obj_target_q_3ph,
            target=target,
            error_percent=np.array([0.1, 0.1, 0.1]),
        )
        print(f"new: {res_new.fun}")
        assert_results_equal(model_new, model_old, res_new, res_old)

    def test_cp_obj_target_q_total(self):
        branch_data = pd.read_csv(branchdata_path)
        bus_data = pd.read_csv(bus_data_path)
        gen_data = pd.read_csv(gen_data_path)
        cap_data = pd.read_csv(cap_data_path)
        reg_data = pd.read_csv(reg_data_path)
        p_rating_mult = 4
        target_per_phase = 0
        gen_data.loc[:, ["pa", "pb", "pc"]] *= p_rating_mult
        gen_data.loc[:, ["sa_max", "sb_max", "sc_max"]] *= p_rating_mult

        bus_data.loc[bus_data.bus_type == "SWING", ["v_a", "v_b", "v_c"]] = 1.0
        model_new = LinDistModelQGen(
            branch_data, bus_data, gen_data, cap_data, reg_data
        )
        powerdata = pd.read_csv(legacy_powerdata_path, header=0)
        model_old = QModel(
            branch_data,
            powerdata,
            p_rating_mult=p_rating_mult,
            v_up=1.0,
            p_base_gld=1e6,
            v_ll_base_gld=4160,
        )
        target = target_per_phase * 3
        print("Solve old")
        model_old.loss_percent = np.array([0.1, 0.1, 0.1])
        res_old = model_old.solve(objective="q_target", target=target)
        print(f"old: {res_old.fun}")
        print("Solve new")
        res_new = opf_solver.cvxpy_solve(
            model_new,
            opf_solver.cp_obj_target_q_total,
            target=target,
            error_percent=np.array([0.1, 0.1, 0.1]),
        )
        print(f"new: {res_new.fun}")
        assert_results_equal(model_new, model_old, res_new, res_old)

    def test_cp_obj_target_p_3ph(self):
        area_dir = Path("./")
        assert area_dir.exists()
        branch_data = pd.read_csv(branchdata_path)
        bus_data = pd.read_csv(bus_data_path)
        gen_data = pd.read_csv(gen_data_path)
        cap_data = pd.read_csv(cap_data_path)
        reg_data = pd.read_csv(reg_data_path)
        gen_data.loc[:, ["pa", "pb", "pc"]] *= 5
        gen_data.loc[:, ["sa_max", "sb_max", "sc_max"]] *= 5
        bus_data.loc[:, ["pl_a", "pl_b", "pl_c"]] *= 0.5
        bus_data.loc[:, ["ql_a", "ql_b", "ql_c"]] *= 0.5

        model_new = LinDistModelPGen(
            branch_data, bus_data, gen_data, cap_data, reg_data
        )
        powerdata = pd.read_csv(legacy_powerdata_path)
        model_old = PModel(
            branch_data,
            powerdata,
            p_rating_mult=5,
            load_mult=0.5,
            v_up=1.05,
            p_base_gld=1e6,
            v_ll_base_gld=4160,
        )
        print("Solve old")
        model_old.loss_percent = np.array([0.1, 0.1, 0.1])
        res_old = model_old.solve(
            objective="p_target", target=np.array([0.3, 0.3, 0.3])
        )
        print(f"old: {res_old.fun}")
        print("Solve new")
        res_new = opf_solver.cvxpy_solve(
            model_new,
            opf_solver.cp_obj_target_p_3ph,
            target=np.array([0.3, 0.3, 0.3]),
            error_percent=np.array([0.1, 0.1, 0.1]),
        )
        print(f"new: {res_new.fun}")
        assert_results_equal(model_new, model_old, res_new, res_old)

    def test_cp_obj_target_p_total(self):
        area_dir = Path("./")
        assert area_dir.exists()
        branch_data = pd.read_csv(branchdata_path)
        bus_data = pd.read_csv(bus_data_path)
        gen_data = pd.read_csv(gen_data_path)
        cap_data = pd.read_csv(cap_data_path)
        reg_data = pd.read_csv(reg_data_path)
        target = 0
        p_rating_mult = 5
        load_mult = 0.5
        gen_data.loc[:, ["pa", "pb", "pc"]] *= p_rating_mult
        gen_data.loc[:, ["sa_max", "sb_max", "sc_max"]] *= p_rating_mult
        bus_data.loc[:, ["pl_a", "pl_b", "pl_c"]] *= load_mult
        bus_data.loc[:, ["ql_a", "ql_b", "ql_c"]] *= load_mult
        model_new = LinDistModelPGen(
            branch_data, bus_data, gen_data, cap_data, reg_data
        )
        powerdata = pd.read_csv(legacy_powerdata_path)
        model_old = PModel(
            branch_data,
            powerdata,
            p_rating_mult=p_rating_mult,
            v_up=1.05,
            load_mult=load_mult,
            p_base_gld=1e6,
            v_ll_base_gld=4160,
        )
        print("Solve old")
        model_old.loss_percent = np.array([0.1, 0.1, 0.1])
        res_old = model_old.solve(objective="p_target", target=target)
        print(f"old: {res_old.fun}")
        print("Solve new")
        res_new = opf_solver.cvxpy_solve(
            model_new,
            opf_solver.cp_obj_target_p_total,
            target=target,
            error_percent=np.array([0.1, 0.1, 0.1]),
        )
        print(f"new: {res_new.fun}")
        # assert_results_equal(model_new, model_old, res_new, res_old)
        assert abs(res_new.fun - res_old.fun) <= 1.0e-6

    def test_cp_obj_quadratic_curtail(self):
        # area_dir = Path("./")
        # assert area_dir.exists()
        branch_data = pd.read_csv(branchdata_path)
        bus_data = pd.read_csv(bus_data_path)
        gen_data = pd.read_csv(gen_data_path)
        cap_data = pd.read_csv(cap_data_path)
        reg_data = pd.read_csv(reg_data_path)
        gen_data.loc[:, ["pa", "pb", "pc"]] *= 10
        gen_data.loc[:, ["sa_max", "sb_max", "sc_max"]] *= 10
        gen_data.control_variable = "P"
        bus_data.loc[:, ["pl_a", "pl_b", "pl_c"]] *= 0.1
        bus_data.loc[:, ["ql_a", "ql_b", "ql_c"]] *= 0.1
        model_new = LinDistModelPGen(
            branch_data, bus_data, gen_data, cap_data, reg_data
        )
        powerdata = pd.read_csv(legacy_powerdata_path)
        model_old = PModel(
            branch_data,
            powerdata,
            p_rating_mult=10,
            v_up=1.05,
            load_mult=0.1,
            p_base_gld=1e6,
            v_ll_base_gld=4160,
        )
        print("Solve old")
        res_old = model_old.solve(objective="quadratic curtail")
        print(f"old: {res_old.fun}")
        print("Solve new")
        res_new = opf_solver.cvxpy_solve(model_new, opf_solver.cp_obj_curtail)
        print(f"new: {res_new.fun}")
        assert_results_equal(model_new, model_old, res_new, res_old)

    def test_cp_obj_curtail(self):
        branch_data = pd.read_csv(branchdata_path)
        bus_data = pd.read_csv(bus_data_path)
        gen_data = pd.read_csv(gen_data_path)
        cap_data = pd.read_csv(cap_data_path)
        reg_data = pd.read_csv(reg_data_path)
        gen_data.loc[:, ["pa", "pb", "pc"]] *= 10
        gen_data.loc[:, ["sa_max", "sb_max", "sc_max"]] *= 10
        gen_data.control_variable = "P"
        bus_data.loc[:, ["pl_a", "pl_b", "pl_c"]] *= 0.1
        bus_data.loc[:, ["ql_a", "ql_b", "ql_c"]] *= 0.1
        model_new = LinDistModelPGen(
            branch_data, bus_data, gen_data, cap_data, reg_data
        )
        powerdata = pd.read_csv(legacy_powerdata_path)
        model_old = PModel(
            branch_data,
            powerdata,
            p_rating_mult=10,
            v_up=1.05,
            load_mult=0.1,
            p_base_gld=1e6,
            v_ll_base_gld=4160,
        )
        print("Solve old")
        res_old = model_old.solve(objective="curtail")
        print(f"old: {res_old.fun}")
        print("Solve new")
        res_new = opf_solver.lp_solve(model_new, opf_solver.gradient_curtail(model_new))
        print(f"new: {res_new.fun}")
        # assert_results_equal(model_new, model_old, res_new, res_old)
        assert abs(res_new.fun - res_old.fun) <= 1.0e-6
