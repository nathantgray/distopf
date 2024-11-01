import unittest
from email.policy import default
from pathlib import Path

import numpy as np
import pandas as pd

from distopf import opf_solver
from distopf.multiperiod.lindist_base_modular_multi import LinDistModelModular as LinDistMulti
from distopf import LinDistModelModular
import distopf as opf
from distopf import CASES_DIR

branchdata_path = Path("./distopf/test/branch_data.csv")
powerdata_path = Path("./distopf/test/powerdata.csv")
legacy_powerdata_path = Path("./distopf/test/legacy/powerdata.csv")
bus_data_path = Path("./distopf/test/bus_data.csv")
gen_data_path = Path("./distopf/test/gen_data.csv")
cap_data_path = Path("./distopf/test/cap_data.csv")
reg_data_path = Path("./distopf/test/reg_data.csv")


class TestMulti(unittest.TestCase):
    def test_loss(self):
        # base_path = CASES_DIR / "csv/2Bus-1ph-batt"
        base_path = CASES_DIR / "csv/ieee123_alternate"

        branch_data = pd.read_csv(base_path / "branch_data.csv")
        bus_data = pd.read_csv(base_path / "bus_data.csv")
        gen_data = pd.read_csv(base_path / "gen_data.csv")
        reg_data = pd.read_csv(base_path / "reg_data.csv")
        # cap_data = pd.read_csv(base_path / "cap_data.csv")
        # battery_data = pd.read_csv(base_path / "battery_data.csv")
        pv_loadshape = pd.read_csv(base_path / "pv_loadshape.csv")
        default_loadshape = pd.read_csv(base_path / "default_loadshape.csv")
        start_time = 12
        load_mult = default_loadshape.loc[start_time, "M"]
        gen_mult = pv_loadshape.loc[start_time, "PV"]
        # bus_data.v_min = 0
        # bus_data.v_max = 1.07
        bus_data.v_a = 1.0
        bus_data.v_b = 1.0
        bus_data.v_c = 1.0
        m1 = LinDistMulti(
            branch_data=branch_data,
            bus_data=bus_data,
            gen_data=gen_data,
            reg_data=reg_data,
            # cap_data=cap_data,
            loadshape_data=default_loadshape,
            pv_loadshape_data=pv_loadshape,
            # bat_data=battery_data,
            start_step=start_time,
            n_steps=1,
        )
        gen_data.pa *= gen_mult
        gen_data.pb *= gen_mult
        gen_data.pc *= gen_mult
        bus_data.pl_a *= load_mult
        bus_data.pl_b *= load_mult
        bus_data.pl_c *= load_mult

        m2 = LinDistModelModular(
            branch_data=branch_data,
            bus_data=bus_data,
            gen_data=gen_data,
            reg_data=reg_data,
        )
        result1 = opf.multiperiod.opf_solver_multi.cvxpy_solve(m1, opf.multiperiod.cp_obj_loss)
        print(result1.runtime)
        print(result1.fun)
        v1 = m1.get_voltages(result1.x)
        s1 = m1.get_apparent_power_flows(result1.x)

        q_der1 = m1.get_q_gens(result1.x)
        p_der1 = m1.get_p_gens(result1.x)
        p_bat_discharge1 = m1.get_p_discharge(result1.x)
        p_bat_charge1 = m1.get_p_charge(result1.x)
        soc1 = m1.get_soc(result1.x)
        p_load1 = m1.get_p_loads(result1.x)
        q_load1 = m1.get_q_loads(result1.x)
        p_flow1 = s1.copy()
        q_flow1 = s1.copy()
        p_flow1.a = p_flow1.a.apply(np.real)
        q_flow1.a = q_flow1.a.apply(np.imag)
        p_flow1.b = p_flow1.b.apply(np.real)
        q_flow1.b = q_flow1.b.apply(np.imag)
        p_flow1.c = p_flow1.c.apply(np.real)
        q_flow1.c = q_flow1.c.apply(np.imag)


        result2 = opf.cvxpy_solve(m2, opf.cp_obj_loss)
        print(result2.runtime)
        print(result2.fun)
        v2 = m1.get_voltages(result2.x)
        s2 = m1.get_apparent_power_flows(result2.x)

        q_der2 = m2.get_q_gens(result2.x)
        p_der2 = m2.get_p_gens(result2.x)
        p_load2 = m2.get_p_loads(result2.x)
        q_load2 = m2.get_q_loads(result2.x)
        p_flow2 = s2.copy()
        q_flow2 = s2.copy()
        p_flow2.a = p_flow2.a.apply(np.real)
        q_flow2.a = q_flow2.a.apply(np.imag)
        p_flow2.b = p_flow2.b.apply(np.real)
        q_flow2.b = q_flow2.b.apply(np.imag)
        p_flow2.c = p_flow2.c.apply(np.real)
        q_flow2.c = q_flow2.c.apply(np.imag)
        assert True
