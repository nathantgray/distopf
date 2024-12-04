import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from distopf import plot_network, compare_flows, compare_voltages
from distopf import opf_solver
from distopf.multiperiod.lindist_base_modular_multi import LinDistModelModular as LinDistMulti
from distopf.multiperiod.lindist_multi_fast import LinDistModelMultiFast
from distopf import LinDistModel
from distopf import LinDistModelQ
from distopf.lindist_fast import LinDistModelFast
from distopf.lindist_q_fast import LinDistModelQFast
import distopf as opf
from distopf import CASES_DIR

branchdata_path = Path("./distopf/test/branch_data.csv")
powerdata_path = Path("./distopf/test/powerdata.csv")
legacy_powerdata_path = Path("./distopf/test/legacy/powerdata.csv")
bus_data_path = Path("./distopf/test/bus_data.csv")
gen_data_path = Path("./distopf/test/gen_data.csv")
cap_data_path = Path("./distopf/test/cap_data.csv")
reg_data_path = Path("./distopf/test/reg_data.csv")


class TestModular(unittest.TestCase):
    def test_loss(self):
        # base_path = CASES_DIR / "csv/2Bus-1ph-batt"
        for start_time in range(0, 24, 6):
            base_path = CASES_DIR / "csv/ieee123_alternate"

            branch_data = pd.read_csv(base_path / "branch_data.csv")
            bus_data = pd.read_csv(base_path / "bus_data.csv")
            gen_data = pd.read_csv(base_path / "gen_data.csv")
            reg_data = pd.read_csv(base_path / "reg_data.csv")
            cap_data = pd.read_csv(base_path / "cap_data.csv")
            # battery_data = pd.read_csv(base_path / "battery_data.csv")
            pv_loadshape = pd.read_csv(base_path / "pv_loadshape.csv")
            default_loadshape = pd.read_csv(base_path / "default_loadshape.csv")
            load_mult = default_loadshape.loc[start_time, "M"]
            gen_mult = pv_loadshape.loc[start_time, "PV"]
            bus_data.v_a = 1.0
            bus_data.v_b = 1.0
            bus_data.v_c = 1.0
            m1 = LinDistMulti(
                branch_data=branch_data,
                bus_data=bus_data,
                gen_data=gen_data,
                reg_data=reg_data,
                cap_data=cap_data,
                loadshape_data=default_loadshape,
                pv_loadshape_data=pv_loadshape,
                # bat_data=battery_data,
                start_step=start_time,
                n_steps=1,
            )
            mf = LinDistModelMultiFast(
                branch_data=branch_data,
                bus_data=bus_data,
                gen_data=gen_data,
                reg_data=reg_data,
                cap_data=cap_data,
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
            bus_data.ql_a *= load_mult
            bus_data.ql_b *= load_mult
            bus_data.ql_c *= load_mult

            m2 = LinDistModel(
                branch_data=branch_data,
                bus_data=bus_data,
                gen_data=gen_data,
                reg_data=reg_data,
                cap_data=cap_data,
            )

            m3 = LinDistModelQ(
                branch_data=branch_data,
                bus_data=bus_data,
                gen_data=gen_data,
                reg_data=reg_data,
                cap_data=cap_data,
            )

            m4 = LinDistModelFast(
                branch_data=branch_data,
                bus_data=bus_data,
                gen_data=gen_data,
                reg_data=reg_data,
                cap_data=cap_data,
            )
            m5 = LinDistModelQFast(
                branch_data=branch_data,
                bus_data=bus_data,
                gen_data=gen_data,
                reg_data=reg_data,
                cap_data=cap_data,
            )


            result1 = opf.multiperiod.opf_solver_multi.cvxpy_solve(m1, opf.multiperiod.cp_obj_loss, solver="CLARABEL")
            resultf = opf.multiperiod.opf_solver_multi.cvxpy_solve(mf, opf.multiperiod.cp_obj_loss, solver="CLARABEL")
            result2 = opf.cvxpy_solve(m2, opf.cp_obj_loss, solver="CLARABEL")
            result3 = opf.cvxpy_solve(m3, opf_solver.cp_obj_loss, solver="CLARABEL")
            result4 = opf.cvxpy_solve(m4, opf_solver.cp_obj_loss, solver="CLARABEL")
            result5 = opf.cvxpy_solve(m5, opf_solver.cp_obj_loss, solver="CLARABEL")

            print(start_time)
            print(f"multi:  objective={result1.fun}\t in {result1.runtime}s")
            print(f"mpFast: objective={resultf.fun}\t in {resultf.runtime}")
            print(f"single: objective={result2.fun}\t in {result2.runtime}s")
            print(f"Q only: objective={result3.fun}\t in {result3.runtime}s")
            print(f"fast:   objective={result4.fun}\t in {result4.runtime}s")
            print(f"Q fast: objective={result5.fun}\t in {result5.runtime}s")
            # print("debug")
            # assert abs(result1.fun - result2.fun) < 1e-3
