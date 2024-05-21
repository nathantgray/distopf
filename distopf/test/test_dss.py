import unittest
from pathlib import Path

import numpy as np

import distopf as opf
from distopf import (
    LinDistModel,
    lp_solve,
)
from distopf.dssconverter.dssparser import DSSParser


class TestDSS(unittest.TestCase):
    def test_dss(self):

        test2 = opf.CASES_DIR / "dss/2Bus/2Bus.DSS"
        test3 = opf.CASES_DIR / "dss/3Bus/3Bus.DSS"
        ieee4 = opf.CASES_DIR / "dss/4Bus-YY-Bal/4Bus-YY-Bal.DSS"
        ieee13 = opf.CASES_DIR / "dss/ieee13_dss/IEEE13Nodeckt.dss"
        ieee34 = opf.CASES_DIR / "dss/34Bus/Run_IEEE34Mod2.dss"
        ieee123 = opf.CASES_DIR / "dss/ieee123_dss/Run_IEEE123Bus.DSS"
        rahul123 = opf.CASES_DIR / "dss/rahul123/ieee123master_base.dss"
        dirs = [
            Path(test2),
            Path(test3),
            Path(ieee4),
            Path(ieee13),
            Path(ieee34),
            Path(ieee123),
            Path(rahul123),
        ]
        for _dir in dirs:
            print(_dir)
            for mult in np.linspace(0, 1, 6):
                dss_parser = DSSParser(_dir, s_base=1e6, v_min=0, v_max=2)
                dss_parser.dss.Solution.LoadMult(mult)
                dss_parser.dss.Solution.Solve()
                dss_parser.update()
                model = opf.LinDistModel(
                    dss_parser.branch_data,
                    dss_parser.bus_data,
                    cap_data=dss_parser.cap_data,
                    reg_data=dss_parser.reg_data,
                )
                try:
                    result = opf.lp_solve(model, np.zeros(model.n_x))
                except ValueError:
                    continue
                v_df = model.get_voltages(result.x)
                s_df = model.get_apparent_power_flows(result.x)
                v_diff = v_df.copy()
                v_diff.loc[:, ["a", "b", "c"]] = (
                    v_df.loc[:, ["a", "b", "c"]].astype(float)
                    - dss_parser.v_solved.loc[:, ["a", "b", "c"]].abs()
                )
                v_rdiff = (
                    v_diff.loc[:, ["a", "b", "c"]]
                    / dss_parser.v_solved.loc[:, ["a", "b", "c"]].abs()
                )

                s_df = (
                    s_df.groupby(by=["fb", "tb"], as_index=False)
                    .agg(
                        {
                            "fb": "first",
                            "tb": "first",
                            "a": "sum",
                            "b": "sum",
                            "c": "sum",
                        }
                    )
                    .reset_index(drop=True)
                    .sort_values(by=["fb"], ignore_index=True)
                    .sort_values(by=["tb"], ignore_index=True)
                )
                p_opf = s_df.loc[:, ["a", "b", "c"]].to_numpy().real
                q_opf = s_df.loc[:, ["a", "b", "c"]].to_numpy().imag
                p_dss = dss_parser.s_solved.loc[:, ["a", "b", "c"]].to_numpy().real
                q_dss = dss_parser.s_solved.loc[:, ["a", "b", "c"]].to_numpy().imag
                p_err = max(abs(p_opf - p_dss).flatten())
                q_err = max(abs(q_opf - q_dss).flatten())
                print(
                    f"{mult:.1f}: V error %: {v_rdiff.max().max():.3e} -- P error (pu): {p_err:.3e} -- Q error (pu): {q_err:.3e}"
                )
                assert v_rdiff.max().max() < 0.12
                assert p_err < 2
                assert q_err < 2
