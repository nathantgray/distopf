import distopf as opf
import unittest
import pandas as pd
from distopf import LinDistModel
from distopf import CASES_DIR


class TestPlots(unittest.TestCase):
    def test_network(self):
        base_path = CASES_DIR / "csv/ieee123_30der"
        branch_data = pd.read_csv(base_path / "branch_data.csv")
        bus_data = pd.read_csv(base_path / "bus_data.csv")
        gen_data = pd.read_csv(base_path / "gen_data.csv")
        reg_data = pd.read_csv(base_path / "reg_data.csv")
        cap_data = pd.read_csv(base_path / "cap_data.csv")
        bus_data.v_a = 1.0
        bus_data.v_b = 1.0
        bus_data.v_c = 1.0
        m = LinDistModel(
            branch_data=branch_data,
            bus_data=bus_data,
            gen_data=gen_data,
            reg_data=reg_data,
            cap_data=cap_data,
        )
        result = opf.cvxpy_solve(m, opf.cp_obj_loss)
        v = m.get_voltages(result.x)
        s = m.get_apparent_power_flows(result.x)
        pg = m.get_p_gens(result.x)
        qg = m.get_q_gens(result.x)
        fig = opf.plot_network(m, v, s, pg, qg)
        fig.show()

    def test_plot_ders(self):
        base_path = CASES_DIR / "csv/ieee123_30der"
        branch_data = pd.read_csv(base_path / "branch_data.csv")
        bus_data = pd.read_csv(base_path / "bus_data.csv")
        gen_data = pd.read_csv(base_path / "gen_data.csv")
        reg_data = pd.read_csv(base_path / "reg_data.csv")
        cap_data = pd.read_csv(base_path / "cap_data.csv")
        bus_data.v_a = 1.0
        bus_data.v_b = 1.0
        bus_data.v_c = 1.0
        m = LinDistModel(
            branch_data=branch_data,
            bus_data=bus_data,
            gen_data=gen_data,
            reg_data=reg_data,
            cap_data=cap_data,
        )
        result = opf.cvxpy_solve(m, opf.cp_obj_loss)
        qg = m.get_q_gens(result.x)
        fig = opf.plot_ders(qg)
        fig.show()

    def test_plot_voltages(self):
        base_path = CASES_DIR / "csv/ieee123_30der"
        branch_data = pd.read_csv(base_path / "branch_data.csv")
        bus_data = pd.read_csv(base_path / "bus_data.csv")
        gen_data = pd.read_csv(base_path / "gen_data.csv")
        reg_data = pd.read_csv(base_path / "reg_data.csv")
        cap_data = pd.read_csv(base_path / "cap_data.csv")
        bus_data.v_a = 1.0
        bus_data.v_b = 1.0
        bus_data.v_c = 1.0
        m = LinDistModel(
            branch_data=branch_data,
            bus_data=bus_data,
            gen_data=gen_data,
            reg_data=reg_data,
            cap_data=cap_data,
        )
        result = opf.cvxpy_solve(m, opf.cp_obj_loss)
        v = m.get_voltages(result.x)
        fig = opf.plot_voltages(v)
        fig.show()

    def test_compare_voltages(self):
        base_path = CASES_DIR / "csv/ieee123_30der"
        branch_data = pd.read_csv(base_path / "branch_data.csv")
        bus_data = pd.read_csv(base_path / "bus_data.csv")
        gen_data = pd.read_csv(base_path / "gen_data.csv")
        reg_data = pd.read_csv(base_path / "reg_data.csv")
        cap_data = pd.read_csv(base_path / "cap_data.csv")
        bus_data.v_a = 1.0
        bus_data.v_b = 1.0
        bus_data.v_c = 1.0
        m = LinDistModel(
            branch_data=branch_data,
            bus_data=bus_data,
            gen_data=gen_data,
            reg_data=reg_data,
            cap_data=cap_data,
        )
        gen_data.a_mode = "CONTROL_PQ"
        gen_data.b_mode = "CONTROL_PQ"
        gen_data.c_mode = "CONTROL_PQ"
        m2 = LinDistModel(
            branch_data=branch_data,
            bus_data=bus_data,
            gen_data=gen_data,
            reg_data=reg_data,
            cap_data=cap_data,
        )
        result = opf.cvxpy_solve(m, opf.cp_obj_loss)
        result2 = opf.cvxpy_solve(m2, opf.cp_obj_loss)
        v = m.get_voltages(result.x)
        v2 = m.get_voltages(result2.x)
        fig = opf.compare_voltages(v, v2)
        fig.show()

    def test_voltage_differences(self):
        base_path = CASES_DIR / "csv/ieee123_30der"
        branch_data = pd.read_csv(base_path / "branch_data.csv")
        bus_data = pd.read_csv(base_path / "bus_data.csv")
        gen_data = pd.read_csv(base_path / "gen_data.csv")
        reg_data = pd.read_csv(base_path / "reg_data.csv")
        cap_data = pd.read_csv(base_path / "cap_data.csv")
        bus_data.v_a = 1.0
        bus_data.v_b = 1.0
        bus_data.v_c = 1.0
        m = LinDistModel(
            branch_data=branch_data,
            bus_data=bus_data,
            gen_data=gen_data,
            reg_data=reg_data,
            cap_data=cap_data,
        )
        gen_data.a_mode = "CONTROL_PQ"
        gen_data.b_mode = "CONTROL_PQ"
        gen_data.c_mode = "CONTROL_PQ"
        m2 = LinDistModel(
            branch_data=branch_data,
            bus_data=bus_data,
            gen_data=gen_data,
            reg_data=reg_data,
            cap_data=cap_data,
        )
        result = opf.cvxpy_solve(m, opf.cp_obj_loss)
        result2 = opf.cvxpy_solve(m2, opf.cp_obj_loss)
        v = m.get_voltages(result.x)
        v2 = m.get_voltages(result2.x)
        fig = opf.voltage_differences(v, v2)
        fig.show()

    def test_plot_power_flows(self):
        base_path = CASES_DIR / "csv/ieee123_30der"
        branch_data = pd.read_csv(base_path / "branch_data.csv")
        bus_data = pd.read_csv(base_path / "bus_data.csv")
        gen_data = pd.read_csv(base_path / "gen_data.csv")
        reg_data = pd.read_csv(base_path / "reg_data.csv")
        cap_data = pd.read_csv(base_path / "cap_data.csv")
        bus_data.v_a = 1.0
        bus_data.v_b = 1.0
        bus_data.v_c = 1.0
        m = LinDistModel(
            branch_data=branch_data,
            bus_data=bus_data,
            gen_data=gen_data,
            reg_data=reg_data,
            cap_data=cap_data,
        )
        result = opf.cvxpy_solve(m, opf.cp_obj_loss)
        s = m.get_apparent_power_flows(result.x)
        fig = opf.plot_power_flows(s)
        fig.show()

    def test_compare_flows(self):
        base_path = CASES_DIR / "csv/ieee123_30der"
        branch_data = pd.read_csv(base_path / "branch_data.csv")
        bus_data = pd.read_csv(base_path / "bus_data.csv")
        gen_data = pd.read_csv(base_path / "gen_data.csv")
        reg_data = pd.read_csv(base_path / "reg_data.csv")
        cap_data = pd.read_csv(base_path / "cap_data.csv")
        bus_data.v_a = 1.0
        bus_data.v_b = 1.0
        bus_data.v_c = 1.0
        m = LinDistModel(
            branch_data=branch_data,
            bus_data=bus_data,
            gen_data=gen_data,
            reg_data=reg_data,
            cap_data=cap_data,
        )
        gen_data.a_mode = "CONTROL_PQ"
        gen_data.b_mode = "CONTROL_PQ"
        gen_data.c_mode = "CONTROL_PQ"
        m2 = LinDistModel(
            branch_data=branch_data,
            bus_data=bus_data,
            gen_data=gen_data,
            reg_data=reg_data,
            cap_data=cap_data,
        )
        result = opf.cvxpy_solve(m, opf.cp_obj_loss)
        result2 = opf.cvxpy_solve(m2, opf.cp_obj_loss)
        s = m.get_apparent_power_flows(result.x)
        s2 = m.get_apparent_power_flows(result2.x)
        fig = opf.compare_flows(s, s2)
        fig.show()

    def test_polar(self):
        base_path = CASES_DIR / "csv/ieee123_30der"
        branch_data = pd.read_csv(base_path / "branch_data.csv")
        bus_data = pd.read_csv(base_path / "bus_data.csv")
        gen_data = pd.read_csv(base_path / "gen_data.csv")
        reg_data = pd.read_csv(base_path / "reg_data.csv")
        cap_data = pd.read_csv(base_path / "cap_data.csv")
        bus_data.v_a = 1.0
        bus_data.v_b = 1.0
        bus_data.v_c = 1.0
        m = LinDistModel(
            branch_data=branch_data,
            bus_data=bus_data,
            gen_data=gen_data,
            reg_data=reg_data,
            cap_data=cap_data,
        )
        result = opf.cvxpy_solve(m, opf.cp_obj_loss)
        pg = m.get_p_gens(result.x)
        qg = m.get_q_gens(result.x)
        fig = opf.plot_polar(pg, qg)
        fig.show()
