from functools import cache

import networkx as nx
import numpy as np
import pandas as pd
from numpy import sqrt, zeros
from scipy.sparse import csr_matrix


# bus_type options
SWING_FREE = "IN"
PQ_FREE = "OUT"
SWING_BUS = "SWING"
PQ_BUS = "PQ"


def get(s: pd.Series, i, default=None):
    """
    Get value at index i from a Series. Return default if it does not exist.
    Parameters
    ----------
    s : pd.Series
    i : index or key for eries
    default : value to return if it fails

    Returns
    -------
    value: value at index i or default if it doesn't exist.
    """
    try:
        return s.loc[i]
    except (KeyError, ValueError, IndexError):
        return default


def _handle_gen_input(gen_data: pd.DataFrame) -> pd.DataFrame:
    if gen_data is None:
        return pd.DataFrame(
            columns=[
                "id",
                "name",
                "pa",
                "pb",
                "pc",
                "qa",
                "qb",
                "qc",
                "sa_max",
                "sb_max",
                "sc_max",
                "phases",
                "qa_max",
                "qb_max",
                "qc_max",
                "qa_min",
                "qb_min",
                "qc_min",
            ]
        )
    gen = gen_data.sort_values(by="id", ignore_index=True)
    gen.index = gen.id.to_numpy() - 1
    return gen


def _handle_cap_input(cap_data: pd.DataFrame) -> pd.DataFrame:
    if cap_data is None:
        return pd.DataFrame(
            columns=[
                "id",
                "name",
                "qa",
                "qb",
                "qc",
                "phases",
            ]
        )
    cap = cap_data.sort_values(by="id", ignore_index=True)
    cap.index = cap.id.to_numpy() - 1
    return cap


def _handle_loadshape_input(loadshape_data: pd.DataFrame) -> pd.DataFrame:
    if loadshape_data is None:
        return pd.DataFrame(
            columns=[
                "time",
                "M",
            ]
        )
    loadshape = loadshape_data.sort_values(by="time", ignore_index=True)
    loadshape.index = loadshape.time.to_numpy()
    return loadshape


def _handle_pv_loadshape_input(pv_loadshape_data: pd.DataFrame) -> pd.DataFrame:
    if pv_loadshape_data is None:
        return pd.DataFrame(
            columns=[
                "time",
                "PV",
            ]
        )
    pv_loadshape = pv_loadshape_data.sort_values(by="time", ignore_index=True)
    pv_loadshape.index = pv_loadshape.time.to_numpy()
    return pv_loadshape


def _handle_bat_input(bat_data: pd.DataFrame) -> pd.DataFrame:
    if bat_data is None:
        return pd.DataFrame(
            columns=[
                "id",
                "name",
                "nc_a",
                "nc_b",
                "nc_c",
                "nd_a",
                "nd_b",
                "nd_c",
                "hmax_a",
                "hmax_b",
                "hmax_c",
                "Pb_max_a",
                "Pb_max_b",
                "Pb_max_c",
                "bmin_a",
                "bmin_b",
                "bmin_c",
                "bmax_a",
                "bmax_b",
                "bmax_c",
                "b0_a",
                "b0_b",
                "b0_c",
                "phases",
            ]
        )
    bat = bat_data.sort_values(by="id", ignore_index=True)
    bat.index = bat.id.to_numpy() - 1


def _handle_reg_input(reg_data: pd.DataFrame) -> pd.DataFrame:
    if reg_data is None:
        return pd.DataFrame(
            columns=[
                "fb",
                "tb",
                "phases",
                "tap_a",
                "tap_b",
                "tap_c",
                "ratio_a",
                "ratio_b",
                "ratio_c",
            ]
        )
    reg = reg_data.sort_values(by="tb", ignore_index=True)
    reg.index = reg.tb.to_numpy() - 1
    for ph in "abc":
        if f"tap_{ph}" in reg.columns and not f"ratio_{ph}" in reg.columns:
            reg[f"ratio_{ph}"] = 1 + 0.00625 * reg[f"tap_{ph}"]
        elif f"ratio_{ph}" in reg.columns and not f"tap_{ph}" in reg.columns:
            reg[f"tap_{ph}"] = (reg[f"ratio_{ph}"] - 1) / 0.00625
        elif f"ratio_{ph}" in reg.columns and f"tap_{ph}" in reg.columns:
            # check consistency
            if any(abs(reg[f"ratio_{ph}"]) - (1 + 0.00625 * reg[f"tap_{ph}"]) > 1e-6):
                raise ValueError(
                    f"Regulator taps and ratio are inconsistent on phase {ph}!"
                )
    return reg


def _handle_branch_input(branch_data: pd.DataFrame) -> pd.DataFrame:
    if branch_data is None:
        raise ValueError("Branch data must be provided.")
    branch = branch_data.sort_values(by="tb", ignore_index=True)
    branch = branch.loc[branch.status != "OPEN", :]
    return branch


def _handle_bus_input(bus_data: pd.DataFrame) -> pd.DataFrame:
    if bus_data is None:
        raise ValueError("Bus data must be provided.")
    bus = bus_data.sort_values(by="id", ignore_index=True)
    bus.index = bus.id.to_numpy() - 1
    return bus


class LinDistModel:
    """
    LinDistFlow Model base class.

    Parameters
    ----------
    branch_data : pd.DataFrame
        DataFrame containing branch data (r and x values, limits)
    bus_data : pd.DataFrame
        DataFrame containing bus data (loads, voltages, limits)
    gen_data : pd.DataFrame
        DataFrame containing generator/DER data
    cap_data : pd.DataFrame
        DataFrame containing capacitor data
    reg_data : pd.DataFrame
        DataFrame containing regulator data
    bat_data : pd DataFrame
        DataFrame containing battery data
    loadshape_data : pd.DataFrame
        DataFrame containing loadshape multipliers for P values
    pv_loadshape_data : pd.DataFrame
        DataFrame containing PV profile of 1h interval for 24h

    """

    def __init__(
        self,
        branch_data: pd.DataFrame = None,
        bus_data: pd.DataFrame = None,
        gen_data: pd.DataFrame = None,
        cap_data: pd.DataFrame = None,
        reg_data: pd.DataFrame = None,
        loadshape_data: pd.DataFrame = None,
        pv_loadshape_data: pd.DataFrame = None,
        bat_data: pd.DataFrame = None,
    ):
        # ~~~~~~~~~~~~~~~~~~~~ Load Data Frames ~~~~~~~~~~~~~~~~~~~~
        self.branch = _handle_branch_input(branch_data)
        self.bus = _handle_bus_input(bus_data)
        self.gen = _handle_gen_input(gen_data)
        self.cap = _handle_cap_input(cap_data)
        self.reg = _handle_reg_input(reg_data)
        self.loadshape = _handle_loadshape_input(loadshape_data)
        self.pv_loadshape = _handle_pv_loadshape_input(pv_loadshape_data)
        self.bat = _handle_bat_input(bat_data)

        # ~~~~~~~~~~~~~~~~~~~~ prepare data ~~~~~~~~~~~~~~~~~~~~
        self.nb = len(self.bus.id)
        self.r, self.x = self._init_rx(self.branch)
        self.der_bus = dict(a=np.array([]), b=np.array([]), c=np.array([]))
        self.battery_bus = dict(a=np.array([]), b=np.array([]), c=np.array([]))
        if self.gen.shape[0] > 0:
            self.der_bus = {
                "a": self.gen.loc[self.gen.phases.str.contains("a")].index.to_numpy(),
                "b": self.gen.loc[self.gen.phases.str.contains("b")].index.to_numpy(),
                "c": self.gen.loc[self.gen.phases.str.contains("c")].index.to_numpy(),
            }
        if self.bat.shape[0] > 0:
            self.bat_bus = {
                "a": self.bat.loc[self.bat.phases.str.contains("a")].index.to_numpy(),
                "b": self.bat.loc[self.bat.phases.str.contains("b")].index.to_numpy(),
                "c": self.bat.loc[self.bat.phases.str.contains("c")].index.to_numpy(),
            }
        self.controlled_load_buses = {
            "a": self.bus.loc[self.bus.bus_type.str.contains(PQ_FREE)].index.to_numpy(),
            "b": self.bus.loc[self.bus.bus_type.str.contains(PQ_FREE)].index.to_numpy(),
            "c": self.bus.loc[self.bus.bus_type.str.contains(PQ_FREE)].index.to_numpy(),
        }
        # ~~ initialize index pointers ~~
        (
            self.nl_a,
            self.nl_b,
            self.nl_c,
            self.line_a,
            self.line_b,
            self.line_c,
            self.basic_length,
        ) = self.basic_var_length(self.branch)
        if LinDistModelQ.der and LinDistModelQ.battery:
            self.period = (
                int(self.basic_length)
                + len(self.der_bus["a"])
                + len(self.der_bus["b"])
                + len(self.der_bus["c"])
                + 3 * len(self.battery_bus["a"])
                + 3 * len(self.battery_bus["b"])
                + 3 * len(self.battery_bus["c"])
                + 6 * sum(self.bus.bus_type == PQ_FREE)
            )
        elif LinDistModelQ.der:
            self.period = (
                int(self.basic_length)
                + len(self.der_bus["a"])
                + len(self.der_bus["b"])
                + len(self.der_bus["c"])
                + 6 * sum(self.bus.bus_type == PQ_FREE)
            )
        elif LinDistModelQ.battery:
            self.period = (
                int(self.basic_length)
                + 3 * len(self.battery_bus["a"])
                + 3 * len(self.battery_bus["b"])
                + 3 * len(self.battery_bus["c"])
                + 6 * sum(self.bus.bus_type == PQ_FREE)
            )
        else:
            self.period = int(self.basic_length) + 6 * sum(self.bus.bus_type == PQ_FREE)
        self.x_maps, self.ctr_var_start_idx = self._variable_tables(self.branch)
        if LinDistModelQ.der and LinDistModelQ.battery:
            (
                self.p_der_start_phase_idx,
                self.q_der_start_phase_idx,
                self.pd_bat_start_phase_idx,
                self.pc_bat_start_phase_idx,
                self.b_bat_start_phase_idx,
                self.p_load_controlled,
                self.q_load_controlled,
            ) = self._control_variables(
                self.der_bus, self.ctr_var_start_idx, self.battery_bus
            )
            self.n_x = int(
                self.b_bat_start_phase_idx[LinDistModelQ.n_steps - 1]["c"]
                + len(self.battery_bus["c"])
            ) + 6 * sum(self.bus.bus_type == PQ_FREE)
            self.row_no = int(
                self.b_bat_start_phase_idx[LinDistModelQ.n_steps - 1]["c"]
                + len(self.battery_bus["c"])
            )
            self.a_eq, self.b_eq, self.a_ineq, self.b_ineq = self.create_model()
            self.bounds = self.init_bounds(self.bus, self.gen, self.bat)
        elif LinDistModelQ.der:
            (
                self.p_der_start_phase_idx,
                self.q_der_start_phase_idx,
                p_load_controlled,
                q_load_controlled,
            ) = self._control_variables(
                self.der_bus, self.ctr_var_start_idx, self.battery_bus
            )
            self.n_x = int(
                self.q_der_start_phase_idx[LinDistModelQ.n_steps - 1]["c"]
                + len(self.der_bus["c"])
            ) + 6 * sum(self.bus.bus_type == PQ_FREE)
            self.row_no = int(self.x_maps[LinDistModelQ.n_steps - 1]["c"].vj.max() + 1)
            self.a_eq, self.b_eq, self.a_ineq, self.b_ineq = self.create_model()
            self.bounds = self.init_bounds(self.bus, self.gen, self.bat)
        elif LinDistModelQ.battery:
            (
                self.pd_bat_start_phase_idx,
                self.pc_bat_start_phase_idx,
                self.b_bat_start_phase_idx,
                p_load_controlled,
                q_load_controlled,
            ) = self._control_variables(
                self.der_bus, self.ctr_var_start_idx, self.battery_bus
            )
            self.n_x = int(
                self.b_bat_start_phase_idx[LinDistModelQ.n_steps - 1]["c"]
                + len(self.battery_bus["c"])
            ) + 6 * sum(self.bus.bus_type == PQ_FREE)
            self.row_no = int(
                self.b_bat_start_phase_idx[LinDistModelQ.n_steps - 1]["c"]
                + len(self.battery_bus["c"])
            )
            self.a_eq, self.b_eq, self.a_ineq, self.b_ineq = self.create_model()
            self.bounds = self.init_bounds(self.bus, self.gen, self.bat)
        else:
            (p_load_controlled, q_load_controlled) = self._control_variables(
                self.der_bus, self.ctr_var_start_idx, self.battery_bus
            )
            self.n_x = (
                self.x_maps[LinDistModelQ.n_steps - 1]["c"].vj.max()
                + 1
                + 6 * sum(self.bus.bus_type == PQ_FREE)
            )
            self.row_no = int(self.x_maps[LinDistModelQ.n_steps - 1]["c"].vj.max() + 1)
            self.a_eq, self.b_eq, self.a_ineq, self.b_ineq = self.create_model()
            self.bounds = self.init_bounds(self.bus, self.gen, self.bat)

        @staticmethod
        def _init_rx(branch):
            row = np.array(np.r_[branch.fb, branch.tb], dtype=int) - 1
            col = np.array(np.r_[branch.tb, branch.fb], dtype=int) - 1
            r = {
                "aa": csr_matrix((np.r_[branch.raa, branch.raa], (row, col))),
                "ab": csr_matrix((np.r_[branch.rab, branch.rab], (row, col))),
                "ac": csr_matrix((np.r_[branch.rac, branch.rac], (row, col))),
                "bb": csr_matrix((np.r_[branch.rbb, branch.rbb], (row, col))),
                "bc": csr_matrix((np.r_[branch.rbc, branch.rbc], (row, col))),
                "cc": csr_matrix((np.r_[branch.rcc, branch.rcc], (row, col))),
            }
            x = {
                "aa": csr_matrix((np.r_[branch.xaa, branch.xaa], (row, col))),
                "ab": csr_matrix((np.r_[branch.xab, branch.xab], (row, col))),
                "ac": csr_matrix((np.r_[branch.xac, branch.xac], (row, col))),
                "bb": csr_matrix((np.r_[branch.xbb, branch.xbb], (row, col))),
                "bc": csr_matrix((np.r_[branch.xbc, branch.xbc], (row, col))),
                "cc": csr_matrix((np.r_[branch.xcc, branch.xcc], (row, col))),
            }
            return r, x

        def basic_var_length(self, branch):
            a_indices = branch.phases.str.contains("a")
            b_indices = branch.phases.str.contains("a")
            c_indices = branch.phases.str.contains("a")
            line_a = branch.loc[a_indices, ["fb", "tb"]].values
            line_b = branch.loc[b_indices, ["fb", "tb"]].values
            line_c = branch.loc[c_indices, ["fb", "tb"]].values
            nl_a = len(line_a)
            nl_b = len(line_b)
            nl_c = len(line_c)
            basic_length = 3 * nl_a + 3 * nl_b + 3 * nl_c + 3
            return nl_a, nl_b, nl_c, line_a, line_b, line_c, basic_length

        def _variable_tables(self, branch):
            nl_a = self.nl_a
            nl_b = self.nl_b
            nl_c = self.nl_c
            line_a = self.line_a
            line_b = self.line_b
            line_c = self.line_c

            g = nx.Graph()
            g_a = nx.Graph()
            g_b = nx.Graph()
            g_c = nx.Graph()
            g.add_edges_from(branch[["fb", "tb"]].values.astype(int) - 1)
            g_a.add_edges_from(line_a.astype(int) - 1)
            g_b.add_edges_from(line_b.astype(int) - 1)
            g_c.add_edges_from(line_c.astype(int) - 1)
            t_a = np.array([])
            t_b = np.array([])
            t_c = np.array([])
            if len(g_a.nodes) > 0:
                t_a = np.array(list(nx.dfs_edges(g_a, source=0)))
            if len(g_b.nodes) > 0:
                t_b = np.array(list(nx.dfs_edges(g_b, source=0)))
            if len(g_c.nodes) > 0:
                t_c = np.array(list(nx.dfs_edges(g_c, source=0)))

            p_a_end = 1 * nl_a
            q_a_end = 2 * nl_a
            v_a_end = 3 * nl_a + 1
            p_b_end = v_a_end + 1 * nl_b
            q_b_end = v_a_end + 2 * nl_b
            v_b_end = v_a_end + 3 * nl_b + 1
            p_c_end = v_b_end + 1 * nl_c
            q_c_end = v_b_end + 2 * nl_c
            v_c_end = v_b_end + 3 * nl_c + 1
            x_maps = {}

            for t in range(LinDistModelQ.n_steps):
                df_a_t = pd.DataFrame(columns=["bi", "bj", "pij", "qij", "vi", "vj"])
                df_b_t = pd.DataFrame(columns=["bi", "bj", "pij", "qij", "vi", "vj"])
                df_c_t = pd.DataFrame(columns=["bi", "bj", "pij", "qij", "vi", "vj"])
                if len(g_a.nodes) > 0:
                    df_a_t = pd.DataFrame(
                        {
                            "bi": t_a[:, 0],
                            "bj": t_a[:, 1],
                            "pij": np.array(
                                [i + t * self.period for i in range(p_a_end)]
                            ),
                            "qij": np.array(
                                [i + t * self.period for i in range(p_a_end, q_a_end)]
                            ),
                            "vi": np.zeros_like(t_a[:, 0]),
                            "vj": np.array(
                                [
                                    i + t * self.period
                                    for i in range(q_a_end + 1, v_a_end)
                                ]
                            ),
                        },
                        dtype=np.int32,
                    )
                    df_a_t.loc[0, "vi"] = df_a_t.at[0, "vj"] - 1
                    for i in df_a_t.bi.values[1:]:
                        df_a_t.loc[df_a_t.loc[:, "bi"] == i, "vi"] = df_a_t.loc[
                            df_a_t.bj == i, "vj"
                        ].values[0]
                if len(g_b.nodes) > 0:
                    df_b_t = pd.DataFrame(
                        {
                            "bi": t_b[:, 0],
                            "bj": t_b[:, 1],
                            "pij": np.array(
                                [i + t * self.period for i in range(v_a_end, p_b_end)]
                            ),
                            "qij": np.array(
                                [i + t * self.period for i in range(p_b_end, q_b_end)]
                            ),
                            "vi": np.zeros_like(t_b[:, 0]),
                            "vj": np.array(
                                [
                                    i + t * self.period
                                    for i in range(q_b_end + 1, v_b_end)
                                ]
                            ),
                        },
                        dtype=np.int32,
                    )
                    df_b_t.loc[0, "vi"] = df_b_t.at[0, "vj"] - 1
                    for i in df_b_t.bi.values[1:]:
                        df_b_t.loc[df_b_t.loc[:, "bi"] == i, "vi"] = df_b_t.loc[
                            df_b_t.bj == i, "vj"
                        ].values[0]
                if len(g_c.nodes) > 0:
                    df_c_t = pd.DataFrame(
                        {
                            "bi": t_c[:, 0],
                            "bj": t_c[:, 1],
                            "pij": [
                                i + t * self.period for i in range(v_b_end, p_c_end)
                            ],
                            "qij": [
                                i + t * self.period for i in range(p_c_end, q_c_end)
                            ],
                            "vi": np.zeros_like(t_c[:, 0]),
                            "vj": [
                                i + t * self.period for i in range(q_c_end + 1, v_c_end)
                            ],
                        },
                        dtype=np.int32,
                    )
                    df_c_t.loc[0, "vi"] = df_c_t.at[0, "vj"] - 1
                    for i in df_c_t.bi.values[1:]:
                        df_c_t.loc[df_c_t.loc[:, "bi"] == i, "vi"] = df_c_t.loc[
                            df_c_t.bj == i, "vj"
                        ].values[0]

                ctr_var_start_idx = v_c_end
                x_maps_t = {"a": df_a_t, "b": df_b_t, "c": df_c_t}
                x_maps[t] = x_maps_t
            return x_maps, ctr_var_start_idx

        def _control_variables(self, der_bus, ctr_var_start_idx, battery_bus):
            ctr_var_start_idx = int(ctr_var_start_idx)
            if LinDistModelQ.der:
                ng_a = len(der_bus["a"])
                ng_b = len(der_bus["b"])
                ng_c = len(der_bus["c"])
                p_der_start_phase_idx = {}
                q_der_start_phase_idx = {}
            if LinDistModelQ.battery:
                nb_a = len(battery_bus["a"])
                nb_b = len(battery_bus["b"])
                nb_c = len(battery_bus["c"])
                pd_bat_start_phase_idx = {}
                pc_bat_start_phase_idx = {}
                b_bat_start_phase_idx = {}
            if LinDistModelQ.der and LinDistModelQ.battery:
                for t in range(LinDistModelQ.n_steps):
                    p_der_start_phase_idx_t = {
                        "a": ctr_var_start_idx + t * self.period,
                        "b": ctr_var_start_idx + t * self.period,
                        "c": ctr_var_start_idx + t * self.period,
                    }
                    p_der_start_phase_idx[t] = p_der_start_phase_idx_t
                    q_der_start_phase_idx_t = {
                        "a": ctr_var_start_idx + t * self.period,
                        "b": ctr_var_start_idx + ng_a + t * self.period,
                        "c": ctr_var_start_idx + ng_a + ng_b + t * self.period,
                    }
                    q_der_start_phase_idx[t] = q_der_start_phase_idx_t
                    pd_bat_start_phase_idx_t = {
                        "a": q_der_start_phase_idx_t["c"] + ng_c,
                        "b": q_der_start_phase_idx_t["c"] + ng_c + nb_a,
                        "c": q_der_start_phase_idx_t["c"] + ng_c + nb_a + nb_b,
                    }
                    pd_bat_start_phase_idx[t] = pd_bat_start_phase_idx_t
                    pc_bat_start_phase_idx_t = {
                        "a": pd_bat_start_phase_idx_t["c"] + nb_c,
                        "b": pd_bat_start_phase_idx_t["c"] + nb_c + nb_a,
                        "c": pd_bat_start_phase_idx_t["c"] + nb_c + nb_a + nb_b,
                    }
                    pc_bat_start_phase_idx[t] = pc_bat_start_phase_idx_t
                    b_bat_start_phase_idx_t = {
                        "a": pc_bat_start_phase_idx_t["c"] + nb_c,
                        "b": pc_bat_start_phase_idx_t["c"] + nb_c + nb_a,
                        "c": pc_bat_start_phase_idx_t["c"] + nb_c + nb_a + nb_b,
                    }
                    b_bat_start_phase_idx[t] = b_bat_start_phase_idx_t
                    load_control_start_idx = b_bat_start_phase_idx_t["c"] + nb_c
                    n_controlled_load_nodes = sum(self.bus.bus_type == PQ_FREE)
                    p_load_controlled_t = {
                        "a": load_control_start_idx,
                        "b": load_control_start_idx + n_controlled_load_nodes,
                        "c": load_control_start_idx + n_controlled_load_nodes * 2,
                    }
                    p_load_controlled[t] = p_load_controlled_t
                    q_load_controlled_t = {
                        "a": load_control_start_idx + n_controlled_load_nodes * 3,
                        "b": load_control_start_idx + n_controlled_load_nodes * 4,
                        "c": load_control_start_idx + n_controlled_load_nodes * 5,
                    }
                    q_load_controlled[t] = q_load_controlled_t
                return (
                    p_der_start_phase_idx,
                    q_der_start_phase_idx,
                    pd_bat_start_phase_idx,
                    pc_bat_start_phase_idx,
                    b_bat_start_phase_idx,
                    p_load_controlled,
                    q_load_controlled,
                )
            elif LinDistModelQ.der:
                for t in range(LinDistModelQ.n_steps):
                    p_der_start_phase_idx_t = {
                        "a": ctr_var_start_idx + t * self.period,
                        "b": ctr_var_start_idx + t * self.period,
                        "c": ctr_var_start_idx + t * self.period,
                    }
                    p_der_start_phase_idx[t] = p_der_start_phase_idx_t
                    q_der_start_phase_idx_t = {
                        "a": ctr_var_start_idx + t * self.period,
                        "b": ctr_var_start_idx + ng_a + t * self.period,
                        "c": ctr_var_start_idx + ng_a + ng_b + t * self.period,
                    }
                    q_der_start_phase_idx[t] = q_der_start_phase_idx_t
                    load_control_start_idx = q_der_start_phase_idx_t["c"] + ng_c
                    n_controlled_load_nodes = sum(self.bus.bus_type == PQ_FREE)
                    p_load_controlled_t = {
                        "a": load_control_start_idx,
                        "b": load_control_start_idx + n_controlled_load_nodes,
                        "c": load_control_start_idx + n_controlled_load_nodes * 2,
                    }
                    p_load_controlled[t] = p_load_controlled_t
                    q_load_controlled_t = {
                        "a": load_control_start_idx + n_controlled_load_nodes * 3,
                        "b": load_control_start_idx + n_controlled_load_nodes * 4,
                        "c": load_control_start_idx + n_controlled_load_nodes * 5,
                    }
                    q_load_controlled[t] = q_load_controlled_t
                return (
                    p_der_start_phase_idx,
                    q_der_start_phase_idx,
                    p_load_controlled,
                    q_load_controlled,
                )
            elif LinDistModelQ.battery:
                for t in range(LinDistModelQ.n_steps):
                    pd_bat_start_phase_idx_t = {
                        "a": ctr_var_start_idx + t * self.period,
                        "b": ctr_var_start_idx + nb_a + t * self.period,
                        "c": ctr_var_start_idx + nb_a + nb_b + t * self.period,
                    }
                    pd_bat_start_phase_idx[t] = pd_bat_start_phase_idx_t
                    pc_bat_start_phase_idx_t = {
                        "a": pd_bat_start_phase_idx_t["c"] + nb_c,
                        "b": pd_bat_start_phase_idx_t["c"] + nb_c + nb_a,
                        "c": pd_bat_start_phase_idx_t["c"] + nb_c + nb_a + nb_b,
                    }
                    pc_bat_start_phase_idx[t] = pc_bat_start_phase_idx_t
                    b_bat_start_phase_idx_t = {
                        "a": pc_bat_start_phase_idx_t["c"] + nb_c,
                        "b": pc_bat_start_phase_idx_t["c"] + nb_c + nb_a,
                        "c": pc_bat_start_phase_idx_t["c"] + nb_c + nb_a + nb_b,
                    }
                    b_bat_start_phase_idx[t] = b_bat_start_phase_idx_t
                    load_control_start_idx = b_bat_start_phase_idx_t["c"] + nb_c
                    n_controlled_load_nodes = sum(self.bus.bus_type == PQ_FREE)
                    p_load_controlled_t = {
                        "a": load_control_start_idx,
                        "b": load_control_start_idx + n_controlled_load_nodes,
                        "c": load_control_start_idx + n_controlled_load_nodes * 2,
                    }
                    p_load_controlled[t] = p_load_controlled_t
                    q_load_controlled_t = {
                        "a": load_control_start_idx + n_controlled_load_nodes * 3,
                        "b": load_control_start_idx + n_controlled_load_nodes * 4,
                        "c": load_control_start_idx + n_controlled_load_nodes * 5,
                    }
                    q_load_controlled[t] = q_load_controlled_t
                return (
                    pd_bat_start_phase_idx,
                    pc_bat_start_phase_idx,
                    b_bat_start_phase_idx,
                    p_load_controlled,
                    q_load_controlled,
                )
            else:
                for t in range(LinDistModelQ.n_steps):
                    load_control_start_idx = ctr_var_start_idx + t * self.period
                    n_controlled_load_nodes = sum(self.bus.bus_type == PQ_FREE)
                    p_load_controlled_t = {
                        "a": load_control_start_idx,
                        "b": load_control_start_idx + n_controlled_load_nodes,
                        "c": load_control_start_idx + n_controlled_load_nodes * 2,
                    }
                    p_load_controlled[t] = p_load_controlled_t
                    q_load_controlled_t = {
                        "a": load_control_start_idx + n_controlled_load_nodes * 3,
                        "b": load_control_start_idx + n_controlled_load_nodes * 4,
                        "c": load_control_start_idx + n_controlled_load_nodes * 5,
                    }
                    q_load_controlled[t] = q_load_controlled_t
                return p_load_controlled, q_load_controlled

        def init_bounds(self, bus, gen, bat):
            default = (
                100e3  # default to very large value for variables that are not bounded.
            )
            x_maps = self.x_maps
            pv_shape = self.pv_loadshape
            # ~~~~~~~~~~ x limits ~~~~~~~~~~
            x_lim_lower = np.ones(self.n_x) * -default
            x_lim_upper = np.ones(self.n_x) * default
            for t in range(LinDistModelQ.n_steps):
                for ph in "abc":
                    if self.phase_exists(ph, t):
                        x_lim_lower[x_maps[t][ph].loc[:, "pij"]] = -default  # P
                        x_lim_upper[x_maps[t][ph].loc[:, "pij"]] = default  # P
                        x_lim_lower[x_maps[t][ph].loc[:, "qij"]] = -default  # Q
                        x_lim_upper[x_maps[t][ph].loc[:, "qij"]] = default  # Q
                        # ~~ v limits ~~:
                        i_root = list(set(x_maps["a"].bi) - set(x_maps["a"].bj))[0]
                        i_v_swing = (
                            x_maps[t][ph]
                            .loc[x_maps[t][ph].loc[:, "bi"] == self.swing_bus, "vi"]
                            .to_numpy()[0]
                        )
                        x_lim_lower[i_v_swing] = bus.loc[self.swing_bus, "v_min"] ** 2
                        x_lim_upper[i_v_swing] = bus.loc[self.swing_bus, "v_max"] ** 2
                        x_lim_lower[x_maps[t][ph].loc[:, "vj"]] = (
                            bus.loc[x_maps[t][ph].loc[:, "bj"], "v_min"] ** 2
                        )
                        x_lim_upper[x_maps[t][ph].loc[:, "vj"]] = (
                            bus.loc[x_maps[t][ph].loc[:, "bj"], "v_max"] ** 2
                        )
                        # ~~ DER limits  ~~:
                        if LinDistModelQ.der:
                            for i in range(self.gen_buses[ph].shape[0]):
                                i_q = self.q_der_start_phase_idx[t][ph] + i
                                # reactive power bounds
                                s_rated = gen["s" + ph + "_max"]
                                p_out = gen["p" + ph] * pv_shape["PV"][t]
                                q_min = -sqrt((s_rated**2) - (p_out**2))
                                q_max = sqrt((s_rated**2) - (p_out**2))
                                x_lim_lower[i_q] = q_min[self.gen_buses[ph][i]]
                                x_lim_upper[i_q] = q_max[self.gen_buses[ph][i]]
                        # ~~ Battery limits ~~:
                        if LinDistModelQ.battery:
                            for i in range(self.bat_buses[ph].shape[0]):
                                pb_max = bat["Pb_max_" + ph]
                                b_min = bat["bmin_" + ph]
                                b_max = bat["bmax_" + ph]
                                i_d = self.pd_bat_start_phase_idx[t][ph] + i
                                i_c = self.pc_bat_start_phase_idx[t][ph] + i
                                i_b = self.b_bat_start_phase_idx[t][ph] + i
                                # battery active power charge/discharge and s.o.c bounds
                                x_lim_lower[i_d] = 0
                                x_lim_lower[i_c] = 0
                                x_lim_lower[i_b] = b_min[self.bat_buses[ph][i]]
                                x_lim_upper[i_d] = pb_max[self.bat_buses[ph][i]]
                                x_lim_upper[i_c] = pb_max[self.bat_buses[ph][i]]
                                x_lim_upper[i_b] = b_max[self.bat_buses[ph][i]]
            bounds = [(l, u) for (l, u) in zip(x_lim_lower, x_lim_upper)]
            return bounds

        @cache
        def branch_into_j(self, var, j, phase, t):
            return (
                self.x_maps[t][phase].loc[self.x_maps[t][phase].bj == j, var].to_numpy()
            )

        @cache
        def branches_out_of_j(self, var, j, phase, t):
            return (
                self.x_maps[t][phase].loc[self.x_maps[t][phase].bi == j, var].to_numpy()
            )

        @cache
        def idx(self, var, node_j, phase, t):
            if var == "pg":  # active power generation at node
                raise ValueError("pg is fixed and is not a valid variable.")
            if var == "qg":  # reactive power generation at node
                raise ValueError("pq is fixed and is not a valid variable.")
            if var == "qg":
                if node_j in set(self.gen_buses[phase]):
                    return (
                        self.q_der_start_phase_idx[t][phase]
                        + np.where(self.gen_buses[phase] == node_j)[0]
                    )
                return []
            if var == "pl":  # active power exported at node (not root node)
                if node_j in set(self.controlled_load_buses[phase]):
                    return (
                        self.p_load_controlled_idxs[t][phase]
                        + np.where(self.controlled_load_buses[phase] == node_j)[0]
                    )
            if var == "ql":  # reactive power exported at node (not root node)
                if node_j in set(self.controlled_load_buses[phase]):
                    return (
                        self.q_load_controlled_idxs[t][phase]
                        + np.where(self.controlled_load_buses[phase] == node_j)[0]
                    )
            if var == "pd":
                if node_j in set(self.bat_bus[phase]):
                    return (
                        self.pd_bat_start_phase_idx[t][phase]
                        + np.where(self.bat_bus[phase] == node_j)[0]
                    )
                return []
            if var == "pc":
                if node_j in set(self.bat_bus[phase]):
                    return (
                        self.pc_bat_start_phase_idx[t][phase]
                        + np.where(self.bat_bus[phase] == node_j)[0]
                    )
                return []
            if var == "b":
                if node_j in set(self.bat_bus[phase]):
                    return (
                        self.b_bat_start_phase_idx[t][phase]
                        + np.where(self.bat_bus[phase] == node_j)[0]
                    )
                return []
            return self.branch_into_j(var, node_j, phase, t)

        @cache
        def _row(self, var, index, phase, t):
            return self.idx(var, index, phase, t)

        @cache
        def phase_exists(self, phase, t, index: int = None):
            if index is None:
                return self.x_maps[t][phase].shape[0] > 0
            return len(self.idx("bj", index, phase, t)) > 0

        def create_model(self):
            r, x = self.r, self.x
            bus = self.bus

            # ########## Aeq and Beq Formation ###########
            n_rows = self.row_no
            n_col = self.n_x
            a_eq = zeros(
                (n_rows, n_col)
            )  # Aeq has the same number of rows as equations with a column for each x
            b_eq = zeros(n_rows)
            a_ineq = zeros((n_rows, n_col))
            b_ineq = zeros(n_rows)

            for t in range(LinDistModelQ.n_steps):
                for j in range(0, self.nb):

                    def col(var, phase):
                        return self.idx(var, j, phase, t)

                    def coll(var, phase):
                        return self.idx(var, j, phase, t - 1)

                    def row(var, phase):
                        return self._row(var, j, phase, t)

                    def children(var, phase):
                        return self.branches_out_of_j(var, j, phase, t)

                    for ph in ["abc", "bca", "cab"]:
                        a, b, c = ph[0], ph[1], ph[2]
                        aa = "".join(sorted(a + a))
                        ab = "".join(
                            sorted(a + b)
                        )  # if ph=='cab', then a+b=='ca'. Sort so ab=='ac'
                        ac = "".join(sorted(a + c))
                        if not self.phase_exists(a, t, j):
                            continue
                        p_load, q_load = 0, 0
                        reg_ratio = 1
                        q_cap = 0
                        if bus.bus_type[j] == "PQ":
                            p_load = self.bus[f"pl_{a}"][j]
                            q_load = self.bus[f"ql_{a}"][j]
                        p_gen = get(self.gen[f"p{a}"], j, 0)  # .get(j, 0)
                        q_gen = get(self.gen[f"q{a}"], j, 0)  # .get(j, 0)
                        if self.cap is not None:
                            q_cap = get(self.cap[f"q{a}"], j, 0)  # .get(j, 0)
                        if self.reg is not None:
                            reg_ratio = get(self.reg[f"ratio_{a}"], j, 1)  # .get(j, 1)
                        # equation indexes
                        p_eqn = row("pij", a)
                        q_eqn = row("qij", a)
                        v_eqn = row("vj", a)
                        a_eq[p_eqn, col("pij", a)] = 1
                        a_eq[p_eqn, col("vj", a)] = (
                            -(bus.cvr_p[j] / 2) * p_load * loadshape["M"][t]
                        )
                        a_eq[p_eqn, children("pij", a)] = -1
                        b_eq[p_eqn] = (
                            (1 - (bus.cvr_p[j] / 2)) * p_load * loadshape["M"][t]
                        )
                        # Q equation
                        a_eq[q_eqn, col("qij", a)] = 1
                        a_eq[q_eqn, col("vj", a)] = -(bus.cvr_q[j] / 2) * q_load
                        a_eq[q_eqn, children("qij", a)] = -1
                        b_eq[q_eqn] = (1 - (bus.cvr_q[j] / 2)) * q_load - q_cap

                        # V equation
                        i = self.idx("bi", j, a, t)[0]
                        if i == self.swing_bus:  # Swing bus
                            a_eq[row("vi", a), col("vi", a)] = 1
                            b_eq[row("vi", a)] = (
                                bus.loc[bus.bus_type == "SWING", f"v_{a}"][0] ** 2
                            )
                        a_eq[v_eqn, col("vj", a)] = 1
                        a_eq[v_eqn, col("vi", a)] = -1 * reg_ratio
                        a_eq[v_eqn, col("pij", a)] = 2 * r[aa][i, j]
                        a_eq[v_eqn, col("qij", a)] = 2 * x[aa][i, j]
                        if self.phase_exists(b, j):
                            a_eq[v_eqn, col("pij", b)] = (
                                -r[ab][i, j] + sqrt(3) * x[ab][i, j]
                            )
                            a_eq[v_eqn, col("qij", b)] = (
                                -x[ab][i, j] - sqrt(3) * r[ab][i, j]
                            )
                        if self.phase_exists(c, j):
                            a_eq[v_eqn, col("pij", c)] = (
                                -r[ac][i, j] - sqrt(3) * x[ac][i, j]
                            )
                            a_eq[v_eqn, col("qij", c)] = (
                                -x[ac][i, j] + sqrt(3) * r[ac][i, j]
                            )
                        if LinDistModelQ.der:
                            a_eq[q_eqn, col("qg", a)] = 1
                            b_eq[p_eqn] = (1 - (bus.cvr_p[j] / 2)) * p_load * loadshape[
                                "M"
                            ][t] - p_gen * pv_shape["PV"][t]
                            b_eq[q_eqn] = (1 - (bus.cvr_q[j] / 2)) * q_load - q_cap
                        if LinDistModelQ.battery:
                            b_eqn = row("b", a)
                            pd_eqn = row("pd", a)
                            nc = bat["nc_" + a].get(j, 0)
                            nd = bat["nd_" + a].get(j, float("inf"))
                            b_initial = bat["bmin_" + a].get(j, 0)
                            a_eq[p_eqn, col("pd", a)] = 1
                            a_eq[p_eqn, col("pc", a)] = -1
                            a_eq[b_eqn, col("b", a)] = 1
                            a_eq[b_eqn, col("pc", a)] = -nc
                            a_eq[b_eqn, col("pd", a)] = 1 / nd
                            if t == 0:
                                b_eq[b_eqn] = b_initial
                            else:
                                # b_eq[row("b", a)] = 0
                                a_eq[b_eqn, coll("b", a)] = -1

                            a_ineq[pd_eqn, col("pd", a)] = 1
                            a_ineq[pd_eqn, col("pc", a)] = -1
                            b_ineq[pd_eqn] = bat["hmax_" + a].get(j, 0)
                        if bus.bus_type[j] == PQ_FREE:
                            a_eq[p_eqn, col("pl", a)] = -1
                            a_eq[q_eqn, col("ql", a)] = -1
            return (
                a_eq,
                b_eq,
                a_ineq,
                b_ineq,
            )

        def fast_model_update(
            self,
            bus_data: pd.DataFrame = None,
            gen_data: pd.DataFrame = None,
            cap_data: pd.DataFrame = None,
            reg_data: pd.DataFrame = None,
        ):
            # ~~~~~~~~~~~~~~~~~~~~ Load Model ~~~~~~~~~~~~~~~~~~~~
            bus_update = bus_data is not None
            gen_update = gen_data is not None
            cap_update = cap_data is not None
            reg_update = reg_data is not None

            if bus_data is not None:
                self.bus = bus_data.sort_values(by="id", ignore_index=True)
                self.bus.index = self.bus.id - 1
            if gen_data is not None:
                self.gen = gen_data.sort_values(by="id", ignore_index=True)
                self.gen.index = self.gen.id - 1
                if self.gen.shape[0] == 0:
                    self.gen = self.gen.reindex(index=[-1])
            if cap_data is not None:
                self.cap = cap_data.sort_values(by="id", ignore_index=True)
                self.cap.index = self.cap.id - 1
                if self.cap.shape[0] == 0:
                    self.cap = self.cap.reindex(index=[-1])
            if reg_data is not None:
                self.reg = reg_data.sort_values(by="tb", ignore_index=True)
                self.reg.index = self.reg.tb - 1
                if self.reg.shape[0] == 0:
                    self.reg = self.reg.reindex(index=[-1])
                self.reg["ratio_a"] = 1 + 0.00625 * self.reg.tap_a
                self.reg["ratio_b"] = 1 + 0.00625 * self.reg.tap_b
                self.reg["ratio_c"] = 1 + 0.00625 * self.reg.tap_c
            self.nb = len(self.bus.id)
            # ~~~~~~~~~~~~~~~~~~~~ initialize Aeq and beq and objective gradient ~~~~~~~~~~~~~~~~~~~~
            self.a_eq, self.b_eq = self._fast_update_aeq_beq(
                bus_update=bus_update,
                gen_update=gen_update,
                cap_update=cap_update,
                reg_update=reg_update,
            )
            self.bounds = self.init_bounds(self.bus, self.gen)

        def _fast_update_aeq_beq(
            self, bus_update=False, gen_update=False, cap_update=False, reg_update=False
        ):
            bus = self.bus
            # ########## Aeq and Beq Formation ###########
            n_rows = self.ctr_var_start_idx
            n_col = self.n_x
            a_eq = zeros(
                (n_rows, n_col)
            )  # Aeq has the same number of rows as equations with a column for each x
            b_eq = zeros(n_rows)
            for j in range(1, self.nb):

                def col(var, phase):
                    return self.idx(var, j, phase)

                def row(var, phase):
                    return self._row(var, j, phase)

                for ph in ["abc", "bca", "cab"]:
                    a = ph[0]
                    if not self.phase_exists(a, j):
                        continue
                    p_load, q_load = 0, 0
                    reg_ratio = 1
                    q_cap = 0
                    if bus.bus_type[j] == "PQ":
                        p_load = self.bus[f"pl_{a}"][j]
                        q_load = self.bus[f"ql_{a}"][j]
                    p_gen = self.gen[f"p{a}"].get(j, 0)
                    q_gen = self.gen[f"q{a}"].get(j, 0)
                    if self.cap is not None:
                        q_cap = self.cap[f"q{a}"].get(j, 0)
                    if self.reg is not None:
                        reg_ratio = self.reg[f"ratio_{a}"].get(j, 1)
                    # equation indexes
                    p_eqn = row("pij", a)
                    q_eqn = row("qij", a)
                    v_eqn = row("vj", a)
                    # Set P equation variable coefficients in a_eq
                    if bus_update:
                        a_eq[p_eqn, col("vj", a)] = -(bus.cvr_p[j] / 2) * p_load
                    # Set P equation constant in b_eq
                    if bus_update or gen_update:
                        b_eq[p_eqn] = (1 - (bus.cvr_p[j] / 2)) * p_load - p_gen
                    # Set Q equation variable coefficients in a_eq
                    if bus_update or cap_update:
                        a_eq[q_eqn, col("vj", a)] = -(bus.cvr_q[j] / 2) * q_load - q_cap
                    # Set Q equation constant in b_eq
                    if bus_update or gen_update:
                        b_eq[q_eqn] = (1 - (bus.cvr_q[j] / 2)) * q_load - q_gen

                    # Set V equation variable coefficients in a_eq and constants in b_eq

                    if reg_update:
                        a_eq[v_eqn, col("vi", a)] = -1 * reg_ratio**2
                    if bus_update:
                        i = self.idx("bi", j, a)[0]
                        if bus.bus_type[i] == "SWING":  # Swing bus
                            a_eq[row("vi", a), col("vi", a)] = 1
                            b_eq[row("vi", a)] = (
                                bus.loc[bus.bus_type == "SWING", f"v_{a}"][0] ** 2
                            )
                        if bus.bus_type[i] == PQ_FREE:
                            a_eq[row("vi", a), col("vi", a)] = 0
                            b_eq[row("vi", a)] = 0
                        # boundary p and q
                        # if j in self.down_buses[a]:
                        if bus.bus_type[j] == PQ_FREE:
                            a_eq[p_eqn, col("pl", a)] = -1
                            a_eq[q_eqn, col("ql", a)] = -1

            return a_eq, b_eq

        def get_decision_variables(self, x):
            ng_a = len(self.gen_buses["a"])
            ng_b = len(self.gen_buses["b"])
            ng_c = len(self.gen_buses["c"])
            nb_a = len(self.gen_buses["a"])
            nb_b = len(self.gen_buses["b"])
            nb_c = len(self.gen_buses["c"])
            ng = dict(a=ng_a, b=ng_b, c=ng_c)
            nbat = dict(a=nb_a, b=nb_b, c=nb_c)
            if LinDistModelQ.der and LinDistModelQ.battery:
                qi = self.q_der_start_phase_idx
                pdi = self.pd_bat_start_phase_idx
                pci = self.pc_bat_start_phase_idx
                bi = self.b_bat_start_phase_idx
                dec_var = {}
                dec_d_var = {}
                dec_c_var = {}
                dec_b_var = {}
                for t in range(LinDistModelQ.n_steps):
                    dec_var_t = pd.DataFrame(columns=["name", "a", "b", "c"])
                    dec_d_var_t = pd.DataFrame(0, columns=["name", "a", "b", "c"])
                    dec_c_var_t = pd.DataFrame(0, columns=["name", "a", "b", "c"])
                    dec_b_var_t = pd.DataFrame(0, columns=["name", "a", "b", "c"])
                    for ph in "abc":
                        for i_gen in range(ng[ph]):
                            i = self.gen_buses[ph][i_gen]
                            dec_var_t.at[i + 1, "name"] = self.bus.at[i, "name"]
                            dec_var_t.at[i + 1, ph] = x[qi[t][ph] + i_gen]
                        for i_bat in range(nbat[ph]):
                            j = self.bat_buses[ph][i_bat]
                            dec_d_var_t.at[j + 1, "name"] = self.bus.at[j, "name"]
                            dec_c_var_t.at[j + 1, "name"] = self.bus.at[j, "name"]
                            dec_b_var_t.at[j + 1, "name"] = self.bus.at[j, "name"]
                            dec_d_var_t.at[j + 1, ph] = x[pdi[t][ph] + i_bat]
                            dec_c_var_t.at[j + 1, ph] = x[pci[t][ph] + i_bat]
                            dec_b_var_t.at[j + 1, ph] = x[bi[t][ph] + i_bat]
                    dec_var[t] = dec_var_t
                    dec_d_var[t] = dec_d_var_t
                    dec_c_var[t] = dec_c_var_t
                    dec_b_var[t] = dec_b_var_t
                return dec_var, dec_d_var, dec_c_var, dec_b_var
            elif LinDistModelQ.der:
                qi = self.q_der_start_phase_idx
                dec_var = {}
                for t in range(LinDistModelQ.n_steps):
                    dec_var_t = pd.DataFrame(columns=["name", "a", "b", "c"])
                    for ph in "abc":
                        for i_gen in range(ng[ph]):
                            i = self.gen_buses[ph][i_gen]
                            dec_var_t.at[i + 1, "name"] = self.bus.at[i, "name"]
                            dec_var_t.at[i + 1, ph] = x[qi[t][ph] + i_gen]
                    dec_var[t] = dec_var_t
                return dec_var
            elif LinDistModelQ.battery:
                pdi = self.pd_bat_start_phase_idx
                pci = self.pc_bat_start_phase_idx
                bi = self.b_bat_start_phase_idx
                dec_d_var = {}
                dec_c_var = {}
                dec_b_var = {}
                for t in range(LinDistModelQ.n_steps):
                    dec_d_var_t = pd.DataFrame(0, columns=["name", "a", "b", "c"])
                    dec_c_var_t = pd.DataFrame(0, columns=["name", "a", "b", "c"])
                    dec_b_var_t = pd.DataFrame(0, columns=["name", "a", "b", "c"])
                    for ph in "abc":
                        for i_bat in range(nbat[ph]):
                            i = self.bat_buses[ph][i_bat]
                            dec_d_var_t.at[i + 1, "name"] = self.bus.at[i, "name"]
                            dec_c_var_t.at[i + 1, "name"] = self.bus.at[i, "name"]
                            dec_b_var_t.at[i + 1, "name"] = self.bus.at[i, "name"]
                            dec_d_var_t.at[i + 1, ph] = x[pdi[t][ph] + i_bat]
                            dec_c_var_t.at[i + 1, ph] = x[pci[t][ph] + i_bat]
                            dec_b_var_t.at[i + 1, ph] = x[bi[t][ph] + i_bat]
                    dec_d_var[t] = dec_d_var_t
                    dec_c_var[t] = dec_c_var_t
                    dec_b_var[t] = dec_b_var_t
                return dec_d_var, dec_c_var, dec_b_var
            else:
                return None

        def get_voltages(self, x):
            v_df = {}
            for t in range(LinDistModelQ.n_steps):
                v_df_t = pd.DataFrame(
                    columns=["name", "a", "b", "c"],
                    index=np.array(range(1, self.nb + 1)),
                )
                v_df_t["name"] = self.bus["name"].to_numpy()
                for ph in "abc":
                    if not self.phase_exists(ph):
                        v_df_t.loc[:, ph] = 0.0
                        continue
                    v_df_t.loc[self.swing_bus + 1, ph] = np.sqrt(
                        x[self.x_maps[t][ph].vi[0]].astype(np.float64)
                    )
                    v_df_t.loc[self.x_maps[t][ph].bj.values + 1, ph] = np.sqrt(
                        x[self.x_maps[t][ph].vj.values].astype(np.float64)
                    )
                v_df[t] = v_df_t
            return v_df

        def get_apparent_power_flows(self, x):
            s_df = {}
            for t in range(LinDistModelQ.n_steps):
                s_df_t = pd.DataFrame(
                    columns=["fb", "tb", "a", "b", "c"], index=range(1, self.nb + 1)
                )
                s_df_t["a"] = s_df_t["a"].astype(complex)
                s_df_t["b"] = s_df_t["b"].astype(complex)
                s_df_t["c"] = s_df_t["c"].astype(complex)
                for ph in "abc":
                    fb_idxs = self.x_maps[ph].bi.values
                    fb_names = self.bus.name[fb_idxs].to_numpy()
                    tb_idxs = self.x_maps[ph].bj.values
                    tb_names = self.bus.name[tb_idxs].to_numpy()
                    s_df_t.loc[self.x_maps[t][ph].bj.values + 1, "fb"] = fb_names
                    s_df_t.loc[self.x_maps[t][ph].bj.values + 1, "tb"] = tb_names
                    s_df_t.loc[self.x_maps[t][ph].bj.values + 1, ph] = (
                        x[self.x_maps[t][ph].pij] + 1j * x[self.x_maps[t][ph].qij]
                    )
                s_df[t] = s_df_t
            return s_df

        @property
        def branch_data(self):
            return self.branch

        @property
        def bus_data(self):
            return self.bus

        @property
        def gen_data(self):
            return self.gen

        @property
        def cap_data(self):
            return self.cap

        @property
        def reg_data(self):
            return self.reg
