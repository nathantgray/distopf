from functools import cache

import networkx as nx
import numpy as np
import pandas as pd
from numpy import sqrt, zeros
from scipy.sparse import csr_matrix
from distopf.multiperiod.lindist_base_multi import LinDistModel, get

# bus_type options
SWING_FREE = "IN"
PQ_FREE = "OUT"
SWING_BUS = "SWING"
PQ_BUS = "PQ"


class LinDistModelP(LinDistModel):
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
        n: int = 24,
        der: bool = True,
        battery: bool = True,
    ):
        super().__init__(
            branch_data,
            bus_data,
            gen_data,
            cap_data=cap_data,
            reg_data=reg_data,
            loadshape_data=loadshape_data,
            pv_loadshape_data=pv_loadshape_data,
            bat_data=bat_data,
            n=n,
            der=der,
            battery=battery,
        )

        # def _control_variables(self, der_bus, ctr_var_start_idx, battery_bus):
        #     ctr_var_start_idx = int(ctr_var_start_idx)
        #     p_load_controlled_idx = {}
        #     q_load_controlled_idx = {}
        #     if LinDistModel.der:
        #         ng_a = len(der_bus["a"])
        #         ng_b = len(der_bus["b"])
        #         ng_c = len(der_bus["c"])
        #         p_der_start_phase_idx = {}
        #         q_der_start_phase_idx = {}
        #     if LinDistModel.battery:
        #         nb_a = len(battery_bus["a"])
        #         nb_b = len(battery_bus["b"])
        #         nb_c = len(battery_bus["c"])
        #         pd_bat_start_phase_idx = {}
        #         pc_bat_start_phase_idx = {}
        #         b_bat_start_phase_idx = {}
        #     if LinDistModel.der and LinDistModel.battery:
        #         for t in range(LinDistModel.n):
        #             p_der_start_phase_idx_t = {
        #                 "a": ctr_var_start_idx + t * self.period,
        #                 "b": ctr_var_start_idx + t * self.period,
        #                 "c": ctr_var_start_idx + t * self.period
        #             }
        #             p_der_start_phase_idx[t] = p_der_start_phase_idx_t
        #             q_der_start_phase_idx_t = {
        #                 "a": ctr_var_start_idx + t * self.period,
        #                 "b": ctr_var_start_idx + ng_a + t * self.period,
        #                 "c": ctr_var_start_idx + ng_a + ng_b + t * self.period,
        #             }
        #             q_der_start_phase_idx[t] = q_der_start_phase_idx_t
        #             pd_bat_start_phase_idx_t = {
        #                 "a": q_der_start_phase_idx_t["c"] + ng_c,
        #                 "b": q_der_start_phase_idx_t["c"] + ng_c + nb_a,
        #                 "c": q_der_start_phase_idx_t["c"] + ng_c + nb_a + nb_b,
        #             }
        #             pd_bat_start_phase_idx[t] = pd_bat_start_phase_idx_t
        #             pc_bat_start_phase_idx_t = {
        #                 "a": pd_bat_start_phase_idx_t["c"] + nb_c,
        #                 "b": pd_bat_start_phase_idx_t["c"] + nb_c + nb_a,
        #                 "c": pd_bat_start_phase_idx_t["c"] + nb_c + nb_a + nb_b
        #             }
        #             pc_bat_start_phase_idx[t] = pc_bat_start_phase_idx_t
        #             b_bat_start_phase_idx_t = {
        #                 "a": pc_bat_start_phase_idx_t["c"] + nb_c,
        #                 "b": pc_bat_start_phase_idx_t["c"] + nb_c + nb_a,
        #                 "c": pc_bat_start_phase_idx_t["c"] + nb_c + nb_a + nb_b
        #             }
        #             b_bat_start_phase_idx[t] = b_bat_start_phase_idx_t
        #             load_control_start_idx = b_bat_start_phase_idx_t["c"] + nb_c
        #             n_controlled_load_nodes = sum(self.bus.bus_type == PQ_FREE)
        #             p_load_controlled_idx_t = {
        #                 "a": load_control_start_idx,
        #                 "b": load_control_start_idx + n_controlled_load_nodes,
        #                 "c": load_control_start_idx + n_controlled_load_nodes * 2,
        #             }
        #             p_load_controlled_idx[t] = p_load_controlled_idx_t
        #             q_load_controlled_idx_t = {
        #                 "a": load_control_start_idx + n_controlled_load_nodes * 3,
        #                 "b": load_control_start_idx + n_controlled_load_nodes * 4,
        #                 "c": load_control_start_idx + n_controlled_load_nodes * 5,
        #             }
        #             q_load_controlled_idx[t] = q_load_controlled_idx_t
        #         return p_der_start_phase_idx, q_der_start_phase_idx, pd_bat_start_phase_idx, pc_bat_start_phase_idx, b_bat_start_phase_idx, p_load_controlled_idx, q_load_controlled_idx
        #     elif LinDistModel.der:
        #         for t in range(LinDistModel.n):
        #             p_der_start_phase_idx_t = {
        #                 "a": ctr_var_start_idx + t * self.period,
        #                 "b": ctr_var_start_idx + t * self.period,
        #                 "c": ctr_var_start_idx + t * self.period
        #             }
        #             p_der_start_phase_idx[t] = p_der_start_phase_idx_t
        #             q_der_start_phase_idx_t = {
        #                 "a": ctr_var_start_idx + t * self.period,
        #                 "b": ctr_var_start_idx + ng_a + t * self.period,
        #                 "c": ctr_var_start_idx + ng_a + ng_b + t * self.period,
        #             }
        #             q_der_start_phase_idx[t] = q_der_start_phase_idx_t
        #             load_control_start_idx = q_der_start_phase_idx_t["c"] + ng_c
        #             n_controlled_load_nodes = sum(self.bus.bus_type == PQ_FREE)
        #             p_load_controlled_idx_t = {
        #                 "a": load_control_start_idx,
        #                 "b": load_control_start_idx + n_controlled_load_nodes,
        #                 "c": load_control_start_idx + n_controlled_load_nodes * 2,
        #             }
        #             p_load_controlled_idx[t] = p_load_controlled_idx_t
        #             q_load_controlled_idx_t = {
        #                 "a": load_control_start_idx + n_controlled_load_nodes * 3,
        #                 "b": load_control_start_idx + n_controlled_load_nodes * 4,
        #                 "c": load_control_start_idx + n_controlled_load_nodes * 5,
        #             }
        #             q_load_controlled_idx[t] = q_load_controlled_idx_t
        #         return p_der_start_phase_idx, q_der_start_phase_idx, p_load_controlled_idx, q_load_controlled_idx
        #     elif LinDistModel.battery:
        #         for t in range(LinDistModel.n):
        #             pd_bat_start_phase_idx_t = {
        #                 "a": ctr_var_start_idx + t * self.period,
        #                 "b": ctr_var_start_idx + nb_a + t * self.period,
        #                 "c": ctr_var_start_idx + nb_a + nb_b + t * self.period,
        #             }
        #             pd_bat_start_phase_idx[t] = pd_bat_start_phase_idx_t
        #             pc_bat_start_phase_idx_t = {
        #                 "a": pd_bat_start_phase_idx_t["c"] + nb_c,
        #                 "b": pd_bat_start_phase_idx_t["c"] + nb_c + nb_a,
        #                 "c": pd_bat_start_phase_idx_t["c"] + nb_c + nb_a + nb_b
        #             }
        #             pc_bat_start_phase_idx[t] = pc_bat_start_phase_idx_t
        #             b_bat_start_phase_idx_t = {
        #                 "a": pc_bat_start_phase_idx_t["c"] + nb_c,
        #                 "b": pc_bat_start_phase_idx_t["c"] + nb_c + nb_a,
        #                 "c": pc_bat_start_phase_idx_t["c"] + nb_c + nb_a + nb_b
        #             }
        #             b_bat_start_phase_idx[t] = b_bat_start_phase_idx_t
        #             load_control_start_idx = b_bat_start_phase_idx_t["c"] + nb_c
        #             n_controlled_load_nodes = sum(self.bus.bus_type == PQ_FREE)
        #             p_load_controlled_idx_t = {
        #                 "a": load_control_start_idx,
        #                 "b": load_control_start_idx + n_controlled_load_nodes,
        #                 "c": load_control_start_idx + n_controlled_load_nodes * 2,
        #             }
        #             p_load_controlled_idx[t] = p_load_controlled_idx_t
        #             q_load_controlled_idx_t = {
        #                 "a": load_control_start_idx + n_controlled_load_nodes * 3,
        #                 "b": load_control_start_idx + n_controlled_load_nodes * 4,
        #                 "c": load_control_start_idx + n_controlled_load_nodes * 5,
        #             }
        #             q_load_controlled_idx[t] = q_load_controlled_idx_t
        #         return pd_bat_start_phase_idx, pc_bat_start_phase_idx, b_bat_start_phase_idx, p_load_controlled_idx, q_load_controlled_idx
        #     else:
        #         for t in range(LinDistModel.n):
        #             load_control_start_idx = ctr_var_start_idx + t * self.period
        #             n_controlled_load_nodes = sum(self.bus.bus_type == PQ_FREE)
        #             p_load_controlled_idx_t = {
        #                 "a": load_control_start_idx,
        #                 "b": load_control_start_idx + n_controlled_load_nodes,
        #                 "c": load_control_start_idx + n_controlled_load_nodes * 2,
        #             }
        #             p_load_controlled_idx[t] = p_load_controlled_idx_t
        #             q_load_controlled_idx_t = {
        #                 "a": load_control_start_idx + n_controlled_load_nodes * 3,
        #                 "b": load_control_start_idx + n_controlled_load_nodes * 4,
        #                 "c": load_control_start_idx + n_controlled_load_nodes * 5,
        #             }
        #             q_load_controlled_idx[t] = q_load_controlled_idx_t
        #         return p_load_controlled_idx, q_load_controlled_idx

        def init_bounds(self, bus, gen, bat):
            default = (
                100e3  # default to very large value for variables that are not bounded.
            )
            x_maps = self.x_maps
            pv_shape = self.pv_loadshape
            # ~~~~~~~~~~ x limits ~~~~~~~~~~
            x_lim_lower = np.ones(self.n_x) * -default
            x_lim_upper = np.ones(self.n_x) * default
            for t in range(self.n):
                for ph in "abc":
                    if self.phase_exists(ph):
                        x_lim_lower[x_maps[t][ph].loc[:, "pij"]] = -default  # P
                        x_lim_upper[x_maps[t][ph].loc[:, "pij"]] = default  # P
                        x_lim_lower[x_maps[t][ph].loc[:, "qij"]] = -default  # Q
                        x_lim_upper[x_maps[t][ph].loc[:, "qij"]] = default  # Q
                        # ~~ v limits ~~:
                        i_root = list(set(x_maps["a"].bi) - set(x_maps["a"].bj))[0]
                        i_v_swing = (
                            x_maps[t][ph]
                            .loc[x_maps[t][ph].loc[:, "bi"] == self.SWING, "vi"]
                            .to_numpy()[0]
                        )
                        x_lim_lower[i_v_swing] = bus.loc[self.SWING, "v_min"] ** 2
                        x_lim_upper[i_v_swing] = bus.loc[self.SWING, "v_max"] ** 2
                        x_lim_lower[x_maps[t][ph].loc[:, "vj"]] = (
                            bus.loc[x_maps[t][ph].loc[:, "bj"], "v_min"] ** 2
                        )
                        x_lim_upper[x_maps[t][ph].loc[:, "vj"]] = (
                            bus.loc[x_maps[t][ph].loc[:, "bj"], "v_max"] ** 2
                        )
                        # ~~ DER limits  ~~:
                        if self.der:
                            for i in range(self.der_bus[ph].shape[0]):
                                i_q = self.q_der_start_phase_idx[t][ph] + i
                                # reactive power bounds
                                p_out = gen["p" + ph] * pv_shape["PV"][t]
                                p_max = p_out
                                p_min = p_out * 0
                                x_lim_lower[i_q] = p_min[self.der_bus[ph][i]]
                                x_lim_upper[i_q] = p_max[self.der_bus[ph][i]]
                        # ~~ Battery limits ~~:
                        if self.battery:
                            for i in range(self.battery_bus[ph].shape[0]):
                                pb_max = bat["Pb_max_" + ph]
                                b_min = bat["bmin_" + ph]
                                b_max = bat["bmax_" + ph]
                                i_d = self.pd_bat_start_phase_idx[t][ph] + i
                                i_c = self.pc_bat_start_phase_idx[t][ph] + i
                                i_b = self.b_bat_start_phase_idx[t][ph] + i
                                # battery active power charge/discharge and s.o.c bounds
                                x_lim_lower[i_d] = 0
                                x_lim_lower[i_c] = 0
                                x_lim_lower[i_b] = b_min[self.battery_bus[ph][i]]
                                x_lim_upper[i_d] = pb_max[self.battery_bus[ph][i]]
                                x_lim_upper[i_c] = pb_max[self.battery_bus[ph][i]]
                                x_lim_upper[i_b] = b_max[self.battery_bus[ph][i]]
            bounds = [(l, u) for (l, u) in zip(x_lim_lower, x_lim_upper)]
            return bounds

        @cache
        def idx(self, var, node_j, phase, t):
            if var == "pg":
                if node_j in set(self.der_bus[phase]):
                    return (
                        self.q_der_start_phase_idx[t][phase]
                        + np.where(self.der_bus[phase] == node_j)[0]
                    )
                return []
            if var == "qg":  # reactive power generation at node
                raise ValueError("qg is fixed and is not a valid variable.")
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
                if node_j in set(self.battery_bus[phase]):
                    return (
                        self.pd_bat_start_phase_idx[t][phase]
                        + np.where(self.battery_bus[phase] == node_j)[0]
                    )
                return []
            if var == "pc":
                if node_j in set(self.battery_bus[phase]):
                    return (
                        self.pc_bat_start_phase_idx[t][phase]
                        + np.where(self.battery_bus[phase] == node_j)[0]
                    )
                return []
            if var == "b":
                if node_j in set(self.battery_bus[phase]):
                    return (
                        self.b_bat_start_phase_idx[t][phase]
                        + np.where(self.battery_bus[phase] == node_j)[0]
                    )
                return []
            return self.branch_into_j(var, node_j, phase, t)

        def create_model(self):
            r, x = self.r, self.x
            bus = self.bus
            loadshape = self.loadshape
            bat = self.bat

            # ########## Aeq and Beq Formation ###########
            n_rows = self.row_no
            n_col = self.n_x
            a_eq = zeros(
                (n_rows, n_col)
            )  # Aeq has the same number of rows as equations with a column for each x
            b_eq = zeros(n_rows)
            a_ineq = zeros((n_rows, n_col))
            b_ineq = zeros(n_rows)

            for t in range(self.n):
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
                        b_eq[q_eqn] = (1 - (bus.cvr_q[j] / 2)) * q_load - q_cap - q_gen

                        # V equation
                        i = self.idx("bi", j, a, t)[0]
                        if i == self.SWING:  # Swing bus
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
                        if self.der:
                            a_eq[q_eqn, col("qg", a)] = 1
                            b_eq[p_eqn] = (
                                (1 - (bus.cvr_p[j] / 2)) * p_load * loadshape["M"][t]
                            )
                            b_eq[q_eqn] = (
                                (1 - (bus.cvr_q[j] / 2)) * q_load - q_cap - q_gen
                            )
                        if self.battery:
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
