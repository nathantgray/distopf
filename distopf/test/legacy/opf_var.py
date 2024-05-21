import os
import shutil
import subprocess
from time import perf_counter
import numpy as np
from numpy import sqrt, real, imag, zeros
import pandas as pd
from pathlib import Path
from scipy.optimize import linprog, OptimizeResult
from scipy.sparse import lil_matrix, csr_matrix, csr_array
import networkx as nx
import cvxpy as cp
import copy

# from branchflow.converter.csv2glm import Csv2glm as Make_glm
# from branchflow.distopf.MatlabFun import dfsearch, find


def dfsearch(graph_, start_node, out_type=None):
    if len(graph_.nodes) < 1:
        return np.empty((0, 2))
    if out_type == "edgetonew":
        return np.array(list(nx.dfs_edges(graph_, source=start_node)))
    else:
        return np.array(list(nx.dfs_preorder_nodes(graph_, source=start_node)))


def find(condition, n=None, direction="first"):
    if n is not None:
        if direction == "first":
            return np.where(condition)[0][:n]
        elif direction == "last":
            return np.where(condition)[0][-n:]
        else:
            raise ValueError('Direction argument must be "first" or "last"')
    return np.where(condition)[0]


class QModel:
    def __init__(
        self,
        branchdata: pd.DataFrame,
        powerdata: pd.DataFrame,
        v_up=None,
        s_down=None,
        n_children=0,
        v_min=0.95,  # voltage minimum limit in p.u.
        v_max=1.05,  # voltage maximum limit in p.u.
        cvr=(0, 0),  #
        p_rating_mult=1,  # Real power rating as a multiple of power listed in powerdata
        s_rating_mult=1.2,  # Apparent power rating of inverters as a multiple of the real power rating (typ. 1.2)
        gen_mult=1.0,  # Multiply the inverter outputs by this. !Does not effect real power rating!
        load_mult=1.0,  # Multiply the loads by this
        # Use the following if values in branchdata and powerdata are not already in pu
        scale_z=1,  # multiply values in model to get to p.u. values
        scale_p=1,  # multiply values in model to get to p.u. values
        # The following are used exclusively for running the model in gridlabd.
        gld_dir=None,  # Provide an existing directory where a new directory with the new glm will be made and run.
        p_base_gld=1e6,  # When the GLM is created powerdata will be multiplied by this.
        v_ll_base_gld=4160,  # GLM SWING bus voltage and impedance will use this to convert to volts and ohms resp.
        # for use with targeting
        loss_percent=None,
    ):
        # ~~~~~~~~~~~~~~~~~~~~ Load Model ~~~~~~~~~~~~~~~~~~~~
        self.powerdata = powerdata.sort_values(by="id", ignore_index=True)
        self.powerdata[["PgA", "PgB", "PgC"]] *= p_rating_mult
        self.branch = branchdata.sort_values(by="tb", ignore_index=True)
        # Make v_up_sq a three-phase array of voltage magnitude squared using whatever v_up is passed in.
        if v_up is None:
            self.v_up_sq = np.array([1**2, 1**2, 1**2])
        if isinstance(v_up, (int, float, complex)):
            self.v_up_sq = np.array(
                [np.abs(v_up) ** 2, np.abs(v_up) ** 2, np.abs(v_up) ** 2]
            )
        if isinstance(v_up, (list, tuple, np.ndarray)):
            self.v_up_sq = np.abs(np.array(v_up)) ** 2
        assert len(self.v_up_sq) == 3

        self.s_dn = s_down
        if self.s_dn is None:
            self.s_dn = np.zeros((n_children, 3))
        if n_children == 0:
            self.s_dn = np.zeros((1, 3))

        self.n_children = n_children
        self.v_max = v_max**2
        self.v_min = v_min**2
        self.cvr = cvr
        self.load_mult = load_mult
        self.gen_mult = gen_mult
        self.rating_mult = s_rating_mult
        self.scale_z = scale_z
        self.scale_p = scale_p
        self.gld_dir = gld_dir
        if self.gld_dir is None:
            self.gld_dir = Path.cwd()
        self.p_base_gld = p_base_gld
        self.v_ll_base_gld = v_ll_base_gld
        self.loss_percent = loss_percent
        if loss_percent is None:
            self.loss_percent = [0, 0, 0]
        # ~~~~~~~~~~~~~~~~~~~~ prepare data ~~~~~~~~~~~~~~~~~~~~
        self.nb = len(self.powerdata.id)

        line_a = self.branch.loc[self.branch.raa != 0, ["fb", "tb"]]
        line_b = self.branch.loc[self.branch.rbb != 0, ["fb", "tb"]]
        line_c = self.branch.loc[self.branch.rcc != 0, ["fb", "tb"]]
        self.nb_a = len(line_a) + 1  # number of buses with 'A' phase
        self.nb_b = len(line_b) + 1  # number of buses with 'B' phase
        self.nb_c = len(line_c) + 1  # number of buses with 'C' phase
        self.nl_a = len(line_a)  # number of lines with 'A' phase
        self.nl_b = len(line_b)  # number of lines with 'B' phase
        self.nl_c = len(line_c)  # number of lines with 'C' phase
        self.p, self.q = self.init_power()
        self.der_bus = {
            "a": find(self.powerdata.PgA != 0),
            "b": find(self.powerdata.PgB != 0),
            "c": find(self.powerdata.PgC != 0),
        }
        self.r, self.x = self.init_rx()
        # ~~ initialize index pointers ~~
        self.edges, self.table, self.v_table, self.der_start_idx = self.variable_tables(
            self.branch
        )
        self.der_start_phase_idx = {
            "a": int(self.der_start_idx),
            "b": int(self.der_start_idx) + self.der_bus["a"].shape[0],
            "c": int(self.der_start_idx)
            + self.der_bus["a"].shape[0]
            + self.der_bus["b"].shape[0],
        }
        # self.der_bus_tot = len(self.der_bus['a']) + len(self.der_bus['b']) + len(self.der_bus['c'])
        self.n_x = int(
            self.der_start_idx
            + len(self.der_bus["a"])
            + len(self.der_bus["b"])
            + len(self.der_bus["c"])
        )
        self.bounds = self.init_bounds()

        # ~~~~~~~~~~~~~~~~~~~~ initialize Aeq and beq and objective gradient ~~~~~~~~~~~~~~~~~~~~
        self.a_eq, self.b_eq = self.create_model()
        self.x0 = None
        # ~~~~~~~~~~~~~~~~~~~~ Approximate loss from previous runs ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.p_loss = 0
        self.q_loss = 0
        self.p_loss_prev = 0
        self.q_loss_prev = 0
        self.solver_runs = 0

    def init_rx(self):
        row = np.array(np.r_[self.branch.fb, self.branch.tb], dtype=int) - 1
        col = np.array(np.r_[self.branch.tb, self.branch.fb], dtype=int) - 1
        r = {
            "aa": csr_matrix(
                (np.r_[self.branch.raa, self.branch.raa] * self.scale_z, (row, col))
            ),
            "ab": csr_matrix(
                (np.r_[self.branch.rab, self.branch.rab] * self.scale_z, (row, col))
            ),
            "ac": csr_matrix(
                (np.r_[self.branch.rac, self.branch.rac] * self.scale_z, (row, col))
            ),
            "bb": csr_matrix(
                (np.r_[self.branch.rbb, self.branch.rbb] * self.scale_z, (row, col))
            ),
            "bc": csr_matrix(
                (np.r_[self.branch.rbc, self.branch.rbc] * self.scale_z, (row, col))
            ),
            "cc": csr_matrix(
                (np.r_[self.branch.rcc, self.branch.rcc] * self.scale_z, (row, col))
            ),
        }
        x = {
            "aa": csr_matrix(
                (np.r_[self.branch.xaa, self.branch.xaa] * self.scale_z, (row, col))
            ),
            "ab": csr_matrix(
                (np.r_[self.branch.xab, self.branch.xab] * self.scale_z, (row, col))
            ),
            "ac": csr_matrix(
                (np.r_[self.branch.xac, self.branch.xac] * self.scale_z, (row, col))
            ),
            "bb": csr_matrix(
                (np.r_[self.branch.xbb, self.branch.xbb] * self.scale_z, (row, col))
            ),
            "bc": csr_matrix(
                (np.r_[self.branch.xbc, self.branch.xbc] * self.scale_z, (row, col))
            ),
            "cc": csr_matrix(
                (np.r_[self.branch.xcc, self.branch.xcc] * self.scale_z, (row, col))
            ),
        }
        return r, x

    def init_power(self, s_dn=None, load_mult=None, gen_mult=None):
        if s_dn is None:
            s_dn = self.s_dn
        if load_mult is None:
            load_mult = self.load_mult
        if gen_mult is None:
            gen_mult = self.gen_mult
        p = {
            "l": {
                "a": self.powerdata.Pa * load_mult * self.scale_p,  # Pload of phase A
                "b": self.powerdata.Pb * load_mult * self.scale_p,  # Pload of phase B
                "c": self.powerdata.Pc * load_mult * self.scale_p,  # Pload of phase C
            },
            "g": {
                "a": self.powerdata.PgA * gen_mult * self.scale_p,  # Pgen of phase A
                "b": self.powerdata.PgB * gen_mult * self.scale_p,  # Pgen of phase B
                "c": self.powerdata.PgC * gen_mult * self.scale_p,  # Pgen of phase C
            },
            "rated": {
                "a": self.powerdata.PgA * self.scale_p,  # Pgen of phase A
                "b": self.powerdata.PgB * self.scale_p,  # Pgen of phase B
                "c": self.powerdata.PgC * self.scale_p,  # Pgen of phase C
            },
        }
        q = {
            "l": {
                "a": self.powerdata.Qa * load_mult * self.scale_p,  # Qload of phase A
                "b": self.powerdata.Qb * load_mult * self.scale_p,  # Qload of phase B
                "c": self.powerdata.Qc * load_mult * self.scale_p,  # Qload of phase C
            },
            "c": {
                "a": self.powerdata.CapA * self.scale_p,
                "b": self.powerdata.CapB * self.scale_p,
                "c": self.powerdata.CapC * self.scale_p,
            },
        }

        if self.n_children > 0:
            for j in range(s_dn.shape[0]):
                p["l"]["a"].iloc[-1 - j] = real(s_dn[-1 - j, 0])  # in PU
                q["l"]["a"].iloc[-1 - j] = imag(s_dn[-1 - j, 0])
                p["l"]["b"].iloc[-1 - j] = real(s_dn[-1 - j, 1])  # in PU
                q["l"]["b"].iloc[-1 - j] = imag(s_dn[-1 - j, 1])
                p["l"]["c"].iloc[-1 - j] = real(s_dn[-1 - j, 2])  # in PU
                q["l"]["c"].iloc[-1 - j] = imag(s_dn[-1 - j, 2])

        return p, q

    def init_bounds(self):
        n_x = self.n_x
        table = self.table
        s_mult = self.rating_mult
        v_min = self.v_min
        v_max = self.v_max
        p, q = self.p, self.q
        # ~~~~~~~~~~ x limits ~~~~~~~~~~
        x_lim_lower = np.ones(n_x) * -100e3
        x_lim_upper = np.ones(n_x) * 100e3
        for ph in "abc":
            if table[ph].shape[0] > 0:
                x_lim_lower[table[ph][0, 2] : table[ph][-1, 2] + 1] = -100e3  # P
                x_lim_upper[table[ph][0, 2] : table[ph][-1, 2] + 1] = 100e3  # P

                x_lim_lower[table[ph][0, 3] : table[ph][-1, 3] + 1] = -100e3  # Q
                x_lim_upper[table[ph][0, 3] : table[ph][-1, 3] + 1] = 100e3  # Q

                x_lim_lower[table[ph][0, 4] - 1 : table[ph][-1, 4] + 1] = v_min  # V
                x_lim_upper[table[ph][0, 4] - 1 : table[ph][-1, 4] + 1] = v_max
                # ~~ DER limits  ~~:
                for i in range(self.der_bus[ph].shape[0]):
                    s_dg_rated = s_mult * p["rated"][ph]
                    p_dg_output = p["g"][ph]
                    min_q = -sqrt(
                        (s_dg_rated**2) - (p_dg_output**2)
                    )  # reactive power bounds
                    max_q = sqrt((s_dg_rated**2) - (p_dg_output**2))
                    x_lim_lower[self.der_start_phase_idx[ph] + i] = min_q[
                        self.der_bus[ph][i]
                    ]
                    x_lim_upper[self.der_start_phase_idx[ph] + i] = max_q[
                        self.der_bus[ph][i]
                    ]
        bounds = [(l, u) for (l, u) in zip(x_lim_lower, x_lim_upper)]
        return bounds

    @staticmethod
    def variable_tables(branch):
        a_indices = (branch.raa != 0) | (branch.xaa != 0)
        b_indices = (branch.rbb != 0) | (branch.xbb != 0)
        c_indices = (branch.rcc != 0) | (branch.xcc != 0)
        line_a = branch.loc[a_indices, ["fb", "tb"]].values
        line_b = branch.loc[b_indices, ["fb", "tb"]].values
        line_c = branch.loc[c_indices, ["fb", "tb"]].values
        nl_a = len(line_a)
        nl_b = len(line_b)
        nl_c = len(line_c)
        g = nx.Graph()
        g_a = nx.Graph()
        g_b = nx.Graph()
        g_c = nx.Graph()
        g.add_edges_from(branch[["fb", "tb"]].values.astype(int) - 1)
        g_a.add_edges_from(line_a.astype(int) - 1)
        g_b.add_edges_from(line_b.astype(int) - 1)
        g_c.add_edges_from(line_c.astype(int) - 1)

        edges = dfsearch(g, 0, "edgetonew")
        t_a = dfsearch(g_a, 0, "edgetonew")
        t_b = dfsearch(g_b, 0, "edgetonew")
        t_c = dfsearch(g_c, 0, "edgetonew")

        p_a_end = 1 * nl_a
        q_a_end = 2 * nl_a
        v_a_end = 3 * nl_a + 1
        p_b_end = v_a_end + 1 * nl_b
        q_b_end = v_a_end + 2 * nl_b
        v_b_end = v_a_end + 3 * nl_b + 1
        p_c_end = v_b_end + 1 * nl_c
        q_c_end = v_b_end + 2 * nl_c
        v_c_end = v_b_end + 3 * nl_c + 1
        p_a = np.array([i for i in range(p_a_end)])  # defining the unknowns for phaseA:
        q_a = np.array([i for i in range(p_a_end, q_a_end)])
        v_a = np.array([i for i in range(q_a_end, v_a_end)])

        p_b = np.array(
            [i for i in range(v_a_end, p_b_end)]
        )  # defining the unknowns for phaseB:
        q_b = np.array([i for i in range(p_b_end, q_b_end)])
        v_b = np.array([i for i in range(q_b_end, v_b_end)])

        p_c = np.array(
            [i for i in range(v_b_end, p_c_end)]
        )  # defining the unknowns for phaseC:
        q_c = np.array([i for i in range(p_c_end, q_c_end)])
        v_c = np.array([i for i in range(q_c_end, v_c_end)])

        table_a = np.c_[t_a[:, 0], t_a[:, 1], p_a, q_a, v_a[1:]]
        table_b = np.c_[t_b[:, 0], t_b[:, 1], p_b, q_b, v_b[1:]]
        table_c = np.c_[t_c[:, 0], t_c[:, 1], p_c, q_c, v_c[1:]]
        table = {"a": table_a, "b": table_b, "c": table_c}
        v_table = {
            "a": np.array([v_a]).T,
            "b": np.array([v_b]).T,
            "c": np.array([v_c]).T,
        }
        der_start_idx = v_c_end  # start with the largest index so far

        return edges, table, v_table, der_start_idx

    def create_model(self):
        v_up_a, v_up_b, v_up_c = self.v_up_sq[0], self.v_up_sq[1], self.v_up_sq[2]
        nb = self.nb
        nl_a = self.nl_a
        nl_b = self.nl_b
        nl_c = self.nl_c
        edges = self.edges
        table = self.table
        v_table = self.v_table
        r, x = self.r, self.x
        pl_nom = self.p["l"]
        ql_nom = self.q["l"]
        pg = self.p["g"]
        qc = self.q["c"]

        # ########## Aeq and Beq Formation ###########
        n_rows = table["c"][-1, 4] + 1
        n_col = self.n_x
        a_eq = zeros(
            (n_rows, n_col)
        )  # Aeq has the same number of rows as equations with a column for each x
        b_eq = zeros(n_rows)

        for j in range(1, nb):
            # Connected area as a fixed load, not a voltage dependent load
            if j >= (nb - self.n_children):
                cvr_p = 0
                cvr_q = 0
            else:
                cvr_p = self.cvr[0]
                cvr_q = self.cvr[1]

            # parent = find(j == edges[:, 1])
            parent_a = find(j == table["a"][:, 1])
            parent_b = find(j == table["b"][:, 1])
            parent_c = find(j == table["c"][:, 1])
            phase_exists = {
                "a": len(parent_a) > 0,
                "b": len(parent_b) > 0,
                "c": len(parent_c) > 0,
            }
            children = {
                "a": find(j == table["a"][:, 0]),
                "b": find(j == table["b"][:, 0]),
                "c": find(j == table["c"][:, 0]),
            }
            i = int(edges[find(j == edges[:, 1]), 0][0])  # Upstream bus
            # Define row indexes which correspond to p, q, and v equations for each bus.
            p_a_row = parent_a
            q_a_row = parent_a + 1 * nl_a
            v_a_row = parent_a + 2 * nl_a
            p_b_row = parent_b + 3 * nl_a + 1
            q_b_row = parent_b + 3 * nl_a + 1 + 1 * nl_b
            v_b_row = parent_b + 3 * nl_a + 1 + 2 * nl_b
            p_c_row = parent_c + 3 * nl_a + 1 + 3 * nl_b + 1
            q_c_row = parent_c + 3 * nl_a + 1 + 3 * nl_b + 1 + 1 * nl_c
            v_c_row = parent_c + 3 * nl_a + 1 + 3 * nl_b + 1 + 2 * nl_c
            p_row = {"a": p_a_row, "b": p_b_row, "c": p_c_row}
            q_row = {"a": q_a_row, "b": q_b_row, "c": q_c_row}
            v_row = {"a": v_a_row, "b": v_b_row, "c": v_c_row}
            # Indexes of variables associated with the branch where j is the to-bus
            # index of p variable in branch data where j is the to-bus
            p_idx = {
                "a": table["a"][parent_a, 2],
                "b": table["b"][parent_b, 2],
                "c": table["c"][parent_c, 2],
            }
            # index of q variable
            q_idx = {
                "a": table["a"][parent_a, 3],
                "b": table["b"][parent_b, 3],
                "c": table["c"][parent_c, 3],
            }
            # index of v variable (v=V^2)
            vj_idx = {
                "a": table["a"][parent_a, 4],
                "b": table["b"][parent_b, 4],
                "c": table["c"][parent_c, 4],
            }
            for ph in ["abc", "bca", "cab"]:
                a, b, c = ph[0], ph[1], ph[2]
                aa = "".join(sorted(a + a))
                ab = "".join(
                    sorted(a + b)
                )  # if ph=='cab', then a+b=='ca'. Sort so ab=='ac'
                ac = "".join(sorted(a + c))
                if phase_exists[a]:
                    vi_idx = v_table[a][find(i == table[a][:, 0])[0]]
                    # P equation
                    a_eq[p_row[a], p_idx[a]] = 1
                    a_eq[p_row[a], vj_idx[a]] = -(cvr_p / 2) * pl_nom[a][j]
                    b_eq[p_row[a]] = (1 - (cvr_p / 2)) * pl_nom[a][j] - pg[a][j]
                    # Q equation
                    a_eq[q_row[a], q_idx[a]] = 1
                    a_eq[q_row[a], vj_idx[a]] = -(cvr_q / 2) * ql_nom[a][j]
                    for child in children[a]:
                        a_eq[p_row[a], table[a][child, 2]] = -1
                        a_eq[q_row[a], table[a][child, 3]] = -1
                    if j in self.der_bus[a]:
                        der_idx = self.der_start_phase_idx[a] + find(
                            self.der_bus[a] == j
                        )
                        a_eq[q_row[a], der_idx] = 1
                    b_eq[q_row[a]] = (1 - (cvr_q / 2)) * ql_nom[a][j] - qc[a][j]
                    # V equation
                    a_eq[v_row[a], vj_idx[a]] = 1
                    a_eq[v_row[a], vi_idx] = -1
                    a_eq[v_row[a], p_idx[a]] = 2 * r[aa][i, j]
                    a_eq[v_row[a], q_idx[a]] = 2 * x[aa][i, j]
                    a_eq[v_row[a], p_idx[b]] = -r[ab][i, j] + sqrt(3) * x[ab][i, j]
                    a_eq[v_row[a], q_idx[b]] = -x[ab][i, j] - sqrt(3) * r[ab][i, j]
                    a_eq[v_row[a], p_idx[c]] = -r[ac][i, j] - sqrt(3) * x[ac][i, j]
                    a_eq[v_row[a], q_idx[c]] = -x[ac][i, j] + sqrt(3) * r[ac][i, j]

        # Force upstream voltage to be as defined.
        a_eq[3 * nl_a, v_table["a"][0]] = 1
        a_eq[3 * nl_a + 1 + 3 * nl_b, v_table["b"][0]] = 1
        a_eq[3 * nl_a + 1 + 3 * nl_b + 1 + 3 * nl_c, v_table["c"][0]] = 1
        b_eq[3 * nl_a] = v_up_a
        b_eq[3 * nl_a + 1 + 3 * nl_b] = v_up_b
        b_eq[3 * nl_a + 1 + 3 * nl_b + 1 + 3 * nl_c] = v_up_c

        return a_eq, b_eq

    def update_model(
        self, v_up=None, s_dn=None, load_mult=None, gen_mult=None, reinit_x=True
    ):
        if v_up is not None:
            self.v_up_sq = (np.array(v_up) * np.array(v_up).conjugate()).real
        if s_dn is not None:
            self.s_dn = s_dn
        if load_mult is not None:
            self.load_mult = load_mult
        if gen_mult is not None:
            self.gen_mult = gen_mult
        if reinit_x:
            self.x0 = None
        # Update parameters
        self.p, self.q = self.init_power(
            s_dn=s_dn, load_mult=load_mult, gen_mult=gen_mult
        )
        self.bounds = self.init_bounds()
        self.bounds_original = copy.copy(self.bounds)

        # Update Aeq and beq
        nl_a = self.nl_a
        nl_b = self.nl_b
        nl_c = self.nl_c

        a_eq = self.a_eq
        b_eq = self.b_eq

        for j in range(1, self.nb):
            # Connected area as a fixed load, not a voltage dependent load
            if j >= (self.nb - self.n_children):
                cvr_p = 0
                cvr_q = 0
            else:
                cvr_p = self.cvr[0]
                cvr_q = self.cvr[1]

            parent_a = find(j == self.table["a"][:, 1])
            parent_b = find(j == self.table["b"][:, 1])
            parent_c = find(j == self.table["c"][:, 1])
            phase_exists = {
                "a": len(parent_a) > 0,
                "b": len(parent_b) > 0,
                "c": len(parent_c) > 0,
            }
            # Define row indexes which correspond to p, q, and v equations for each bus.
            p_a_row = parent_a
            q_a_row = parent_a + 1 * nl_a
            p_b_row = parent_b + 3 * nl_a + 1
            q_b_row = parent_b + 3 * nl_a + 1 + 1 * nl_b
            p_c_row = parent_c + 3 * nl_a + 1 + 3 * nl_b + 1
            q_c_row = parent_c + 3 * nl_a + 1 + 3 * nl_b + 1 + 1 * nl_c
            p_row = {"a": p_a_row, "b": p_b_row, "c": p_c_row}
            q_row = {"a": q_a_row, "b": q_b_row, "c": q_c_row}
            vj_idx = {
                "a": self.table["a"][parent_a, 4],
                "b": self.table["b"][parent_b, 4],
                "c": self.table["c"][parent_c, 4],
            }
            for ph in "abc":
                if phase_exists[ph]:
                    # P equation
                    a_eq[p_row[ph], vj_idx[ph]] = -(cvr_p / 2) * self.p["l"][ph][j]
                    b_eq[p_row[ph]] = (1 - (cvr_p / 2)) * self.p["l"][ph][j] - self.p[
                        "g"
                    ][ph][j]
                    # Q equation
                    a_eq[q_row[ph], vj_idx[ph]] = -(cvr_q / 2) * self.q["l"][ph][j]
                    b_eq[q_row[ph]] = (1 - (cvr_q / 2)) * self.q["l"][ph][j] - self.q[
                        "c"
                    ][ph][j]

        # Force upstream voltage to be as defined.
        b_eq[3 * nl_a] = self.v_up_sq[0]
        b_eq[3 * nl_a + 1 + 3 * nl_b] = self.v_up_sq[1]
        b_eq[3 * nl_a + 1 + 3 * nl_b + 1 + 3 * nl_c] = self.v_up_sq[2]

        self.a_eq = a_eq
        self.b_eq = b_eq

        return a_eq, b_eq

    def parse_output(self, x_sol):
        nb = self.nb
        table = self.table
        v_up_sq = self.v_up_sq
        der_bus_a = self.der_bus["a"]
        der_bus_b = self.der_bus["b"]
        der_bus_c = self.der_bus["c"]
        # ~~ Solution of Voltage and S:
        VA = zeros(nb)
        VB = zeros(nb)
        VC = zeros(nb)
        S_allA = zeros(nb - 1) * 1j
        S_allB = zeros(nb - 1) * 1j
        S_allC = zeros(nb - 1) * 1j
        for j in range(table["a"].shape[0]):
            VA[table["a"][j, 1]] = x_sol[table["a"][j, 4]]
            S_allA[table["a"][j, 1] - 1] = complex(
                x_sol[table["a"][j, 2]], x_sol[table["a"][j, 3]]
            )
        VA[0] = v_up_sq[0]

        for j in range(table["b"].shape[0]):
            VB[table["b"][j, 1]] = x_sol[table["b"][j, 4]]
            S_allB[table["b"][j, 1] - 1] = complex(
                x_sol[table["b"][j, 2]], x_sol[table["b"][j, 3]]
            )
        VB[0] = v_up_sq[1]

        for j in range(table["c"].shape[0]):
            VC[table["c"][j, 1]] = x_sol[table["c"][j, 4]]
            S_allC[table["c"][j, 1] - 1] = complex(
                x_sol[table["c"][j, 2]], x_sol[table["c"][j, 3]]
            )
        VC[0] = v_up_sq[2]

        # Decision Variable result:
        ng_a = len(der_bus_a)
        ng_b = len(der_bus_b)
        ng_c = len(der_bus_c)
        dec_var_q_only_a = x_sol[self.der_start_idx : self.der_start_idx + ng_a]
        dec_var_q_only_b = x_sol[
            self.der_start_idx + ng_a : self.der_start_idx + ng_a + ng_b
        ]
        dec_var_q_only_c = x_sol[
            self.der_start_idx + ng_a + ng_b : self.der_start_idx + ng_a + ng_b + ng_c
        ]

        dec_var = np.zeros((nb, 3))
        for j in range(ng_a):
            dec_var[der_bus_a[j], 0] = dec_var_q_only_a[j]
        for j in range(ng_b):
            dec_var[der_bus_b[j], 1] = dec_var_q_only_b[j]
        for j in range(ng_c):
            dec_var[der_bus_c[j], 2] = dec_var_q_only_c[j]

        all_v_sq = np.r_[[VA], [VB], [VC]]
        all_s = np.r_[[S_allA], [S_allB], [S_allC]]
        return all_v_sq, all_s, dec_var, None

    def get_dec_variables(self, x_sol):
        ng_a = len(self.der_bus["a"])
        ng_b = len(self.der_bus["b"])
        ng_c = len(self.der_bus["c"])
        dec_var_q_only_a = x_sol[self.der_start_idx : self.der_start_idx + ng_a]
        dec_var_q_only_b = x_sol[
            self.der_start_idx + ng_a : self.der_start_idx + ng_a + ng_b
        ]
        dec_var_q_only_c = x_sol[
            self.der_start_idx + ng_a + ng_b : self.der_start_idx + ng_a + ng_b + ng_c
        ]
        dec_var = np.zeros((self.nb, 3))
        for j in range(ng_a):
            dec_var[self.der_bus["a"][j], 0] = dec_var_q_only_a[j]
        for j in range(ng_b):
            dec_var[self.der_bus["b"][j], 1] = dec_var_q_only_b[j]
        for j in range(ng_c):
            dec_var[self.der_bus["c"][j], 2] = dec_var_q_only_c[j]
        return dec_var

    def get_v_raw(self, x_sol):
        table = self.table
        # ~~ Solution of Voltage:
        v_df = pd.DataFrame()
        for ph in "abc":
            v_df.loc[1, ph] = x_sol[self.v_table[ph][0, 0]]
            for j in range(table[ph].shape[0]):
                bus_id = table[ph][j, 1] + 1
                v_df.loc[bus_id, ph] = x_sol[table[ph][j, 4]]
        return v_df.sort_index()

    def get_v_solved(self, x_sol):
        table = self.table
        # ~~ Solution of Voltage:
        v_df = pd.DataFrame()
        for ph in "abc":
            v_df.loc[1, ph] = sqrt(x_sol[self.v_table[ph][0, 0]])
            for j in range(table[ph].shape[0]):
                bus_id = table[ph][j, 1] + 1
                v_df.loc[bus_id, ph] = sqrt(x_sol[table[ph][j, 4]])
        return v_df.sort_index()

    def get_s_solved(self, x_sol):
        s_df = pd.DataFrame(
            columns=["fb", "tb", "a", "b", "c"], index=range(2, self.nb + 1)
        )
        s_df["a"] = s_df["a"].astype(complex)
        s_df["b"] = s_df["b"].astype(complex)
        s_df["c"] = s_df["c"].astype(complex)
        for ph in "abc":
            s_df.loc[self.table[ph][:, 1] + 1, "fb"] = self.table[ph][:, 0] + 1
            s_df.loc[self.table[ph][:, 1] + 1, "tb"] = self.table[ph][:, 1] + 1
            s_df.loc[self.table[ph][:, 1] + 1, ph] = (
                x_sol[self.table[ph][:, 2]] + 1j * x_sol[self.table[ph][:, 3]]
            )
        return s_df

    # For debugging. Check aeq against constraints equations in constraints()
    def check_aeq(self, x0):
        # ~~~~~~~~~~ check jacobian ~~~~~~~~~~
        g0 = self.constraints(x0)
        for row in range(self.a_eq.shape[0]):
            print(f"row: {row}")
            for col in range(len(x0)):
                if np.abs(self.a_eq[row, col]) > 1e-6:
                    xh = x0.copy()
                    xh[col] = x0[col] + 1e-10
                    est_dif = (self.constraints(xh) - g0) / 1e-10
                    est_dif = est_dif[row]
                    err = self.a_eq[row, col] - est_dif
                    if np.abs(err) > 2e-6:
                        print(
                            f"error at ({row}, {col}) = {err} \t estimate: {est_dif} \t Aeq: {self.a_eq[row, col]}"
                        )

    # only used to double-check a_eq in check_aeq()
    def constraints(self, xk):
        nb = self.nb
        nl_a = self.nl_a
        nl_b = self.nl_b
        nl_c = self.nl_c
        table = self.table
        v_table = self.v_table
        edges = self.edges
        der_a_buses = self.der_bus["a"]
        der_b_buses = self.der_bus["b"]
        der_c_buses = self.der_bus["c"]
        r, x = self.r, self.x
        v_up_sq = self.v_up_sq
        pl_a_nom, pl_b_nom, pl_c_nom = (
            self.p["l"]["a"],
            self.p["l"]["b"],
            self.p["l"]["c"],
        )
        ql_a_nom, ql_b_nom, ql_c_nom = (
            self.q["l"]["a"],
            self.q["l"]["b"],
            self.q["l"]["c"],
        )
        qc_a, qc_b, qc_c = self.q["c"]["a"], self.q["c"]["b"], self.q["c"]["c"]
        pg_a, pg_b, pg_c = self.p["g"]["a"], self.p["g"]["b"], self.p["g"]["c"]
        cvr_p = self.cvr[0]
        cvr_q = self.cvr[1]
        g = np.zeros(3 * nl_a + 1 + 3 * nl_b + 1 + 3 * nl_c + 1)

        # initialize variables:
        paa = lil_matrix((nb, nb))
        qaa = lil_matrix((nb, nb))
        pbb = lil_matrix((nb, nb))
        qbb = lil_matrix((nb, nb))
        pcc = lil_matrix((nb, nb))
        qcc = lil_matrix((nb, nb))
        va = np.zeros(nb, dtype=float)
        vb = np.zeros(nb, dtype=float)
        vc = np.zeros(nb, dtype=float)
        va[0] = xk[int(v_table["a"][0])]
        vb[0] = xk[int(v_table["b"][0])]
        vc[0] = xk[int(v_table["c"][0])]
        for j in range(1, self.nb):
            # parent = find(j == edges[:, 1])
            parent_a = find(j == table["a"][:, 1])
            parent_b = find(j == table["b"][:, 1])
            parent_c = find(j == table["c"][:, 1])
            children_a = find(j == table["a"][:, 0])
            children_b = find(j == table["b"][:, 0])
            children_c = find(j == table["c"][:, 0])
            # current bus is j
            i = int(edges[find(j == edges[:, 1]), 0])  # Upstream bus
            k_buses_a = table["a"][children_a, 1]  # Downstream buses
            k_buses_b = table["b"][children_b, 1]
            k_buses_c = table["c"][children_c, 1]
            r_aa, r_ab, r_ac = r["aa"][i, j], r["ab"][i, j], r["ac"][i, j]
            r_bb, r_bc, r_cc = r["bb"][i, j], r["bc"][i, j], r["cc"][i, j]
            x_aa, x_ab, x_ac = x["aa"][i, j], x["ab"][i, j], x["ac"][i, j]
            x_bb, x_bc, x_cc = x["bb"][i, j], x["bc"][i, j], x["cc"][i, j]
            r_ba, r_ca, r_cb = r_ab, r_ac, r_bc
            x_ba, x_ca, x_cb = x_ab, x_ac, x_bc

            # ~~~~~~~~ DEFINE VARIABLES in X vector ~~~~~~~~
            a_exists = len(parent_a) > 0
            b_exists = len(parent_b) > 0
            c_exists = len(parent_c) > 0
            if a_exists:
                paa[i, j] = xk[int(table["a"][parent_a, 2])]
                qaa[i, j] = xk[int(table["a"][parent_a, 3])]
                Poc = find(i == table["a"][:, 0])
                va[i] = xk[int(v_table["a"][Poc[0]])]
                va[j] = xk[int(table["a"][parent_a, 4])]  # v variable (v=V^2)
                if len(k_buses_a) > 0:
                    for child in children_a:
                        k = table["a"][child, 1]
                        paa[j, k] = xk[int(table["a"][child, 2])]
                        qaa[j, k] = xk[int(table["a"][child, 3])]
            if b_exists:
                pbb[i, j] = xk[int(table["b"][parent_b, 2])]
                qbb[i, j] = xk[int(table["b"][parent_b, 3])]
                Poc = find(i == table["b"][:, 0])
                vb[i] = xk[int(v_table["b"][Poc[0]])]
                vb[j] = xk[int(table["b"][parent_b, 4])]
                if len(k_buses_b) > 0:
                    for child in children_b:
                        k = table["b"][child, 1]
                        pbb[j, k] = xk[int(table["b"][child, 2])]
                        qbb[j, k] = xk[int(table["b"][child, 3])]
            if c_exists:
                pcc[i, j] = xk[int(table["c"][parent_c, 2])]
                qcc[i, j] = xk[int(table["c"][parent_c, 3])]
                Poc = find(i == table["c"][:, 0])
                vc[i] = xk[int(v_table["c"][Poc[0]])]
                vc[j] = xk[int(table["c"][parent_c, 4])]
                if len(k_buses_c) > 0:
                    for child in children_c:
                        k = table["c"][child, 1]
                        pcc[j, k] = xk[int(table["c"][child, 2])]
                        qcc[j, k] = xk[int(table["c"][child, 3])]
            qg_a = 0
            qg_b = 0
            qg_c = 0
            if j in der_a_buses:
                qg_a_idx = int(self.der_start_phase_idx["a"] + find(j == der_a_buses))
                qg_a = xk[qg_a_idx]
            if j in der_b_buses:
                qg_b_idx = int(self.der_start_phase_idx["b"] + find(j == der_b_buses))
                qg_b = xk[qg_b_idx]
            if j in der_b_buses:
                qg_c_idx = int(self.der_start_phase_idx["c"] + find(j == der_c_buses))
                qg_c = xk[qg_c_idx]
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

            # ~~~~~~~~ Initial Calculations ~~~~~~~~
            # Derived from eqns 5 and 6
            pab = qab = pba = qba = pac = qac = pca = qca = pbc = qbc = pcb = qcb = 0
            if (
                a_exists and b_exists
            ):  # can be linearly approximated if sqrt(va[i]/vb[i]) removed
                pab = (-pbb[i, j] - sqrt(3) * qbb[i, j]) / 2
                qab = (-qbb[i, j] + sqrt(3) * pbb[i, j]) / 2
                pba = (-paa[i, j] + sqrt(3) * qaa[i, j]) / 2
                qba = (-qaa[i, j] - sqrt(3) * paa[i, j]) / 2
            if a_exists and c_exists:
                pac = (-pcc[i, j] + sqrt(3) * qcc[i, j]) / 2
                qac = (-qcc[i, j] - sqrt(3) * pcc[i, j]) / 2
                pca = (-paa[i, j] - sqrt(3) * qaa[i, j]) / 2
                qca = (-qaa[i, j] + sqrt(3) * paa[i, j]) / 2
            if b_exists and c_exists:
                pbc = (-pcc[i, j] - sqrt(3) * qcc[i, j]) / 2
                qbc = (-qcc[i, j] + sqrt(3) * pcc[i, j]) / 2
                pcb = (-pbb[i, j] + sqrt(3) * qbb[i, j]) / 2
                qcb = (-qbb[i, j] - sqrt(3) * pbb[i, j]) / 2
            # ~~~~~~~~ Primary Constraint Equations ~~~~~~~~
            p_a_row = parent_a
            q_a_row = parent_a + 1 * nl_a
            v_a_row = parent_a + 2 * nl_a
            p_b_row = parent_b + 3 * nl_a + 1
            q_b_row = parent_b + 3 * nl_a + 1 + 1 * nl_b
            v_b_row = parent_b + 3 * nl_a + 1 + 2 * nl_b
            p_c_row = parent_c + 3 * nl_a + 1 + 3 * nl_b + 1
            q_c_row = parent_c + 3 * nl_a + 1 + 3 * nl_b + 1 + 1 * nl_c
            v_c_row = parent_c + 3 * nl_a + 1 + 3 * nl_b + 1 + 2 * nl_c
            # ~~~~~~~~ Phase A equations
            if a_exists:
                # eqn 30 Voltage dependence of real load power
                pl_a = pl_a_nom[j] * (1 + cvr_p / 2 * (va[j] - 1))
                # eqn 31 Voltage dependence of reactive load power
                ql_a = ql_a_nom[j] * (1 + cvr_q / 2 * (va[j] - 1))
                # eqn 27
                g[p_a_row] = (
                    paa[i, j] - sum([(paa[j, k]) for k in k_buses_a]) - pl_a + pg_a[j]
                )
                # eqn 28
                g[q_a_row] = (
                    qaa[i, j]
                    - sum([(qaa[j, k]) for k in k_buses_a])
                    - ql_a
                    + qg_a
                    + qc_a[j]
                )
                # eqn 29
                g[v_a_row] = (
                    -va[i]
                    + va[j]
                    + (
                        2 * (paa[i, j] * r_aa + qaa[i, j] * x_aa)
                        + 2 * (pab * r_ab + qab * x_ab)
                        + 2 * (pac * r_ac + qac * x_ac)
                    )
                )

                # fixed source node voltage
                g[3 * nl_a] = va[0] - v_up_sq[0]
            # ~~~~~~~~ Phase B equations
            if b_exists:
                # eqn 30 Voltage dependence of real load power
                pl_b = pl_b_nom[j] * (1 + cvr_p / 2 * (vb[j] - 1))
                # eqn 31 Voltage dependence of reactive load power
                ql_b = ql_b_nom[j] * (1 + cvr_q / 2 * (vb[j] - 1))
                # eqn 27
                g[p_b_row] = (
                    pbb[i, j] - sum([(pbb[j, k]) for k in k_buses_b]) - pl_b + pg_b[j]
                )
                # eqn 28
                g[q_b_row] = (
                    qbb[i, j]
                    - sum([(qbb[j, k]) for k in k_buses_b])
                    - ql_b
                    + qg_b
                    + qc_b[j]
                )
                # eqn 29
                g[v_b_row] = (
                    -vb[i]
                    + vb[j]
                    + (
                        2 * (pba * r_ba + qba * x_ba)
                        + 2 * (pbb[i, j] * r_bb + qbb[i, j] * x_bb)
                        + 2 * (pbc * r_bc + qbc * x_bc)
                    )
                )
                # fixed source node voltage
                g[3 * nl_a + 1 + 3 * nl_b] = vb[0] - v_up_sq[1]
            # ~~~~~~~~ Phase C equations
            if c_exists:
                # eqn 30 Voltage dependence of real load power
                pl_c = pl_c_nom[j] * (1 + cvr_p / 2 * (vc[j] - 1))
                # eqn 31 Voltage dependence of reactive load power
                ql_c = ql_c_nom[j] * (1 + cvr_q / 2 * (vc[j] - 1))
                # eqn 27
                g[p_c_row] = (
                    pcc[i, j] - sum([(pcc[j, k]) for k in k_buses_c]) - pl_c + pg_c[j]
                )
                # eqn 28
                g[q_c_row] = (
                    qcc[i, j]
                    - sum([(qcc[j, k]) for k in k_buses_c])
                    - ql_c
                    + qg_c
                    + qc_c[j]
                )
                # eqn 29
                g[v_c_row] = (
                    -vc[i]
                    + vc[j]
                    + (
                        2 * (pca * r_ca + qca * x_ca)
                        + 2 * (pcb * r_cb + qcb * x_cb)
                        + 2 * (pcc[i, j] * r_cc + qcc[i, j] * x_cc)
                    )
                )
                # fixed source node voltage
                g[3 * nl_a + 1 + 3 * nl_b + 1 + 3 * nl_c] = vc[0] - v_up_sq[2]
        return g

    # ~~~ linear objective functions for use with solve_lin() ~~~
    def gradient_load_min(self):
        c = np.zeros(self.n_x)
        for ph in "abc":
            if self.table[ph].shape[0] > 0:
                c[self.table[ph][0, 2]] = 1
        return c

    def gradient_curtail(self):
        c = np.zeros(self.n_x)
        for i in range(self.der_start_idx, self.n_x):
            c[i] = -1
        return c

    # ~~~ Quadratic objective with linear constraints for use with solve_quad()~~~
    def cp_obj_loss(self, xk):
        nb = self.nb
        table = self.table
        r, x = self.r, self.x
        edges = self.edges
        f: cp.Expression = 0
        for j in range(1, nb):
            parent_a = find(
                j == table["a"][:, 1]
            )  # returns the row index for where j is the to-bus
            parent_b = find(j == table["b"][:, 1])
            parent_c = find(j == table["c"][:, 1])
            parent = find(j == edges[:, 1])
            # current bus is j
            i = int(edges[parent, 0][0])  # Upstream bus is i
            # ~~~ Phase A ~~~
            if len(parent_a) > 0:
                pa_idx = table["a"][parent_a, 2]
                qa_idx = table["a"][parent_a, 3]
                f += r["aa"][i, j] * (xk[pa_idx] ** 2)
                f += r["aa"][i, j] * (xk[qa_idx] ** 2)

            # ~~~ Phase B ~~~
            if len(parent_b) > 0:
                pb_idx = table["b"][parent_b, 2]
                qb_idx = table["b"][parent_b, 3]
                f += r["bb"][i, j] * (xk[pb_idx] ** 2)
                f += r["bb"][i, j] * (xk[qb_idx] ** 2)

            # ~~~ Phase C ~~~
            if len(parent_c) > 0:
                pc_idx = table["c"][parent_c, 2]
                qc_idx = table["c"][parent_c, 3]
                f += r["cc"][i, j] * (xk[pc_idx] ** 2)
                f += r["cc"][i, j] * (xk[qc_idx] ** 2)
        return f

    def cp_obj_loss_q(self, xk):
        nb = self.nb
        table = self.table
        r, x = self.r, self.x
        edges = self.edges
        f: cp.Expression = 0
        for j in range(1, nb):
            parent_a = find(
                j == table["a"][:, 1]
            )  # returns the row index for where j is the to-bus
            parent_b = find(j == table["b"][:, 1])
            parent_c = find(j == table["c"][:, 1])
            parent = find(j == edges[:, 1])
            # current bus is j
            i = int(edges[parent, 0])  # Upstream bus is i
            # ~~~ Phase A ~~~
            if len(parent_a) > 0:
                pa_idx = table["a"][parent_a, 2]
                qa_idx = table["a"][parent_a, 3]
                f += r["aa"][i, j] * (xk[qa_idx] ** 2)

            # ~~~ Phase B ~~~
            if len(parent_b) > 0:
                pb_idx = table["b"][parent_b, 2]
                qb_idx = table["b"][parent_b, 3]
                f += r["bb"][i, j] * (xk[qb_idx] ** 2)

            # ~~~ Phase C ~~~
            if len(parent_c) > 0:
                pc_idx = table["c"][parent_c, 2]
                qc_idx = table["c"][parent_c, 3]
                f += r["cc"][i, j] * (xk[qc_idx] ** 2)
        return f

    def cp_obj_target_p(self, xk, target_p):
        # f: cp.Expression = cp.Expression()
        f = cp.Constant(0)
        for i, ph in enumerate("abc"):
            if self.table[ph].shape[0] > 0:
                f += (
                    target_p[i]
                    - xk[self.table[ph][0, 2]] * (1 + self.loss_percent[i] / 100)
                ) ** 2
        return f

    def cp_obj_target_p_total(self, xk, target):
        actual = 0
        for i, ph in enumerate("abc"):
            if self.table[ph].shape[0] > 0:
                actual += xk[self.table[ph][0, 2]]
        f = (target - actual * (1 + self.loss_percent / 100)) ** 2
        return f

    def cp_obj_target_q(self, xk, target_q):
        f = cp.Constant(0)
        for i, ph in enumerate("abc"):
            if self.table[ph].shape[0] > 0:
                f += (
                    target_q[i]
                    - xk[self.table[ph][0, 3]] * (1 + self.loss_percent[i] / 100)
                ) ** 2
        return f

    def cp_obj_target_q_total(self, xk, target):
        actual = 0
        for i, ph in enumerate("abc"):
            if self.table[ph].shape[0] > 0:
                actual += xk[self.table[ph][0, 3]]
        f = (target - actual * (1 + self.loss_percent[0] / 100)) ** 2
        return f

    def cp_obj_curtail(self, xk):
        f = cp.Constant(0)
        for i in range(self.der_start_idx, self.n_x):
            f += (self.bounds[i][1] - xk[i]) ** 2
        return f

    @staticmethod
    def cp_obj_none(xk):
        return cp.Constant(0)

    # ~~~ solvers ~~~
    def solve_lin(self, c, method=None):
        tic = perf_counter()
        res = linprog(
            c, A_eq=csr_array(self.a_eq), b_eq=self.b_eq.flatten(), bounds=self.bounds
        )
        if not res.success:  # if first method fails, fallback to revised simplex
            print(res.message)
            print(f"solve_lin Failed.")
            # u_residual = self.find_upper_voltage_violation_lin(c)
            assert res.success
        runtime = perf_counter() - tic
        res["runtime"] = runtime
        return res

    def solve_quad(self, obj_func, target=None):
        if self.x0 is None:
            lin_res = self.solve_lin(np.zeros(self.n_x))
            self.x0 = lin_res.x.copy()
        x = cp.Variable(shape=(self.n_x,), name="x", value=self.x0)
        tic = perf_counter()
        g = [self.a_eq @ x - self.b_eq.flatten() == 0]
        lb = [x[i] >= self.bounds[i][0] for i in range(self.n_x)]
        ub = [x[i] <= self.bounds[i][1] for i in range(self.n_x)]
        # obj: cp.Expression = m.objective_quad(x)
        if target is not None:
            prob = cp.Problem(cp.Minimize(obj_func(x, target)), g + ub + lb)
        else:
            prob = cp.Problem(cp.Minimize(obj_func(x)), g + ub + lb)
        # try:
        prob.solve(verbose=False, solver=cp.CLARABEL)
        # except cp.SolverError:
        # prob.solve(verbose=False, solver=cp.CLARABEL)

        x_res = x.value
        result = OptimizeResult(
            fun=prob.value,  # should function value be updated with gld_correction?
            success=(prob.status == "optimal"),
            message=prob.status,
            x=x_res,
            nit=prob.solver_stats.num_iters,
            runtime=perf_counter() - tic,
        )
        return result

    # def solve_target_linear_loss(self, target=None):
    #     if self.x0 is None:
    #         lin_res = self.solve_lin(np.zeros(self.n_x))
    #         self.x0 = lin_res.x.copy()
    #     x = cp.Variable(shape=(self.n_x,), name="x", value=self.x0)
    #     tic = perf_counter()
    #     g = [self.a_eq @ x - self.b_eq.flatten() == 0]
    #     lb = [x[i] >= self.bounds[i][0] for i in range(self.n_x)]
    #     ub = [x[i] <= self.bounds[i][1] for i in range(self.n_x)]
    #     # obj: cp.Expression = m.objective_quad(x)
    #
    #     actual = 0
    #     for i, ph in enumerate("abc"):
    #         if self.table[ph].shape[0] > 0:
    #             actual += x[self.table[ph][0, 3]]
    #     f = (target - actual - self.q_loss_prev) ** 2
    #
    #     prob = cp.Problem(cp.Minimize(f), g + ub + lb)
    #     prob.solve(verbose=False, solver=cp.OSQP)
    #     x_res = x.value
    #
    #     x_res_gld = self.gld_solve(
    #         x_res, p_base=self.p_base_gld, v_ll_base=self.v_ll_base_gld
    #     )
    #     # res.fun = c @ x_res
    #     all_v_sq, all_s, dec_var, _ = self.parse_output(x_res)
    #     all_v_sq_gld, all_s_gld, dec_var_gld, _ = self.parse_output(x_res_gld)
    #     self.p_loss_prev = self.p_loss
    #     self.q_loss_prev = self.q_loss
    #     self.p_loss = sum(all_s_gld[:, 0].real) - sum(all_s[:, 0].real)
    #     self.q_loss = sum(all_s_gld[:, 0].imag) - sum(all_s[:, 0].imag)
    #
    #     x = cp.Variable(shape=(self.n_x,), name="x", value=x_res)
    #     g = [self.a_eq @ x - self.b_eq.flatten() == 0]
    #     lb = [x[i] >= self.bounds[i][0] for i in range(self.n_x)]
    #     ub = [x[i] <= self.bounds[i][1] for i in range(self.n_x)]
    #     # obj: cp.Expression = m.objective_quad(x)
    #
    #     actual = 0
    #     for i, ph in enumerate("abc"):
    #         if self.table[ph].shape[0] > 0:
    #             actual += x[self.table[ph][0, 3]]
    #     f = (target - actual - self.q_loss_prev) ** 2
    #
    #     prob = cp.Problem(cp.Minimize(f), g + ub + lb)
    #     prob.solve(verbose=False, solver=cp.OSQP)
    #     x_res = x.value
    #     res = OptimizeResult(
    #         fun=prob.value,  # should function value be updated with gld_correction?
    #         success=(prob.status == "optimal"),
    #         message=prob.status,
    #         x=x_res,
    #         nit=prob.solver_stats.num_iters,
    #         runtime=perf_counter() - tic,
    #     )
    #     return res

    # USE THIS TO SOLVE
    def solve(self, objective, target=None, gld_correction=False, loss_percent=None):
        self.solver_runs += 1
        if loss_percent is not None:
            self.loss_percent = loss_percent

        if objective == "load":
            res = self.solve_lin(self.gradient_load_min())
        elif objective == "curtail":
            res = self.solve_lin(self.gradient_curtail())
        elif objective == "loss":
            res = self.solve_quad(self.cp_obj_loss)
        elif objective == "loss_q_only":
            res = self.solve_quad(self.cp_obj_loss_q)
        elif objective == "pf":
            res = self.solve_lin(np.zeros(self.n_x))
        elif objective == "p_target":
            if target is not None:
                if isinstance(target, (int, float)):
                    res = self.solve_quad(self.cp_obj_target_p_total, target=target)
                elif len(target) == 1:
                    res = self.solve_quad(self.cp_obj_target_p_total, target=target[0])
                else:
                    res = self.solve_quad(self.cp_obj_target_p, target=target)
            else:
                raise AttributeError(f"p_target requires a target to be supplied.")
        elif objective == "q_target":
            if target is not None:
                if isinstance(target, (int, float)):
                    res = self.solve_quad(self.cp_obj_target_q_total, target=target)
                elif len(target) == 1:
                    res = self.solve_quad(self.cp_obj_target_q_total, target=target[0])
                else:
                    res = self.solve_quad(self.cp_obj_target_q, target=target)
            else:
                raise AttributeError(f"q_target requires a target to be supplied.")
        # elif objective == "q_target_linear_loss":
        #     if target is not None:
        #         res = self.solve_target_linear_loss(target=target)
        #     else:
        #         raise AttributeError(f"q_target requires a target to be supplied.")

        else:
            raise NotImplementedError(f"Objective: {objective}, not supported.")

        # if gld_correction:
        #     x_prev = res.x
        #     x_res = self.gld_solve(
        #         res.x, p_base=self.p_base_gld, v_ll_base=self.v_ll_base_gld
        #     )
        #     # res.fun = c @ x_res
        #     res.x = x_res
        #
        #     all_v_sq, all_s, dec_var, _ = self.parse_output(x_prev)
        #     all_v_sq_gld, all_s_gld, dec_var_gld, _ = self.parse_output(res.x)
        #     self.p_loss_prev = self.p_loss
        #     self.q_loss_prev = self.q_loss
        #     self.p_loss = sum(all_s_gld[:, 0].real) - sum(all_s[:, 0].real)
        #     self.q_loss = sum(all_s_gld[:, 0].imag) - sum(all_s[:, 0].imag)
        #     print("Max correction:")
        #     print(f"v: {np.max(np.abs(np.sqrt(all_v_sq_gld) - np.sqrt(all_v_sq)))}")
        #     print(f"P: {np.max(np.abs(all_s_gld.real - all_s.real))}")
        #     print(f"Q: {np.max(np.abs(all_s_gld.imag - all_s.imag))}")
        #     print(f"Gen: {np.max(np.abs(dec_var_gld - dec_var))}")
        #     res.p_loss = self.p_loss
        #     res.q_loss = self.q_loss
        return res

    # def gld_solve(self, xk, p_base=None, v_ll_base=None, dir_path=None):
    #     all_v_sq, all_s, q_gen, all_l = self.parse_output(xk)
    #     v_up = np.sqrt(self.v_up_sq)
    #     s_dn = (
    #         self.s_dn
    #     )  # np.array([all_s[:, i] for i in range(all_s.shape[1] - self.n_children, all_s.shape[1])])
    #     if p_base is None:
    #         p_base = self.p_base_gld
    #     if v_ll_base is None:
    #         v_ll_base = self.v_ll_base_gld
    #     if dir_path is None:
    #         dir_path = self.gld_dir
    #
    #     model_dir = Path(dir_path)
    #     v_ln_base = v_ll_base / np.sqrt(3)
    #     curr_base = p_base / v_ln_base
    #     down_nodes = []
    #     if self.n_children > 0:
    #         for j in range(s_dn.shape[0]):
    #             down_nodes.append(int(self.powerdata.id.values[-1 - j]))
    #     down_nodes.reverse()
    #     tmp_dir = model_dir / "gld_tmp"
    #     if tmp_dir in list(model_dir.glob("*")):
    #         shutil.rmtree(tmp_dir)
    #     tmp_dir.mkdir()
    #     (tmp_dir / "output").mkdir()
    #     output_name = tmp_dir / "system.glm"
    #     # s_dn = np.array([0.01+0.001j, 0.009+0.001j, 0.011+0.001j])
    #     Make_glm(
    #         output_name,
    #         self.branch,
    #         self.powerdata,
    #         q_gen=q_gen,  # * p_base,
    #         down_nodes=down_nodes,
    #         model_results_out_dir="output",
    #         cvr=self.cvr,
    #         single_run=True,
    #         v_ln_base=v_ln_base,
    #         v_ss_pu=np.abs(v_up),
    #         s_dn_pu=s_dn,
    #         s_base=p_base,
    #         gen_mult=self.gen_mult,
    #         load_mult=self.load_mult,
    #         rating_mult=self.rating_mult,
    #         tz="PST+8PDT",
    #         starttime="'2001-08-01 12:00:00'",
    #         stoptime="'2001-08-01 12:00:01'",
    #     )
    #
    #     print("Running GridLAB-D model...")
    #     subprocess.run(["gridlabd", output_name.name], env=os.environ, cwd=tmp_dir)
    #     print("Processing GridLAB-D results...")
    #     fn = tmp_dir / "output" / "output_voltage.csv"
    #     nodes = pd.Series(["node_"] * self.nb) + self.powerdata.id.astype(int).astype(
    #         str
    #     )
    #     volt_dump = pd.read_csv(fn, sep=",", header=1, index_col=0)
    #     # volt_dump.loc[nodes, :]
    #     volt_mag = np.zeros((self.nb, 3))
    #     if "voltA_mag" in volt_dump.keys():  # polar format
    #         volt_mag = (
    #             volt_dump.loc[nodes, :]
    #             .loc[nodes, ["voltA_mag", "voltB_mag", "voltC_mag"]]
    #             .values
    #             / v_ln_base
    #         )
    #     if "voltA_real" in volt_dump.loc[nodes, :].keys():  # rectangular format
    #         volt_mag = (
    #             np.sqrt(
    #                 volt_dump.loc[nodes, :]
    #                 .loc[nodes, ["voltA_real", "voltB_real", "voltC_real"]]
    #                 .values
    #                 ** 2
    #                 + volt_dump.loc[nodes, :]
    #                 .loc[nodes, ["voltA_imag", "voltB_imag", "voltC_imag"]]
    #                 .values
    #                 ** 2
    #             )
    #             / v_ln_base
    #         )
    #     assert np.max(volt_mag[0, :] - np.abs(v_up) * np.ones(3)) < 1e-8
    #
    #     fn = tmp_dir / "output" / "output_current.csv"
    #     curr_dump = pd.read_csv(fn, sep=",", header=1, index_col=0) / curr_base
    #     curr_dump["fb"] = curr_dump.index.str.split("_").str.get(-2).astype(int) - 1
    #     curr_dump["tb"] = curr_dump.index.str.split("_").str.get(-1).astype(int) - 1
    #     curr_dump = curr_dump.sort_values(by="tb")
    #     if "currA_mag" in curr_dump.keys():  # polar format
    #         curr_dump["la"] = (np.abs(curr_dump["currA_mag"])) ** 2
    #         curr_dump["lb"] = (np.abs(curr_dump["currB_mag"])) ** 2
    #         curr_dump["lc"] = (np.abs(curr_dump["currC_mag"])) ** 2
    #     if "currA_real" in curr_dump.keys():  # rectangular format
    #         curr_dump["la"] = (
    #             np.abs(curr_dump["currA_real"] + 1j * curr_dump["currA_imag"])
    #         ) ** 2
    #         curr_dump["lb"] = (
    #             np.abs(curr_dump["currB_real"] + 1j * curr_dump["currB_imag"])
    #         ) ** 2
    #         curr_dump["lc"] = (
    #             np.abs(curr_dump["currC_real"] + 1j * curr_dump["currC_imag"])
    #         ) ** 2
    #     x_ = np.zeros_like(xk)
    #     # slack bus voltages
    #     for i, ph in enumerate("abc"):
    #         if self.table[ph].shape[0] > 0:  # if phase exists
    #             x_[self.table[ph][0, 4] - 1] = volt_mag[0, i] ** 2
    #     # p, q, l, v for each phase
    #     for ph_idx, ph in enumerate("abc"):
    #         for line_idx in range(self.table[ph].shape[0]):
    #             fb = self.table[ph][line_idx, 0]
    #             tb = self.table[ph][line_idx, 1]
    #             p_idx = self.table[ph][line_idx, 2]
    #             q_idx = self.table[ph][line_idx, 3]
    #             v_idx = self.table[ph][line_idx, 4]
    #             cur = None
    #             if "currA_mag" in curr_dump.keys():  # polar format
    #                 cur = curr_dump[f"curr{ph.upper()}_mag"][
    #                     curr_dump["tb"] == tb
    #                 ] * np.exp(
    #                     1j * curr_dump[f"curr{ph.upper()}_angle"][curr_dump["tb"] == tb]
    #                 )
    #             if "currA_real" in curr_dump.keys():  # rectangular format
    #                 cur = (
    #                     curr_dump[f"curr{ph.upper()}_real"][curr_dump["tb"] == tb]
    #                     + curr_dump[f"curr{ph.upper()}_imag"][curr_dump["tb"] == tb]
    #                     * 1j
    #                 )
    #             volt = volt_mag[fb, ph_idx] * np.exp(
    #                 volt_dump.loc[f"node_{fb+1}", f"volt{ph.upper()}_angle"] * 1j
    #             )
    #             s = volt * cur[0].conjugate()
    #             x_[p_idx] = s.real
    #             x_[q_idx] = s.imag
    #             x_[v_idx] = volt_mag[tb, ph_idx] ** 2
    #     # q_gen
    #     x_[self.der_start_idx :] = xk[self.der_start_idx :]
    #     return x_

    def find_inequality_residuals(self, x, bounds):
        lb = [bounds[i][0] for i in range(len(bounds))]
        ub = [bounds[i][1] for i in range(len(bounds))]
        l_residual = x - lb
        u_residual = ub - x
        return l_residual, u_residual

    def adjust_v_bounds(self, x):
        child_voltage_idxs = []
        v_down = []
        v_idxs = np.r_[
            self.table["a"][:, 4], self.table["b"][:, 4], self.table["c"][:, 4]
        ]
        for child_id in range(self.nb - self.n_children, self.nb):
            child_voltage_idxs.append(
                self.table["a"][np.where(self.table["a"][:, 1] == child_id)[0], 4][0]
            )
            child_voltage_idxs.append(
                self.table["b"][np.where(self.table["b"][:, 1] == child_id)[0], 4][0]
            )
            child_voltage_idxs.append(
                self.table["c"][np.where(self.table["c"][:, 1] == child_id)[0], 4][0]
            )
            v_down.append(
                [
                    x[
                        self.table["a"][
                            np.where(self.table["a"][:, 1] == child_id)[0], 4
                        ][0]
                    ],
                    x[
                        self.table["b"][
                            np.where(self.table["b"][:, 1] == child_id)[0], 4
                        ][0]
                    ],
                    x[
                        self.table["c"][
                            np.where(self.table["c"][:, 1] == child_id)[0], 4
                        ][0]
                    ],
                ]
            )
        v_down = np.array(v_down) ** 0.5
        print(v_down)
        adjusted = False
        for i in v_idxs:
            u_violation = self.bounds_original[i][1] - x[i]
            if u_violation < 0:
                print("violation of " + str(-u_violation) + " on index: " + str(i))
                self.bounds[i] = (
                    self.bounds_original[i][0],
                    self.bounds[i][1] + u_violation,
                )
                adjusted = True
        return adjusted

    def find_upper_voltage_violation_lin(self, c):
        lb = np.array([self.bounds[i][0] for i in range(len(self.bounds))])
        ub = np.array([self.bounds[i][1] for i in range(len(self.bounds))])
        ub_modified = np.array([self.bounds[i][1] for i in range(self.n_x)])
        u_residual = None
        print("relaxing bounds:")
        for i in range(10):
            v_idxs = np.r_[
                self.table["a"][:, 4], self.table["b"][:, 4], self.table["c"][:, 4]
            ]
            print(".", end="")
            ub_modified[v_idxs] = ub_modified[v_idxs] + 1e-9
            bounds = [(l, u) for (l, u) in zip(lb, ub_modified)]
            res = linprog(
                c, A_eq=csr_array(self.a_eq), b_eq=self.b_eq.flatten(), bounds=bounds
            )
            if res.success:
                l_residual, u_residual = self.find_inequality_residuals(
                    res.x, self.bounds
                )
                idx_violation = np.where(u_residual < 0)
                relaxed_limits = ub_modified[idx_violation]
                ub_modified[v_idxs] = ub[v_idxs]
                ub_modified[idx_violation] = relaxed_limits
                res = linprog(
                    c,
                    A_eq=csr_array(self.a_eq),
                    b_eq=self.b_eq.flatten(),
                    bounds=bounds,
                )
                break
        return u_residual


if __name__ == "__main__":
    pass
