import distopf as opf
import numpy as np
import pandas as pd
from numpy import sqrt, zeros


class LinDistPQ(opf.LinDistModelModular):
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

    """

    def __init__(
        self,
        branch_data: pd.DataFrame = None,
        bus_data: pd.DataFrame = None,
        gen_data: pd.DataFrame = None,
        cap_data: pd.DataFrame = None,
        reg_data: pd.DataFrame = None,
    ):

        super().__init__(
            branch_data, bus_data, gen_data, cap_data=cap_data, reg_data=reg_data
        )
        # ~~~~~~~~~~~~~~~~~~~~ initialize Aeq and beq ~~~~~~~~~~~~~~~~~~~~
        self._a_ub, self._b_ub = None, None

    def init_bounds(self):
        default = 100e3  # Default for unbounded variables.
        # ~~~~~~~~~~ x limits ~~~~~~~~~~
        x_lim_lower = np.ones(self.n_x) * -default
        x_lim_upper = np.ones(self.n_x) * default
        x_lim_lower, x_lim_upper = self.add_voltage_limits(x_lim_lower, x_lim_upper)
        x_lim_lower, x_lim_upper = self.user_added_limits(x_lim_lower, x_lim_upper)
        bounds = np.c_[x_lim_lower, x_lim_upper]
        # bounds = [(l, u) for (l, u) in zip(x_lim_lower, x_lim_upper)]
        return bounds

    def create_hexagon_constraints(self):

        """
        Create inequality constraints for the optimization problem.
        """

        # ########## Aineq and Bineq Formation ###########
        n_rows_ineq = 6 * (
                len(self.gen_buses["a"])
                + len(self.gen_buses["b"])
                + len(self.gen_buses["c"])
        )
        a_ineq = zeros((n_rows_ineq, self.n_x))
        b_ineq = zeros(n_rows_ineq)
        ineq1 = 0
        ineq2 = 1
        ineq3 = 2
        ineq4 = 3
        ineq5 = 4
        ineq6 = 5

        for j in self.gen.index:
            for a in "abc":
                if not self.phase_exists(a, j):
                    continue
                pg = self.idx("pg", j, a)
                qg = self.idx("qg", j, a)
                s_rated = self.gen.at[j, f"s{a}_max"]
                # equation indexes
                a_ineq[ineq1, pg] = -sqrt(3)
                a_ineq[ineq1, qg] = -1
                b_ineq[ineq1] = sqrt(3)*s_rated
                a_ineq[ineq2, pg] = sqrt(3)
                a_ineq[ineq2, qg] = 1
                b_ineq[ineq2] = sqrt(3)*s_rated
                a_ineq[ineq3, qg] = -1
                b_ineq[ineq3] = sqrt(3)/2*s_rated
                a_ineq[ineq4, qg] = 1
                b_ineq[ineq4] = sqrt(3)/2*s_rated
                a_ineq[ineq5, pg] = sqrt(3)
                a_ineq[ineq5, qg] = -1
                b_ineq[ineq5] = sqrt(3)*s_rated
                a_ineq[ineq6, pg] = -sqrt(3)
                a_ineq[ineq6, qg] = 1
                b_ineq[ineq6] = -sqrt(3)*s_rated
                ineq1 += 6
                ineq2 += 6
                ineq3 += 6
                ineq4 += 6
                ineq5 += 6
                ineq6 += 6

        return a_ineq, b_ineq

    def create_octagon_constraints(self):

        """
        Create inequality constraints for the optimization problem.
        """

        # ########## Aineq and Bineq Formation ###########
        n_inequalities = 5
        n_rows_ineq = n_inequalities * (
                len(self.gen_buses["a"])
                + len(self.gen_buses["b"])
                + len(self.gen_buses["c"])
        )
        a_ineq = zeros((n_rows_ineq, self.n_x))
        b_ineq = zeros(n_rows_ineq)
        ineq1 = 0
        ineq2 = 1
        ineq3 = 2
        ineq4 = 3
        ineq5 = 4

        for j in self.gen.index:
            for a in "abc":
                if not self.phase_exists(a, j):
                    continue
                pg = self.idx("pg", j, a)
                qg = self.idx("qg", j, a)
                s_rated = self.gen.at[j, f"s{a}_max"]
                # equation indexes
                a_ineq[ineq1, pg] = sqrt(2)
                a_ineq[ineq1, qg] = -2+sqrt(2)
                b_ineq[ineq1] = sqrt(2)*s_rated
                a_ineq[ineq2, pg] = sqrt(2)
                a_ineq[ineq2, qg] = 2-sqrt(2)
                b_ineq[ineq2] = sqrt(2)*s_rated
                a_ineq[ineq3, pg] = -1+sqrt(2)
                a_ineq[ineq3, qg] = 1
                b_ineq[ineq3] = s_rated
                a_ineq[ineq4, pg] = -1+sqrt(2)
                a_ineq[ineq4, qg] = -1
                b_ineq[ineq4] = s_rated
                a_ineq[ineq5, pg] = -1
                b_ineq[ineq5] = 0
                ineq1 += n_inequalities
                ineq2 += n_inequalities
                ineq3 += n_inequalities
                ineq4 += n_inequalities
                ineq5 += n_inequalities

        return a_ineq, b_ineq

    @property
    def a_ub(self):
        if self._a_ub is None:
            self._a_ub, self._b_ub = self.create_octagon_constraints()
        return self._a_ub

    @property
    def b_ub(self):
        if self._b_ub is None:
            self._a_ub, self._b_ub = self.create_octagon_constraints()
        return self._b_ub

