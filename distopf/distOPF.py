"""
This module contains high-level helper functions for creating and running provided models, solvers, and objectives.
"""

from collections.abc import Callable
from pathlib import Path
import json
import pandas as pd
import numpy as np

from distopf import DSSParser, CASES_DIR, LinDistModel, LinDistModelP, LinDistModelQ
from distopf.opf_solver import (
    cp_obj_loss,
    cp_obj_curtail,
    cp_obj_target_p_3ph,
    cp_obj_target_p_total,
    cp_obj_target_q_3ph,
    cp_obj_target_q_total,
    cvxpy_solve,
    gradient_load_min,
    gradient_curtail,
    lp_solve,
)
from distopf.plot import plot_network, plot_voltages, plot_power_flows, plot_ders
from distopf.lindist_base import (
    _handle_branch_input,
    _handle_bus_input,
    _handle_gen_input,
    _handle_cap_input,
    _handle_reg_input,
)


def create_model(control_variable: str = "", **kwargs) -> LinDistModel:
    """
    Create the correct LinDistModel object based on the control variable.
    Parameters
    ----------
    control_variable : str, optional : No Control Variables-None, Active Power Control-'p', Reactive Power Control-'q'
    kwargs :
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
    Returns
    -------
    model: LinDistModel, or LinDistModelP, or LinDistModelQ object appropriate for the control variable
    """
    if control_variable is None or control_variable == "":
        return LinDistModel(**kwargs)
    if control_variable.lower() == "p":
        return LinDistModelP(**kwargs)
    if control_variable.lower() == "q":
        return LinDistModelQ(**kwargs)

    raise ValueError(
        f"Unknown control variable '{control_variable}'. Valid options are 'P', 'Q' or None"
    )


def auto_solve(model, objective_function=None, **kwargs):
    """
    Solve with selected objective function and model. Automatically chooses the appropriate

    Parameters
    ----------
    model : LinDistModel
    objective_function : str or Callable
    kwargs : kwargs to pass to objective function and solver function.
        solver: str
            Solver to use for solving with CVXPY. Default is CLARABEL. OSQP is also recommended.
        target:
            Used with target objectives. Target to track.
            Scalar for target_p_total and target_q_total and size-3 array for target_p_3ph, and target_q_3ph.
        error_percent:
            Used with target objectives. Percent error expected in total system load compared exact solution.

    Returns
    -------
    result: scipy.optimize.OptimizeResult

    """
    solver = kwargs.get("solver", "CLARABEL")
    if objective_function is None:
        return lp_solve(model, np.zeros(model.n_x))
    if isinstance(objective_function, Callable):
        return cvxpy_solve(model, objective_function, **kwargs)
    if isinstance(objective_function, (np.ndarray, list)):
        return lp_solve(model, objective_function)
    if not isinstance(objective_function, str):
        raise TypeError("objective_function must be a function handle or a string")
    if objective_function.lower() == "gen_max":
        return lp_solve(model, gradient_curtail(model))
    if objective_function.lower() == "load_min":
        return lp_solve(model, gradient_load_min(model))
    if objective_function.lower() == "loss_min":
        return cvxpy_solve(model, cp_obj_loss, solver=solver)
    if objective_function.lower() == "curtail_min":
        return cvxpy_solve(model, cp_obj_curtail, solver=solver)
    if objective_function.lower() == "target_p_3ph":
        return cvxpy_solve(model, cp_obj_target_p_3ph, **kwargs)
    if objective_function.lower() == "target_q_3ph":
        return cvxpy_solve(model, cp_obj_target_q_3ph, **kwargs)
    if objective_function.lower() == "target_p_total":
        return cvxpy_solve(model, cp_obj_target_p_total, **kwargs)
    if objective_function.lower() == "target_q_total":
        return cvxpy_solve(model, cp_obj_target_q_total, **kwargs)


def _handle_path_input(data_path: Path) -> Path:
    cwd = Path.cwd()
    if data_path.is_absolute():
        return data_path
    if (cwd / data_path).exists():
        return cwd / data_path
    if (CASES_DIR / "csv" / data_path).exists():
        return CASES_DIR / "csv" / data_path
    if (CASES_DIR / "dss" / data_path).exists():
        return CASES_DIR / "dss" / data_path
    return data_path


def _get_data_from_path(data_path: Path) -> dict:
    branch_data = None
    bus_data = None
    gen_data = None
    cap_data = None
    reg_data = None
    if not data_path.exists():
        raise FileNotFoundError()
    if data_path.is_file() and data_path.suffix.lower() != ".dss":
        raise ValueError(
            "The variable, data_path, must point to a directory containing model CSVs or an OpenDSS model file."
        )
    if data_path.is_dir():
        branch_path = data_path / "branch_data.csv"
        if branch_path.exists():
            branch_data = pd.read_csv(branch_path, header=0)
        bus_path = data_path / "bus_data.csv"
        if bus_path.exists():
            bus_data = pd.read_csv(bus_path, header=0)
        gen_path = data_path / "gen_data.csv"
        if gen_path.exists():
            gen_data = pd.read_csv(gen_path, header=0)
        cap_path = data_path / "cap_data.csv"
        if cap_path.exists():
            cap_data = pd.read_csv(cap_path, header=0)
        reg_path = data_path / "reg_data.csv"
        if reg_path.exists():
            reg_data = pd.read_csv(data_path / "reg_data.csv", header=0)
    if data_path.suffix.lower() == ".dss":
        dss_parser = DSSParser(data_path)
        branch_data = dss_parser.branch_data
        bus_data = dss_parser.bus_data
        gen_data = dss_parser.gen_data
        cap_data = dss_parser.cap_data
        reg_data = dss_parser.reg_data

    branch_data = _handle_branch_input(branch_data)
    bus_data = _handle_bus_input(bus_data)
    gen_data = _handle_gen_input(gen_data)
    cap_data = _handle_cap_input(cap_data)
    reg_data = _handle_reg_input(reg_data)
    return {
        "branch_data": branch_data,
        "bus_data": bus_data,
        "gen_data": gen_data,
        "cap_data": cap_data,
        "reg_data": reg_data,
    }


class DistOPFCase(object):
    """
    Use this class to create a distOPF case, run it, and save and plot results.
    Parameters
    ----------
    config: str or dict
        Path to JSON config or dictionary with parameters to create case. Alternative to using **config.
    data_path: str or pathlib.Path
        Path to the directory containing the data CSVs or path to OpenDSS model. Will also accept names of
        cases include in package e.g. "ieee13", "ieee34", "ieee123".
    output_dir: str or pathlib.Path
        (default: "output") Directory to save results.
    branch_data : pd.DataFrame or None
        DataFrame containing branch data (r and x values, limits). Overrides data found from data_path.
    bus_data : pd.DataFrame or None
        DataFrame containing bus data (loads, voltages, limits). Overrides data found from data_path.
    gen_data : pd.DataFrame or None
        DataFrame containing generator/DER data. Overrides data found from data_path.
    cap_data : pd.DataFrame or None
        DataFrame containing capacitor data. Overrides data found from data_path.
    reg_data : pd.DataFrame or None
        DataFrame containing regulator data. Overrides data found from data_path.
    v_swing: Number or size-3 array
        Override substation voltage. Scalar or 3-phase array. Per Unit.
    v_min: Number
        Override all voltage minimum limits. Per Unit.
    v_max: Number
        Override all voltage maximum limits. Per Unit.
    gen_mult: Number
        Scale all generator outputs and ratings. Per Unit.
    load_mult:
        Scale all loads.
    control_variable: str
        Control variable for optimization. Options (case-insensitive):
            None: Power flow only with no optimization. `objective_function` options will be ignored.
            "P": Active power injections from generators. Active power outputs set in gen_data.csv will be ignored
                 and reactive power outputs set in gen_data static.
            "Q": Reactive power injections from generators.
                 Active power outputs set in gen_data.csv are constant and reactive power outputs set in
                 gen_data.csv will be ignored.
    objective_function: str or Callable
        Objective function for optimization. Options (case-insensitive):
            "gen_max": Maximize output of generators. Uses scipy.optimize.linprog.
            "load_min": Minimize total substation active power load. Uses scipy.optimize.linprog.
            "loss_min": Minimize total line active power losses. Quadratic. Uses CVXPY.
            "curtail_min": Minimize DER/Generator curtailment. Quadratic. Uses CVXPY.
            "target_p_3ph": Substation load tracks active power target on each phase. Quadratic. Uses CVXPY.
            "target_q_3ph": Substation load tracks reactive power target on each phase. Quadratic. Uses CVXPY.
            "target_p_total": Substation load tracks total active power. Quadratic. Uses CVXPY.
            "target_q_total": Substation load tracks total reactive power. Quadratic. Uses CVXPY.
    show_plots: bool
        (default False) If true, renders plots in browser
    save_results: bool
        (default False) If true, saves result data to CSVs in output_dir
    save_plots: bool
        (default False) If true, saves interactive plots as html to output folder
    save_inputs: bool
        (default False) If true, saves model CSV and other input parameters.
        NOTE CSVs include any modifications made by other parameters such as gen_mult, load_mult, v_max, v_min, or
        v_swing.
    """

    def __init__(self, **kwargs):
        config = kwargs.get("config")
        if config is not None:
            if len(kwargs) != 1:
                raise ValueError(
                    "If config is provided, other parameters are not allowed."
                )
            if isinstance(config, (str, Path)):
                config = _handle_path_input(Path(config))
                if not config.suffix.lower() == ".json":
                    raise ValueError("config file must be a JSON formatted file.")
                with open(config) as f:
                    config = json.load(f)
            if not isinstance(config, dict):
                raise ValueError(
                    "config must be a dictionary or a path to a JSON formatted file."
                )
            kwargs = config

        self.data_path = kwargs.get("data_path")
        self.v_swing = kwargs.get("v_swing")
        self.v_max = kwargs.get("v_max")
        self.v_min = kwargs.get("v_min")
        self.gen_mult = kwargs.get("gen_mult")
        self.load_mult = kwargs.get("load_mult")

        self.control_variable = kwargs.get("control_variable", "")
        self.objective_function = kwargs.get("objective_function")
        self.target = kwargs.get("target")
        self.error_percent = kwargs.get("error_percent")
        self.solver = kwargs.get("solver", "CLARABEL")

        self.output_dir = Path(kwargs.get("output_dir", "output"))
        self.save_inputs = kwargs.get("save_inputs", False)
        self.save_results = kwargs.get("save_results", False)
        self.save_plots = kwargs.get("save_plots", False)
        self.show_plots = kwargs.get("show_plots", False)

        # Import case
        self.branch_data = None
        self.bus_data = None
        self.gen_data = None
        self.cap_data = None
        self.reg_data = None
        if self.data_path is not None:
            self.data_path = _handle_path_input(Path(self.data_path))
            case_data = _get_data_from_path(self.data_path)
            self.branch_data = case_data["branch_data"]
            self.bus_data = case_data["bus_data"]
            self.gen_data = case_data["gen_data"]
            self.cap_data = case_data["cap_data"]
            self.reg_data = case_data["reg_data"]
        if kwargs.get("branch_data") is not None:
            self.branch_data = _handle_branch_input(kwargs.get("branch_data"))
        if kwargs.get("bus_data") is not None:
            self.bus_data = _handle_bus_input(kwargs.get("bus_data"))
        if kwargs.get("gen_data") is not None:
            self.gen_data = _handle_gen_input(kwargs.get("gen_data"))
        if kwargs.get("cap_data") is not None:
            self.cap_data = _handle_cap_input(kwargs.get("cap_data"))
        if kwargs.get("reg_data") is not None:
            self.reg_data = _handle_reg_input(kwargs.get("reg_data"))
        if self.branch_data is None or self.bus_data is None:
            raise ValueError(
                "At least one of branch_data or bus_data was not found. "
                "Either provide them as CSV files found at the location "
                "specified in data_path or pass them in directly."
            )

        # Modify case
        if self.gen_mult is not None and self.gen_data is not None:
            self.gen_data.loc[:, ["pa", "pb", "pc"]] *= self.gen_mult
            self.gen_data.loc[:, ["qa", "qb", "qc"]] *= self.gen_mult
            self.gen_data.loc[:, ["sa_max", "sb_max", "sc_max"]] *= self.gen_mult
        if self.load_mult is not None:
            self.bus_data.loc[
                :, ["pl_a", "ql_a", "pl_b", "ql_b", "pl_c", "ql_c"]
            ] *= self.load_mult
        if self.v_swing is not None:
            self.bus_data.loc[
                self.bus_data.bus_type == "SWING", ["v_a", "v_b", "v_c"]
            ] = self.v_swing
        if self.v_min is not None:
            self.bus_data.loc[:, "v_min"] = self.v_min
        if self.v_max is not None:
            self.bus_data.loc[:, "v_max"] = self.v_max

        self.model = None
        self.results = None
        self.voltages_df = None
        self.power_flows_df = None
        self.decision_variables_df = None

    def run_pf(self):
        """
        Run the unconstrained power flow, save and plot the results.
        Returns
        -------
        voltages_df: pd.DataFrame
        power_flows_df: pd.DataFrame
        """
        bus_data = self.bus_data.copy()
        bus_data.loc[:, "v_min"] = 0.0
        bus_data.loc[:, "v_max"] = 2.0
        # Create model
        self.model = create_model(
            "",
            branch_data=self.branch_data,
            bus_data=bus_data,
            gen_data=self.gen_data,
            cap_data=self.cap_data,
            reg_data=self.reg_data,
        )
        # Solve
        result = auto_solve(self.model)

        self.voltages_df = self.model.get_voltages(result.x)
        self.power_flows_df = self.model.get_apparent_power_flows(result.x)
        fig1 = plot_network(
            self.model, self.voltages_df, self.power_flows_df, show_reactive_power=False
        )
        fig2 = plot_power_flows(self.power_flows_df)
        fig3 = plot_voltages(self.voltages_df)

        if self.save_inputs:
            config_parameters = {
                "objective_function": str(self.objective_function),
                "control_variable": self.control_variable,
            }
            if not self.output_dir.exists():
                self.output_dir.mkdir()
            case_data_dir = Path(self.output_dir) / "case_data"
            if not case_data_dir.exists():
                case_data_dir.mkdir()
            with open(Path(self.output_dir) / "case_data" / "config.json", "w") as f:
                json.dump(config_parameters, f, ensure_ascii=False, indent=4)
            self.branch_data.to_csv(
                Path(self.output_dir) / "case_data" / "branch_data.csv", index=False
            )
            self.bus_data.to_csv(
                Path(self.output_dir) / "case_data" / "bus_data.csv", index=False
            )

            if self.gen_data is not None:
                self.gen_data.to_csv(
                    Path(self.output_dir) / "case_data" / "gen_data.csv", index=False
                )

            if self.cap_data is not None:
                self.cap_data.to_csv(
                    Path(self.output_dir) / "case_data" / "cap_data.csv", index=False
                )

            if self.reg_data is not None:
                self.reg_data.to_csv(
                    Path(self.output_dir) / "case_data" / "reg_data.csv", index=False
                )
        if self.save_results:
            if not self.output_dir.exists():
                self.output_dir.mkdir()
            self.voltages_df.to_csv(
                Path(self.output_dir) / "node_voltages.csv", index=False
            )
            self.power_flows_df.to_csv(
                Path(self.output_dir) / "power_flows.csv", index=False
            )
            self.decision_variables_df.to_csv(
                Path(self.output_dir) / "decision_variables.csv", index=False
            )
        if self.save_plots:
            fig1.write_html(self.output_dir / "network_plot.html")
            fig2.write_html(self.output_dir / "power_flow_plot.html")
            fig3.write_html(self.output_dir / "voltage_plot.html")
        if self.show_plots:
            fig1.show(renderer="browser")
            fig2.show(renderer="browser")
            fig3.show(renderer="browser")
        return self.voltages_df, self.power_flows_df

    def run(self):
        """
        Run the optimization, save and plot the results.
        Returns
        -------
        voltages_df: pd.DataFrame
        power_flows_df: pd.DataFrame
        decision_variables_df: pd.DataFrame
        """

        # Create model
        self.model = create_model(
            self.control_variable,
            branch_data=self.branch_data,
            bus_data=self.bus_data,
            gen_data=self.gen_data,
            cap_data=self.cap_data,
            reg_data=self.reg_data,
        )
        # Solve
        result = auto_solve(
            self.model,
            self.objective_function,
            target=self.target,
            error_percent=self.error_percent,
        )

        self.voltages_df = self.model.get_voltages(result.x)
        self.power_flows_df = self.model.get_apparent_power_flows(result.x)
        self.decision_variables_df = self.model.get_decision_variables(result.x)
        fig1 = plot_network(
            self.model, self.voltages_df, self.power_flows_df, show_reactive_power=False
        )
        fig2 = plot_power_flows(self.power_flows_df)
        fig3 = plot_voltages(self.voltages_df)
        fig4 = plot_ders(self.decision_variables_df)

        if self.save_inputs:
            config_parameters = {
                "objective_function": str(self.objective_function),
                "control_variable": self.control_variable,
            }
            if not self.output_dir.exists():
                self.output_dir.mkdir()
            case_data_dir = Path(self.output_dir) / "case_data"
            if not case_data_dir.exists():
                case_data_dir.mkdir()
            with open(Path(self.output_dir) / "case_data" / "config.json", "w") as f:
                json.dump(config_parameters, f, ensure_ascii=False, indent=4)
            self.branch_data.to_csv(
                Path(self.output_dir) / "case_data" / "branch_data.csv", index=False
            )
            self.bus_data.to_csv(
                Path(self.output_dir) / "case_data" / "bus_data.csv", index=False
            )
            if self.gen_data is not None:
                self.gen_data.to_csv(
                    Path(self.output_dir) / "case_data" / "gen_data.csv", index=False
                )
            if self.cap_data is not None:
                self.cap_data.to_csv(
                    Path(self.output_dir) / "case_data" / "cap_data.csv", index=False
                )
            if self.reg_data is not None:
                self.reg_data.to_csv(
                    Path(self.output_dir) / "case_data" / "reg_data.csv", index=False
                )
        if self.save_results:
            if not self.output_dir.exists():
                self.output_dir.mkdir()
            self.voltages_df.to_csv(
                Path(self.output_dir) / "node_voltages.csv", index=False
            )
            self.power_flows_df.to_csv(
                Path(self.output_dir) / "power_flows.csv", index=False
            )
            self.decision_variables_df.to_csv(
                Path(self.output_dir) / "decision_variables.csv", index=False
            )
        if self.save_plots:
            fig1.write_html(self.output_dir / "network_plot.html")
            fig2.write_html(self.output_dir / "power_flow_plot.html")
            fig3.write_html(self.output_dir / "voltage_plot.html")
            fig4.write_html(self.output_dir / "decision_variable_plot.html")
        if self.show_plots:
            fig1.show(renderer="browser")
            fig2.show(renderer="browser")
            fig3.show(renderer="browser")
            fig4.show(renderer="browser")
        return self.voltages_df, self.power_flows_df, self.decision_variables_df

    def plot_network(
        self,
        v_min: int = 0.95,
        v_max: int = 1.05,
        show_phases: str = "abc",
        show_reactive_power: bool = False,
    ):
        """
        Plot the distribution network showing voltage and power results.
        Parameters
        ----------
        v_min : (default=0.95) Used for scaling node colors.
        v_max : (default=1.05) Used for scaling node colors.
        show_phases : (default="abc") valid options: "a", "b", "c", or "abc"
        show_reactive_power : (default=False) If True, show reactive power flows instead of active power flows.

        Returns
        -------
        fig: plotly.graph_objects.Figure
        """
        return plot_network(
            self.model,
            v=self.voltages_df,
            s=self.power_flows_df,
            control_values=self.decision_variables_df,
            control_variable=self.control_variable,
            v_min=v_min,
            v_max=v_max,
            show_phases=show_phases,
            show_reactive_power=show_reactive_power,
        )

    def plot_power_flows(self):
        """
        Plot the power flows
        Returns
        -------
        fig: plotly.graph_objects.Figure
        """
        return plot_power_flows(self.power_flows_df)

    def plot_voltages(self):
        """
        Plot the bus voltages
        Returns
        -------
        fig: plotly.graph_objects.Figure
        """
        return plot_voltages(self.voltages_df)

    def plot_decision_variables(self):
        """
        Plot the decision variables
        Returns
        -------
        fig: plotly.graph_objects.Figure
        """
        return plot_ders(self.decision_variables_df)

    # def delete_generator(self, node_name: str) -> None:
    #     gen = self.gen_data.copy()
    def add_generator(
        self,
        name: str,
        phases: str = None,
        p=0,
        q=0,
        s_rated=None,
        q_max=None,
        q_min=None,
    ):
        gen = self.gen_data.copy()
        i = gen.shape[0]
        _ids = self.bus_data.loc[self.bus_data.name == name, "id"].to_numpy()
        if len(_ids) == 0:
            raise ValueError(f"Bus {name} (type: {type(name)}) not found in bus_data.")
        _id = _ids[0]
        if _id in gen.loc[:, "id"].to_numpy():
            i = self.gen_data.loc[self.gen_data.id == _id, "id"].index[0]
        gen.at[i, "name"] = name
        gen.at[i, "id"] = _id
        bus_phases = self.bus_data.loc[self.bus_data.name == "13", "phases"].to_numpy()[
            0
        ]
        if phases is None:
            phases = bus_phases
        if s_rated is None:
            s_rated = (p**2 + q**2) ** (1 / 2) * 1.2
        n_phases = len(phases)
        p_phase = round(p / n_phases, 9)
        q_phase = round(q / n_phases, 9)
        s_rated_phase = round(s_rated / n_phases, 9)
        gen.loc[i, "phases"] = phases
        gen.loc[i, [f"s{ph}_max" for ph in phases]] = s_rated_phase  # unlimited
        gen.loc[i, [f"p{ph}" for ph in phases]] = p_phase  # unlimited
        gen.loc[i, [f"q{ph}" for ph in phases]] = q_phase  # unlimited
        if q_max is None:
            q_max = s_rated
        if q_min is None:
            q_min = -s_rated
        gen.loc[i, ["qa_max", "qb_max", "qc_max"]] = q_max  # unlimited
        gen.loc[i, ["qa_min", "qb_min", "qc_min"]] = q_min  # unlimited

        gen.loc[:, ["pa", "pb", "pc", "qa", "qb", "qc"]] = (
            gen.loc[:, ["pa", "pb", "pc", "qa", "qb", "qc"]].astype(float).fillna(0.0)
        )
        gen.loc[:, [f"s{a}_max" for a in "abc"]] = (
            gen.loc[:, [f"s{a}_max" for a in "abc"]].astype(float).fillna(0.0)
        )
        self.gen_data = gen

    def add_capacitor(
        self,
        name: str,
        phases: str = None,
        q=0,
    ):
        cap = self.cap_data.copy()
        i = cap.shape[0]
        _ids = self.bus_data.loc[self.bus_data.name == name, "id"].to_numpy()
        if len(_ids) == 0:
            raise ValueError(f"Bus {name} (type: {type(name)}) not found in bus_data.")
        _id = _ids[0]
        if _id in cap.loc[:, "id"].to_numpy():
            i = self.cap_data.loc[self.cap_data.id == _id, "id"].index[0]
        cap.at[i, "name"] = name
        cap.at[i, "id"] = _id
        bus_phases = self.bus_data.loc[self.bus_data.name == "13", "phases"].to_numpy()[
            0
        ]
        if phases is None:
            phases = bus_phases
        n_phases = len(phases)
        q_phase = round(q / n_phases, 9)
        cap.loc[i, "phases"] = phases
        cap.loc[i, [f"q{ph}" for ph in phases]] = q_phase  # unlimited
        cap.loc[i, [f"q{ph}" for ph in phases]] = (
            cap.loc[i, [f"q{ph}" for ph in phases]].astype(float).fillna(0.0)
        )
        self.cap_data = cap


if __name__ == "__main__":
    test_config = {
        "data_path": "ieee123_dss/Run_IEEE123Bus.DSS",
        "output_dir": "output",
        "control_variable": "Q",
        "v_max": 1.05,
        "v_min": 0.95,
        # etc...
        "objective_function": "loss_min",
        "solver": "CLARABEL",
        "show_plots": True,
    }
    # run_from_dict(config)
    case = DistOPFCase(
        config=test_config,
    )
    case.run()
