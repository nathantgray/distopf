from typing import Optional
import pandas as pd
from distopf.base import LinDistBase
import distopf as opf


class LinDistModel(LinDistBase):
    """
    LinDistFlow Model class for linear power flow modeling.

    This class represents a linearized distribution model used for calculating
    power flows, voltages, and other system properties in a distribution network
    using the linearized branch-flow formulation from [1]. The model is composed of several power system components
    such as buses, branches, generators, capacitors, and regulators.

    Parameters
    ----------
    branch_data : pd.DataFrame
        DataFrame containing branch data including resistance and reactance values and limits.
    bus_data : pd.DataFrame
        DataFrame containing bus data such as loads, voltages, and limits.
    gen_data : pd.DataFrame
        DataFrame containing generator data.
    cap_data : pd.DataFrame
        DataFrame containing capacitor data.
    reg_data : pd.DataFrame
        DataFrame containing regulator data.

    References
    ----------
    [1] R. R. Jha, A. Dubey, C.-C. Liu, and K. P. Schneider,
    “Bi-Level Volt-VAR Optimization to Coordinate Smart Inverters
    With Voltage Control Devices,”
    IEEE Trans. Power Syst., vol. 34, no. 3, pp. 1801–1813,
    May 2019, doi: 10.1109/TPWRS.2018.2890613.

    Examples
    --------
    This example demonstrates how to set up and solve a linear distribution flow model
    using a provided case, and visualize the results.

    >>> import distopf as opf
    >>> # Prepare the case data
    >>> case = opf.DistOPFCase(data_path="ieee123_30der")
    >>> # Initialize the LinDistModel
    >>> model = LinDistModel(
    ...     branch_data=case.branch_data,
    ...     bus_data=case.bus_data,
    ...     gen_data=case.gen_data,
    ...     cap_data=case.cap_data,
    ...     reg_data=case.reg_data,
    ... )
    >>> # Solve the model using the specified objective function
    >>> result = opf.lp_solve(model, opf.gradient_load_min(model))
    >>> # Extract and plot results
    >>> v = model.get_voltages(result.x)
    >>> s = model.get_apparent_power_flows(result.x)
    >>> p_gens = model.get_p_gens(result.x)
    >>> q_gens = model.get_q_gens(result.x)
    >>> # Visualize network and power flows
    >>> opf.plot_network(model, v=v, s=s, p_gen=p_gens, q_gen=q_gens).show()
    >>> opf.plot_voltages(v).show()
    >>> opf.plot_power_flows(s).show()
    >>> opf.plot_gens(p_gens, q_gens).show()
    """

    def __init__(
        self,
        branch_data: Optional[pd.DataFrame] = None,
        bus_data: Optional[pd.DataFrame] = None,
        gen_data: Optional[pd.DataFrame] = None,
        cap_data: Optional[pd.DataFrame] = None,
        reg_data: Optional[pd.DataFrame] = None,
    ):
        super().__init__(branch_data, bus_data, gen_data, cap_data, reg_data)
        self.build()


if __name__ == "__main__":

    # Prepare the case data
    case = opf.DistOPFCase(data_path="ieee123_30der")
    # Initialize the LinDistModel
    model = LinDistModel(
        branch_data=case.branch_data,
        bus_data=case.bus_data,
        gen_data=case.gen_data,
        cap_data=case.cap_data,
        reg_data=case.reg_data,
    )
    # Solve the model using the specified objective function
    result = opf.lp_solve(model, opf.gradient_load_min(model))
    # Extract and plot results
    v = model.get_voltages(result.x)
    s = model.get_apparent_power_flows(result.x)
    # s = model.get_apparent_power_flows(result.x)
    # p_gens = model.get_p_gens(result.x)
    # q_gens = model.get_q_gens(result.x)
    # # Visualize network and power flows
    # opf.plot_network(model, v=v, s=s, p_gen=p_gens, q_gen=q_gens).show()
    # opf.plot_voltages(v).show()
    # opf.plot_power_flows(s).show()
    # opf.plot_gens(p_gens, q_gens).show()
