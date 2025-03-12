import pandas as pd
from distopf.multiperiod.base_mp import LinDistBaseMP


class LinDistMP(LinDistBaseMP):
    """
    LinDistMP Model class for linear multistep optimal power flow modeling.

    This class represents a linearized distribution model used for calculating
    power flows, voltages, and other system properties in a distribution network
    using the linearized branch-flow formulation from [1]. The model is composed of several power system components
    such as buses, branches, generators, capacitors, and regulators.

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
    n_steps : int,
        Number of time intervals for multi period optimization. Default is 24.

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
    """

    def __init__(
        self,
        branch_data: pd.DataFrame = None,
        bus_data: pd.DataFrame = None,
        gen_data: pd.DataFrame = None,
        cap_data: pd.DataFrame = None,
        reg_data: pd.DataFrame = None,
        bat_data: pd.DataFrame = None,
        loadshape_data: pd.DataFrame = None,
        pv_loadshape_data: pd.DataFrame = None,
        start_step: int = 0,
        n_steps: int = 24,
        delta_t: float = 1,  # hours per step
    ):
        super().__init__(
            branch_data=branch_data,
            bus_data=bus_data,
            gen_data=gen_data,
            cap_data=cap_data,
            reg_data=reg_data,
            bat_data=bat_data,
            loadshape_data=loadshape_data,
            pv_loadshape_data=pv_loadshape_data,
            start_step=start_step,
            n_steps=n_steps,
            delta_t=delta_t,
        )
        self.build()
