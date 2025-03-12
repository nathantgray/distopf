import pandas as pd
from typing import Optional


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


def handle_gen_input(gen_data: Optional[pd.DataFrame]) -> pd.DataFrame:
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
                "control_variable",
            ]
        )
    if f"control_variable" not in gen_data.columns:
        gen_data[f"control_variable"] = ""
    gen = gen_data.sort_values(by="id", ignore_index=True)
    gen.index = gen.id.to_numpy() - 1
    return gen


def handle_cap_input(cap_data: Optional[pd.DataFrame]) -> pd.DataFrame:
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


def handle_reg_input(reg_data: Optional[pd.DataFrame]) -> pd.DataFrame:
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
            reg[f"ratio_{ph}"] = 1 + 0.00625 * reg[f"tap_{ph}"]
            # check consistency
            # if any(abs(reg[f"ratio_{ph}"]) - (1 + 0.00625 * reg[f"tap_{ph}"]) > 1e-6):
            #     raise ValueError(
            #         f"Regulator taps and ratio are inconsistent on phase {ph}!"
            #     )
    return reg


def handle_branch_input(branch_data: Optional[pd.DataFrame]) -> pd.DataFrame:
    if branch_data is None:
        raise ValueError("Branch data must be provided.")
    branch = branch_data.sort_values(by="tb", ignore_index=True)
    branch = branch.loc[branch.status != "OPEN", :]
    return branch


def handle_bus_input(bus_data: Optional[pd.DataFrame]) -> pd.DataFrame:
    if bus_data is None:
        raise ValueError("Bus data must be provided.")
    bus = bus_data.sort_values(by="id", ignore_index=True)
    bus.index = bus.id.to_numpy() - 1
    return bus


def handle_loadshape_input(loadshape_data: Optional[pd.DataFrame]) -> pd.DataFrame:
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


def handle_pv_loadshape_input(
    pv_loadshape_data: Optional[pd.DataFrame],
) -> pd.DataFrame:
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


def handle_bat_input(bat_data: Optional[pd.DataFrame]) -> pd.DataFrame:
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
    if "b0_a" not in bat_data.columns:
        bat_data["b0_a"] = bat_data.bmin_a
    if "b0_b" not in bat_data.columns:
        bat_data["b0_b"] = bat_data.bmin_a
    if "b0_c" not in bat_data.columns:
        bat_data["b0_c"] = bat_data.bmin_a
    bat = bat_data.sort_values(by="id", ignore_index=True)
    bat.index = bat.id.to_numpy() - 1
    return bat
