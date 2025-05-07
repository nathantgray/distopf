import time

import distopf.cases
from distopf.pyomo_models.parse_input import parse_all_data
from distopf.pyomo_models.constraints import build_pyomo_model
from distopf.pyomo_models.objective import (
    pyomo_solve,
    cost_minimize,
    loss_minimize,
    power_flow,
    loss_minimize_with_scd,
    cost_minimize_with_scd,
)
from distopf.pyomo_models.store import store_results
import pandas as pd
from pathlib import Path

from distopf.cases import CASES_DIR

system_name = "ieee123_30der"
filepath = CASES_DIR / ("csv/" + system_name)
# Import CSV files
bus_data = pd.read_csv(Path(filepath / "bus_data.csv"))
branch_data = pd.read_csv(Path(filepath / "branch_data.csv"))
gen_data = pd.read_csv(Path(filepath / "gen_data.csv"))
bat_data = pd.read_csv(Path(filepath / "battery_data.csv"))
loadshape_data = pd.read_csv(Path(filepath / "default_loadshape.csv"))
pvshape_data = pd.read_csv(Path(filepath / "pv_loadshape.csv"))
# price = [
#     0.026, 0.025, 0.022, 0.02, 0.022, 0.024, 0.025, 0.026,
#     0.028, 0.034, 0.038, 0.035, 0.036, 0.037, 0.038, 0.04,
#     0.04, 0.03, 0.031, 0.029, 0.027, 0.025, 0.023, 0.026]
price = [
    0.027,
    0.025,
    0.023,
    0.022,
    0.022,
    0.026,
    0.029,
    0.030,
    0.031,
    0.031,
    0.035,
    0.036,
    0.033,
    0.029,
    0.032,
    0.032,
    0.038,
    0.040,
    0.034,
    0.037,
    0.027,
    0.028,
    0.025,
    0.024,
]
# data_area = split_data_into_areas(data, area_info)
# plot_network(bus_data,branch_data,gen_data,bat_data,data_area)
# data = parse_all_data(bus_data, branch_data,price=price,n_steps=1)
# from rawData.IEEE_123_other.dss_scripts.write_dss_scripts import create_opendss_scripts
# create_opendss_scripts(data)
# %%
if __name__ == "__main__":
    data = parse_all_data(
        bus_data,
        branch_data,
        gen=gen_data,
        bat=bat_data,
        loadshape=loadshape_data,
        pvshape=pvshape_data,
        price=price,
        start_step=0,
        n_steps=5,
    )
    obj = loss_minimize_with_scd
    print(
        f"Solving centralized problem for {system_name} and objective function {obj}..."
    )
    start_time = time.time()  # Start timing
    centralized_model = build_pyomo_model(data)
    centralized_model = pyomo_solve(centralized_model, obj)
    copfVals = store_results(centralized_model)
    end_time = time.time()  # End timing
    centralized_time = end_time - start_time
    print(f"Total substation Real Power Flows: {sum(copfVals['P_subs'].values())}")
    print(f"Total substation Reactive Power Flows: {sum(copfVals['Q_subs'].values())}")
    print(f"Total reactive power from PV : {sum(copfVals['q_D'].values())}")
    print(f"Total battery Charging Power : {sum(copfVals['P_c'].values())}")
    print(f"Total battery disCharging Power : {sum(copfVals['P_d'].values())}")
    print(f"Centralized Objective Value: {copfVals['objective_value']}")
    print(f"Centralized Solver Time: {centralized_time:.2f} seconds")
