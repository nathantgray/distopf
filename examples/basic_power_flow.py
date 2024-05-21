import distopf as opf
import pandas as pd

case_name = "ieee123"

branch_data = pd.read_csv(
    opf.CASES_DIR / "csv" / case_name / "branch_data.csv", header=0
)
bus_data = pd.read_csv(opf.CASES_DIR / "csv" / case_name / "bus_data.csv", header=0)
gen_data = pd.read_csv(opf.CASES_DIR / "csv" / case_name / "gen_data.csv", header=0)
cap_data = pd.read_csv(opf.CASES_DIR / "csv" / case_name / "cap_data.csv", header=0)
reg_data = pd.read_csv(opf.CASES_DIR / "csv" / case_name / "reg_data.csv", header=0)

case = opf.DistOPFCase(
    branch_data=branch_data,
    bus_data=bus_data,
    cap_data=cap_data,
    reg_data=reg_data,
)
case.run_pf()
case.plot_network().show(renderer="browser")
