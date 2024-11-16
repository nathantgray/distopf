import pandas as pd

from distopf.glmpy.converter.csv2glm import Csv2glm
from distopf.glmpy.converter.glm2csv import Glm2csv
from distopf.glmpy import Gridlabd
import distopf as opf
from pathlib import Path


"""
(
        self,
        output_name,
        branch_csv,
        bus_csv,
        p_gen_override=None,  # P_gen override used for DG curtailment.
        q_gen=None,
        seed_model=None,
        down_nodes=None,
        make_up_player=False,
        model_results_out_dir="",
        sep=",",
        cvr=None,
        single_run=False,
        helics_config=None,
        v_ln_base=2401.78,
        v_ss_pu=1,
        s_dn_pu=None,
        s_base=1000,
        gen_mult=None,
        q_gen_mult=None,
        load_mult=None,
        rating_mult=1.2,
        multiplier_update_period=60,
        opf_model=None,
        tz="PST+8PDT",
        starttime="'2001-08-01 12:00:00'",
        stoptime="'2001-08-01 12:00:00'",
    ):
"""

if __name__ == "__main__":
    case_path = opf.CASES_DIR / "csv" / "ieee123_30der"
    case_path.exists()
    csv2glm = Csv2glm(
        output_name="system.glm",
        branch_data=pd.read_csv(case_path / "branch_data.csv"),
        bus_data=pd.read_csv(case_path / "bus_data.csv"),
        gen_data=pd.read_csv(case_path / "gen_data.csv"),
        cap_data=pd.read_csv(case_path / "cap_data.csv"),
        reg_data=pd.read_csv(case_path / "reg_data.csv"),
        tz="PST+8PDT",
        starttime="'2001-08-01 12:00:00'",
        stoptime="'2001-08-01 12:00:010'",
    )
    results = csv2glm.glm.run(Path.cwd())
    print(results)
