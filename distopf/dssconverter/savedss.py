from distopf import DSSParser
from pathlib import Path


def savedsscsv(
    dssparser: DSSParser, folderpath: str = None, overwrite: bool = True
) -> None:

    if folderpath is None:
        folderpath = "testfiles"

    Path(folderpath).mkdir(parents=True, exist_ok=overwrite)
    dssparser.branch_data.to_csv(f"{folderpath}/branch_data.csv", index=False)
    dssparser.bus_data.to_csv(f"{folderpath}/bus_data.csv", index=False)
    dssparser.cap_data.to_csv(f"{folderpath}/cap_data.csv", index=False)
    dssparser.gen_data.to_csv(f"{folderpath}/gen_data.csv", index=False)
    dssparser.reg_data.to_csv(f"{folderpath}/reg_data.csv", index=False)


def main() -> None:

    # dss_data = DSSParser(r'ieee13Bus/IEEE13Nodeckt.dss')
    # savedsscsv(dss_data, folderpath="13buscsv")

    dss_data = DSSParser(r"ieee9500_dss/Master-unbal-initial-config.dss")
    savedsscsv(dss_data, folderpath="9500buscsv")


if __name__ == "__main__":
    main()
