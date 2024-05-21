import pandas as pd
from distopf.dssconverter.dssparser import DSSParser
from distopf.dssconverter.savedss import savedsscsv

dss_data13 = DSSParser(r"dssfiles/ieee13_dss/IEEE13Nodeckt.dss")
savedsscsv(dss_data13, folderpath="parsedcsvs/IEEE13buscsvs")
dss_data123 = DSSParser(r"dssfiles/ieee123_dss/Run_IEEE123Bus.DSS")
savedsscsv(dss_data123, folderpath="parsedcsvs/IEEE123buscsvs")
# dss_data9500 = DSSParser(r"dssfiles/ieee9500_dss/Master-unbal-initial-config.dss")
# savedsscsv(dss_data9500, folderpath="parsedcsvs/IEEE9500buscsvs")
