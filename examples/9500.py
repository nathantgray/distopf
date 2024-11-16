import distopf as opf

# dss_parser = opf.DSSParser(opf.CASES_DIR/"dss"/"9500-primary-network"/"Master.dss")
# dss_parser.to_csv(opf.CASES_DIR/"csv"/"9500")
case = opf.DistOPFCase(data_path="9500")
result = case.run_pf()
case.plot_network().write_html("9500_network.html")
pass
