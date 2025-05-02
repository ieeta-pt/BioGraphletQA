import kgw

oregano = kgw.biomedicine.Oregano(version="v3", workdir="../_graph")
oregano.to_graphml()
oregano.to_schema()
oregano.to_statistics()
oregano.to_metta()
oregano.to_csv()

status = kgw.run(oregano)