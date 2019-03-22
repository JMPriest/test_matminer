from matminer.data_retrieval.retrieve_MP import MPDataRetrieval
from pymatgen.electronic_structure.plotter import BSDOSPlotter
from matminer.data_retrieval.retrieve_Citrine import CitrineDataRetrieval
from matminer.data_retrieval.retrieve_MDF import MDFDataRetrieval

mpdr = MPDataRetrieval()

df = mpdr.get_dataframe(criteria={"nelements": 1}, properties=['density', 'pretty_formula'])
print("There are {} entries on MP with 1 element".format(df['density'].count()))
print(df.head())
df = mpdr.get_dataframe({"band_gap": {"$gt": 4.0}}, ['pretty_formula', 'band_gap'])
print("There are {} entries on MP with a band gap larger than 4.0".format(df['band_gap'].count()))
df.to_csv('gt4.csv')
df = mpdr.get_dataframe({"elasticity": {"$exists": True}, "elasticity.warnings": []},
                        ['pretty_formula', 'elasticity.K_VRH', 'elasticity.G_VRH'])
print("There are {} elastic entries on MP with no warnings".format(df['elasticity.K_VRH'].count()))
df = mpdr.get_dataframe(criteria={"elasticity": {"$exists": True},
                         "elasticity.warnings": [],
                         "elements": {"$all": ["Pb", "Te"]},
                         "e_above_hull": {"$lt": 1e-6}}, # to limit the number of hits for the sake of time
                        properties = ["elasticity.K_VRH", "elasticity.G_VRH", "pretty_formula",
                                      "e_above_hull", "bandstructure", "dos"])
print("Pb,Te(K_VRH,G_VRH,pretty_formula,e_above_hull,bandstructure,dos):")
print(df.head())

mpid = 'mp-20740'
idx = df.index[df.index==mpid][0]
plt = BSDOSPlotter().get_plot(bs=df.loc[idx, 'bandstructure'], dos=df.loc[idx, 'dos'])
plt.savefig('mp-20740.png')

cdr = CitrineDataRetrieval()

df_OH = cdr.get_dataframe(criteria={}, properties=['adsorption energy of OH'], secondary_fields=True)
df_O = cdr.get_dataframe(criteria={}, properties=['adsorption energy of O'], secondary_fields=True)
print('adsorption energy of OH\n')
print(df_OH.head())
print('adsorption energy of O\n')
print(df_O.head())

mdf_dr = MDFDataRetrieval(anonymous=True)

df = mdf_dr.get_dataframe(criteria={'elements': ['Ag', 'Be'], 'sources': ["oqmd"]})
print('Ag,Be(oqmd):\n')
print(df.head())
print("There are {} entries in the Ag-Be chemical system".format(len(df)))