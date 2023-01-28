import numpy as np
import pandas as pd
import re
from os import system

# FETCH NEW EXO DATA:
# > cat nasa_exo_query.txt | xargs wget -o nasa_exo_PSCP.csv
fetch_DBs = True
update_exos = True
load_exos = True

'''
DATA/COLUMNS THAT ARE NEEDED
 - Prot/PRot/st_rotp
 - pl_name (Planet Name)
   - kic_id/kepid
   - koi/koi_name
   - tic_id
   - toi
   - gaia_id
 - pl_bmasse (Planet Mass in Earth Masses)
 - pl_rade (Planet Radius in Earth Radii)
 - pl_orbsmax (Planet Orbit Semi-Major Axis in AU)
 - pl_orbeccen (Planet Orbit Eccentricity)

 - sy_vmag (V-band magnitude of star)
 - sy_kmag (K-band magnitude of star)
 - st_rad (Stellar Radius in Solar Radii)
 - st_mass (Stellar Mass in Solar Masses)

 - notes:
   - ALL parameters EXCEPT names/ids and Prots are available via nasa/caltech
     - currently selecting names+prots from surveys and merging onto nasa
   - may want to use consistent parameter set (dont use PSCP db, instead use
   PS database and select only where 'Default Parameter Set is 1') or parameter
   set offered by the paper offering the Prot
'''

def init_nasa_exo():

    # Load NASA exoplanet database
    nasa_exo = pd.read_csv("tables/nasa_exo_PSCP.csv")

    # !! need to fetch new caltech DB of Kep confs and update
    # Add KOI/KIC ID columns of confirmed Kepler planets
    p_koi = re.compile(r'^K(\d+)\.\d{2}$')
    nasa_kep = pd.read_csv("tables/nasa_exo_kep.csv")
    nasa_kep['KOI'] = nasa_kep.apply(lambda r: int(r.koi_name[1:-3]) if p_koi.match(str(r.koi_name)) is not None else np.nan, axis=1)
    nasa_kep.rename(columns={'kepid': 'KIC'}, inplace=True)
    nasa_exo = pd.merge(nasa_exo, nasa_kep, how='left', on='pl_name')
    nasa_exo['KOI'].fillna(-1, inplace=True, downcast='infer')
    nasa_exo['KIC'].fillna(-1, inplace=True, downcast='infer')

    # Add TIC ID column
    p_tic = re.compile(r'^TIC (\d+)$')
    nasa_exo['TIC'] = nasa_exo.apply(lambda r: int(str(r.tic_id)[4:]) if p_tic.match(str(r.tic_id)) is not None else -1, axis=1)

    # Add GAIA ID columns
    nasa_exo['GAIA'] = nasa_exo.apply(lambda x: int(x['gaia_id'][9:]) if re.match(r'^Gaia DR2 \d+$', str(x['gaia_id'])) is not None else np.nan, axis=1)

    nasa_exo.to_csv('current-exo-data/nasa_exo.csv', index=False)

    return nasa_exo

# Update the short-list of exoplanet systems/stars shared by mcq13 and nasa
def update_exos_mcq13(nasa_exo):

    # Load McQuillan 2013 KOI stellar rotation rate survey
    mcq13_kois = pd.read_csv("tables/prot_mcq_2013.dat", sep="\s+")
    mcq13_kois = mcq13_kois[['KOI','PRot', 'e_PRot']]
    mcq13_kois.rename(columns={'PRot':'Prot','e_PRot': 'e_Prot'}, inplace=True)
    mcq13_kois.insert(len(mcq13_kois.columns), 'db', 'mcq13')
    mcq13_kois = mcq13_kois[mcq13_kois.Prot.notnull()]

    # Select stars from mcq13 whose KOI is in nasa db
    exos_mcq13 = pd.merge(nasa_exo, mcq13_kois, how='inner', on='KOI')[['hostname','KOI','Prot','e_Prot', 'db']]
    exos_mcq13.drop_duplicates(subset='hostname', inplace=True)
    exos_mcq13.to_csv('current-exo-data/exos_mcq13.csv', index=False)
    return exos_mcq13
    
def update_exos_mcq14(nasa_exo):

    # Load McQuillan 2014 KIC stellar rotation rate survey
    mcq14_kics = pd.read_csv("tables/prot_mcq_2014.dat", sep="\s+")
    mcq14_kics = mcq14_kics[['KIC','PRot', 'e_PRot']]
    mcq14_kics.rename(columns={'PRot':'Prot','e_PRot': 'e_Prot'}, inplace=True)
    mcq14_kics.insert(len(mcq14_kics.columns), 'db', 'mcq14')
    mcq14_kics = mcq14_kics[mcq14_kics.Prot.notnull()]

    exos_mcq14 = pd.merge(nasa_exo, mcq14_kics, how='inner', on='KIC')[['hostname','KIC','Prot','e_Prot', 'db']]
    exos_mcq14.drop_duplicates(subset='hostname', inplace=True)
    exos_mcq14.to_csv('current-exo-data/exos_mcq14.csv', index=False)
    return exos_mcq14

def update_exos_mar20(nasa_exo):

    # Load Martin 2020 TOI stellar rotation period survey
    mar20 = pd.read_csv("tables/prot_martin_2020_tic.csv", usecols=[0,1,2,3,4])
    mar20['TIC'] = mar20.apply(lambda x: int(x['TIC_ID']), axis=1)
    mar20.drop(columns='TIC_ID',inplace=True)
    mar20.rename(columns={'eProt':'e_Prot'}, inplace=True)
    mar20.insert(len(mar20.columns), 'db', 'mar20')
    mar20 = mar20[mar20.Prot.notnull()]

    # Selection criteria for rotation rates based on their classification/certainty
    #selection = (mar20['Variability_Classification'] == "Unambiguous_Rotation") & mar20['Prot'].notnull()
    selection = ((mar20['Variability_Classification'] == "Unambiguous_Rotation") | (mar20['Variability_Classification'] == "Dubious_Rotation")) & mar20['Prot'].notnull()
    mar20_prot = mar20[selection].copy(deep=False)
    mar20_prot['Prot'] = mar20_prot.apply(lambda x: select_prot_martin(x['Prot']), axis=1)

    exos_mar20 = pd.merge(nasa_exo, mar20_prot, how='inner', on='TIC')[['hostname','TIC','Prot','e_Prot', 'db']]
    exos_mar20.drop_duplicates(subset='hostname', inplace=True)
    exos_mar20.to_csv('current-exo-data/exos_mar20.csv', index=False)
    return exos_mar20

# fn can be np.mean/min/max, lambda x: np.nan
def select_prot_martin(prot_str, fn=np.min):
    if "/" in str(prot_str):
        prots = prot_str.split("/")
        protf = []
        for s in prots:
            protf.append(float(s))
        return fn(protf)
    else: return float(prot_str)

def update_exos_custom(nasa_exo):
    arm16_prot = pd.read_csv("tables/custom_prot.txt", sep="\s+", header=14, nrows=7)
    
    arm16_prot = arm16_prot[['hostname','prot_acf','eprot_acf']]
    arm16_prot.rename(columns={'prot_acf':'Prot','eprot_acf':'e_Prot'}, inplace=True)
    arm16_prot.insert(len(arm16_prot.columns), 'db', 'arm16')
    arm16_prot = arm16_prot[arm16_prot.Prot.notnull()]

    exos_arm16 = pd.merge(nasa_exo, arm16_prot, how='inner', on='hostname')[['hostname','Prot','e_Prot', 'db']]
    exos_arm16.drop_duplicates(subset='hostname', inplace=True)
    exos_arm16.to_csv('current-exo-data/exos_arm16.csv', index=False)
    return exos_arm16

def update_exos_lu22(nasa_exo):
    lu22_gaia = pd.read_csv('tables/prot_lu_gaia.txt', header=21, sep="\s+")

    lu22_gaia = lu22_gaia[['GAIA','Prot']]
    lu22_gaia.insert(len(lu22_gaia.columns), 'db', 'lu22')
    lu22_gaia = lu22_gaia[lu22_gaia.Prot.notnull()]

    # problem here: discrepancy in number of entries
    # exos_lu22 = lu22_gaia[lu22_gaia.GAIA.isin(nasa_exo.GAIA)]

    exos_lu22 = pd.merge(nasa_exo, lu22_gaia, how='inner', on='GAIA')[['hostname','GAIA','Prot', 'db']]
    exos_lu22.drop_duplicates(subset='hostname', inplace=True)

    exos_lu22.to_csv('current-exo-data/exos_lu22.csv',index=False)
    return exos_lu22

def update_exos_nasa(nasa_exo):
    exos_nasa = nasa_exo[nasa_exo.st_rotp.notnull()][['hostname','st_rotp','st_rotperr1','st_rotperr2']]
    exos_nasa['st_rotperr'] = nasa_exo.apply(lambda r: np.max([r.st_rotperr1, np.fabs(r.st_rotperr2)]), axis=1)
    exos_nasa.rename(columns={'st_rotp':'Prot', 'st_rotperr': 'e_Prot'}, inplace=True)
    exos_nasa.drop_duplicates(subset='hostname', inplace=True)
    exos_nasa.drop(columns=['st_rotperr1','st_rotperr2'], inplace=True)
    exos_nasa.insert(len(exos_nasa.columns), 'db', 'nasa')

    exos_nasa.to_csv('current-exo-data/exos_nasa.csv', index=False)

    return exos_nasa

def update_exos_habitable(nasa_exo):
    habitable = pd.read_csv('tables/habitable.txt', header=1)

    exos_habitable = pd.merge(nasa_exo, habitable, how='inner', on='pl_name')[['hostname','pl_name']]

    exos_habitable.to_csv('current-exo-data/exos_habitable.csv', index=False)

def main():
    if fetch_DBs:

        print('Updating Planetary Systems Composite Parameters (pscomppars) database!')
        # Get updated Planetary Systems Combined Parameters DB
        system('cat nasa_exo_query.txt | xargs wget -o log_PSCP -O tables/nasa_exo_PSCP.csv')

        print('Updating Kepler confirmed planets (kepnames) database!')
        system('cat nasa_exo_kep_query.txt | xargs wget -o log_kep -O tables/nasa_exo_kep.csv')

    if update_exos:

        nasa_exo = init_nasa_exo()

        exos_mcq13 = update_exos_mcq13(nasa_exo)
        exos_mcq14 = update_exos_mcq14(nasa_exo)
        exos_arm16 = update_exos_custom(nasa_exo)
        exos_mar20 = update_exos_mar20(nasa_exo)
        exos_lu22 = update_exos_lu22(nasa_exo)
        exos_nasa = update_exos_nasa(nasa_exo)
        update_exos_habitable(nasa_exo)

        print(exos_mcq13.count())
        print(exos_mcq14.count())
        print(exos_arm16.count())
        print(exos_mar20.count())
        print(exos_lu22.count())
        print(exos_nasa.count())

    if load_exos:
        exos_mcq13 = pd.read_csv('current-exo-data/exos_mcq13.csv')
        exos_mcq14 = pd.read_csv('current-exo-data/exos_mcq14.csv')
        exos_arm16 = pd.read_csv('current-exo-data/exos_arm16.csv')
        exos_mar20 = pd.read_csv('current-exo-data/exos_mar20.csv')
        exos_lu22 = pd.read_csv('current-exo-data/exos_lu22.csv')
        exos_nasa = pd.read_csv('current-exo-data/exos_nasa.csv')

        exos = pd.concat([exos_mcq13, exos_mcq14, exos_arm16, exos_mar20, exos_lu22, exos_nasa], ignore_index=True)
        exos['e_Prot'] = exos.apply(lambda r: np.nan if r.e_Prot == 0.0 else r.e_Prot, axis=1)
        exos['KOI'].fillna(-1, inplace=True, downcast='infer')
        exos['KIC'].fillna(-1, inplace=True, downcast='infer')
        exos['TIC'].fillna(-1, inplace=True, downcast='infer')
        exos['GAIA'].fillna(-1, inplace=True, downcast='infer')

        exos.sort_values('e_Prot', inplace=True, ignore_index=True, ascending=True)
        exos.drop_duplicates(subset='hostname', inplace=True)
        exos = exos[['hostname', 'Prot', 'e_Prot', 'KOI', 'KIC', 'TIC', 'GAIA', 'db']]
        exos.to_csv('current-exo-data/exos.csv', index=False)
        print(exos.count())
    

if __name__ == '__main__':
    main()