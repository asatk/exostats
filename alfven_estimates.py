import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from plot import plot

calculate = True

AVK = 0.64
dAVK = 0.12
BVK = 0.25
dBVK = 0.08

AM = 2.33
dAM = 0.06
BM = -1.50
dBM = 0.21
CM = 0.31
dCM = 0.17

LN10 = np.log(10)

# wright et al 2018 (eqn 5): valid for range 1.1 < V-K < 7.0
def taucVK(V, K):
    if np.fabs(V - K - 4.05) <= 2.95:
        return pow(10., AVK + BVK * (V - K))
    else:
        return np.nan
    
def dtaucVK(V, K, dV, dK):
    return taucVK(V, K) * LN10 * np.sqrt(pow(dAVK, 2.) + 
                                         pow((V - K) * dBVK, 2.) + 
                                         pow(BVK * np.sqrt(pow(dV, 2.) + pow(dK, 2.)), 2.))

def RoVK(Prot, V, K):
    return np.fabs(Prot) / taucVK(V,K)

def dRoVK(Prot, V, K, dProt, dV, dK):
    return RoVK(Prot, V, K) * np.sqrt(pow(dProt / Prot, 2.) +
                                      pow(dtaucVK(V, K, dV, dK) / taucVK(V, K), 2.))

# wright et al 2018 (eqn 6): valid for range 0.08 < M/Msol < 1.36
#M in solar masses
def taucM(M):
    if np.fabs(M - 0.72) <= 0.64:
        return pow(10., AM + BM * M + CM * pow(M, 2.)) 
    else:
        return np.nan
    
def dtaucM(M, dM):
    return taucM(M) * LN10 * np.sqrt(pow(dAM, 2.) +
                                     pow(M * dBM, 2.) +
                                     pow(pow(M, 2.) * dCM, 2.) +
                                     pow((2 * M * CM + BM) * dM, 2.))
    
def RoM(Prot, M):
    return np.fabs(Prot) / taucM(M)

def dRoM(Prot, M, dProt, dM):
    return RoM(Prot, M) * np.sqrt(pow(dProt / Prot, 2.) + 
                                  pow(dtaucM(M, dM) / taucM(M), 2.))

def RoAvg(RoVK, RoM):
    return np.nanmean([RoVK, RoM])
    
def dRoAvg(dRoVK, dRoM):
    return np.sqrt(np.nansum([pow(dRoVK, 2.),pow(dRoM, 2.)]))

def chooseRo(RoVK, RoM):
    if RoVK != np.nan:
        return RoVK
    elif RoM != np.nan:
        return RoM
    return np.nan

def choosedRo(dRoVK, dRoM):
    if dRoVK is not np.nan:
        return dRoVK
    elif dRoM is not np.nan:
        return dRoM
    return np.nan

prot_sol = 27.
ro_sol = 1.85
dro_sol = 0.26
ra_sol = 0.1
dra_sol = 0.02
r_sol = 1.
# s = -1.38
# ds = 0.14
s = -1.19
ds = 0.14
r = -0.16
dr = 0.13
# r = r - dr
# r = r + dr
# all values scaled to solar values at maximum
# using Vidotto et al. 2014 p 6
# combining eqns: (7) Farrish 2019 and (2) Farrish 2021

# def ra_schrijver(ro, rad):
#     return ra_sol * np.real(pow(ro / ro_sol, s * r) * pow(rad / r_sol, 2 * r))

# def dra(ro, rad, dro, drad):
#     return ra_schrijver(ro, rad) * np.sqrt(pow(dra_sol / ra_sol, 2.) +
#                                            pow(s * r * dro / ro, 2.) +
#                                            pow(s * r * dro_sol / ro_sol, 2.) +
#                                            pow(2 * r * drad / rad, 2.) +
#                                            pow(r * np.log(ro / ro_sol) * ds, 2.) +
#                                            pow((s * np.log(ro / ro_sol) + 2 * np.log(rad)) * dr, 2.))

def ra_schrijver(ro):
    return ra_sol * np.real(pow(ro / ro_sol, s * r))
                                           

def dra(ro, dro):
    return ra_schrijver(ro) * np.sqrt(pow(dra_sol / ra_sol, 2.) +
                                           pow(s * r * dro / ro, 2.) +
                                           pow(s * r * dro_sol / ro_sol, 2.) +
                                           pow(r * np.log(ro / ro_sol) * ds, 2.) +
                                           pow((s * np.log(ro / ro_sol)) * dr, 2.))

def measured_uncertainties(nasa_exo):
    nasa_exo['pl_bmasseerr'] = nasa_exo.apply(lambda x: np.max([x['pl_bmasseerr1'],np.fabs(x['pl_bmasseerr2'])]), axis=1)
    nasa_exo['pl_orbsmaxerr'] = nasa_exo.apply(lambda x: np.max([x['pl_orbsmaxerr1'],np.fabs(x['pl_orbsmaxerr2'])]), axis=1)
    nasa_exo['pl_orbeccenerr'] = nasa_exo.apply(lambda x: np.max([x['pl_orbeccenerr1'],np.fabs(x['pl_orbeccenerr2'])]), axis=1)
    nasa_exo['pl_radeerr'] = nasa_exo.apply(lambda x: np.max([x['pl_radeerr1'],np.fabs(x['pl_radeerr2'])]), axis=1)
    nasa_exo['st_masserr'] = nasa_exo.apply(lambda x: np.max([x['st_masserr1'],np.fabs(x['st_masserr2'])]), axis=1)
    nasa_exo['st_raderr'] = nasa_exo.apply(lambda x: np.max([x['st_raderr1'],np.fabs(x['st_raderr2'])]), axis=1)
    nasa_exo['sy_disterr'] = nasa_exo.apply(lambda x: np.max([x['sy_disterr1'],np.fabs(x['sy_disterr2'])]), axis=1)
    nasa_exo['sy_kmagerr'] = nasa_exo.apply(lambda x: np.max([x['sy_kmagerr1'],np.fabs(x['sy_kmagerr2'])]), axis=1)
    nasa_exo['sy_vmagerr'] = nasa_exo.apply(lambda x: np.max([x['sy_vmagerr1'],np.fabs(x['sy_vmagerr2'])]), axis=1)
    return nasa_exo

def estimate_rossby(prot_data):
    prot_data['RoVK'] = prot_data.apply(lambda x: RoVK(x['Prot'], x['sy_vmag'], x['sy_kmag']), axis=1)
    prot_data['dRoVK'] = prot_data.apply(lambda x: dRoVK(x['Prot'], x['sy_vmag'], x['sy_kmag'], x['e_Prot'], x['sy_vmagerr'], x['sy_kmagerr']), axis=1)
    prot_data['TaucVK'] = prot_data.apply(lambda x: taucVK(x['sy_vmag'], x['sy_kmag']), axis=1)
    prot_data['dTaucVK'] = prot_data.apply(lambda x: dtaucVK(x['sy_vmag'], x['sy_kmag'], x['sy_vmagerr'], x['sy_kmagerr']), axis=1)
    prot_data['RoM'] = prot_data.apply(lambda x: RoM(x['Prot'], x['st_mass']), axis=1)
    prot_data['dRoM'] = prot_data.apply(lambda x: dRoM(x['Prot'], x['st_mass'], x['e_Prot'], x['st_masserr']), axis=1)
    prot_data['TaucM'] = prot_data.apply(lambda x: taucM(x['st_mass']), axis=1)
    prot_data['dTaucM'] = prot_data.apply(lambda x: dtaucM(x['st_mass'], x['st_masserr']), axis=1)
    prot_data['RoAvg'] = prot_data.apply(lambda x: RoAvg(x['RoVK'], x['RoM']), axis=1)
    prot_data['dRoAvg'] = prot_data.apply(lambda x: dRoAvg(x['dRoVK'], x['dRoM']), axis=1)

    #choose which Ro is used in plots - could probably be done more elegantly
    prot_data['Ro'] = prot_data.apply(lambda x: chooseRo(x['RoVK'], x['RoM']), axis=1)
    prot_data['dRo'] = prot_data.apply(lambda x: choosedRo(x['dRoVK'], x['dRoM']), axis=1)

    return prot_data

def estimate_alfven(data):
    
    # Estimate value and uncertainty for periastron
    data['rperi'] = data.apply(lambda x: x['pl_orbsmax'] * (1 - x['pl_orbeccen']), axis=1)
    data['drperi'] = data.apply(lambda x: np.sqrt(np.nansum([
        pow(x['pl_orbsmaxerr'] * (1 - x['pl_orbeccen']),2.),
        pow(x['pl_orbsmax'] * x['pl_orbeccenerr'], 2.)])), axis=1)

    # Calculate Shrijver scaling relation for mean AS radius

    # Estimate closest distance from star vs mean AS radius ratio
    # alfven = data.apply(lambda x: x['rperi'] / ra_schrijver(x['Ro'], x['st_rad']), axis=1)
    # dalfven = data.apply(lambda x: x['rperi'] / ra_schrijver(x['Ro'], x['st_rad']) *
    #                     np.sqrt(pow(x['drperi'] / x['rperi'], 2.) +
    #                     pow(dra(x['Ro'], x['st_rad'], x['dRo'], x['st_raderr'] / ra_schrijver(x['Ro'], x['st_rad'])), 2.)), axis=1)
    
    alfven = data.apply(lambda x: x['rperi'] / ra_schrijver(x['Ro']), axis=1)
    dalfven = data.apply(lambda x: x['rperi'] / ra_schrijver(x['Ro']) *
                        np.sqrt(pow(x['drperi'] / x['rperi'], 2.) +
                        pow(dra(x['Ro'], x['dRo'] / ra_schrijver(x['Ro'])), 2.)), axis=1)

    alfven_data = data.copy(deep=False)
    alfven_data['orbit2alfven'] = alfven
    alfven_data['dorbit2alfven'] = dalfven
    
    # select out data where the Stellar Radius entry is non-empty
    alfven_data = alfven_data[alfven_data['st_rad'].notnull()]
    
    #optional: remove 99th quantile and bad vals
    alfven_data = alfven_data[(alfven_data['Ro'] < np.quantile(alfven_data['Ro'], 0.99)) & (alfven_data['orbit2alfven'] < np.quantile(alfven_data['orbit2alfven'], 0.99))]
    alfven_data = alfven_data[(alfven_data['Ro'] > 0.0) & (alfven_data['orbit2alfven'] > 0.0)]

    return alfven_data

CLASS_COLORS = ['#DDBB44','#00AA00','#FF5500','#00FFFF','#000000']
CLASS_LABELS = ['subterran', 'terran', 'superterran', 'giant','']

def mass_class(pl_mass):
    # subterran - 0.1 to 0.5 ME 
    if np.fabs(pl_mass - 0.3) <= 0.2:
        return 0
    # terran - 0.5 to 3.0 ME
    elif np.fabs(pl_mass - 1.75) <= 1.25:
        return 1
    # superterran - 3.0 to 10. ME
    elif np.fabs(pl_mass - 6.5) <= 3.5:
        return 2
    # giant - > 10. ME
    elif pl_mass > 10.:
        return 3
    else:
        return -1

def rad_class(pl_rad):
    # subterran - 0.4 to 0.8 RE
    if np.fabs(pl_rad - 0.6) <= 0.2:
        return 0
    # terran - 0.8 to 1.6 RE
    elif np.fabs(pl_rad - 1.2) <= 0.4:
        return 1
    # superterran - 1.6 to 2.5 RE
    elif np.fabs(pl_rad - 2.05) <= 0.45:
        return 2
    # giant - > 2.5 RE
    elif pl_rad > 2.5:
        return 3
    else:
        return -1


def planet_classes(alfven_data):

    exos_habitable = pd.read_csv('current-exo-data/exos_habitable.csv')
    exos_hill23 = pd.read_csv('current-exo-data/exos_hill23.csv')

    # habitable_pl_names = set(exos_habitable.pl_name)
    habitable_pl_names = set(exos_habitable.pl_name).union(set(exos_hill23.pl_name))
    
    
    alfven_data['mass_class'] = alfven_data.apply(lambda r: mass_class(r.pl_bmasse), axis=1)
    alfven_data['rad_class'] = alfven_data.apply(lambda r: rad_class(r.pl_rade), axis=1)
    alfven_data['habitable'] = alfven_data.apply(lambda r: 1 if r.pl_name in habitable_pl_names else 0, axis=1)

    return alfven_data

def calculate_exos():
    nasa_exo = pd.read_csv('current-exo-data/nasa_exo.csv')
    nasa_exo = measured_uncertainties(nasa_exo)
    exos = pd.read_csv('current-exo-data/exos.csv')

    # List exoplanets with a host star that has a rotation period + stats
    prot_col_list = ['pl_name','hostname','Prot','e_Prot','pl_bmasse',
        'pl_bmasseerr','pl_bmassprov','pl_rade','pl_radeerr','pl_orbsmax',
        'pl_orbsmaxerr','pl_orbeccen','pl_orbeccenerr','sy_vmag','sy_vmagerr',
        'sy_kmag','sy_kmagerr','st_rad','st_raderr','st_mass','st_masserr',
        'sy_dist','sy_disterr','KOI','KIC','TIC','GAIA','db']
    prot_data = pd.merge(nasa_exo, exos, how='inner', on='hostname', suffixes=('_x', None))[prot_col_list]
    
    # Estimate Ro from Wright et. al. 2018 Tc formulae
    prot_data = estimate_rossby(prot_data)

    # List exoplanets that have all of the relevant stats: Ro, a, e 
    data_col_list = ['pl_name','pl_orbsmax','pl_orbsmaxerr','pl_orbeccen',
        'pl_orbeccenerr','pl_rade','pl_bmasse','pl_bmassprov','st_rad',
        'st_raderr','RoVK','RoM','RoAvg','Ro','dRo','KOI','KIC','TIC','GAIA','db']
    
    data = prot_data[prot_data['Ro'].notnull() & prot_data['pl_orbsmax'].notnull() & prot_data['pl_orbeccen'].notnull()]
    data = data[data_col_list].copy()

    # List exoplanets with periastron to mean AS ratio
    alfven_data = estimate_alfven(data)

    # Assign size and habitability classifications to each planet
    alfven_data = planet_classes(alfven_data)

    alfven_data.to_csv('current-exo-data/alfven_data.csv',index=False)

    return alfven_data

def main():

    if calculate:
        alfven_data = calculate_exos()
    else:
        alfven_data = pd.read_csv('current-exo-data/alfven_data.csv')

    print("[alfven_data]", alfven_data.count())
    print("[alfven_data: habitable]", alfven_data[alfven_data.habitable == 1].count())
    print("[alfven_data: mag habitable]", alfven_data[(alfven_data.orbit2alfven > 1.0) & (alfven_data.habitable == 1)].pl_name.count())

    CHZ_MHC_names = alfven_data[alfven_data.habitable == 1][['pl_name', 'Ro','orbit2alfven']]
    CHZ_MHC_names.to_csv('current-exo-data/CHZ_MHC_names.csv', index=False)

    # condition = alfven_data.habitable == 0
    # title = "Exoplanet Periastrons and RA Estimates"
    # imgname = 'current-exo-data/plot.png'
    # plot(alfven_data, condition, title, imgname, errorbars=False, names=False)
    # condition = alfven_data.habitable == 1
    # color_conditions = [alfven_data.mass_class == i for i in range(0,4)]
    # name_conditions = (alfven_data.habitable == 1)
    # title = "Magnetic Habitability Criterion of Goldilocks Exoplanets"
    # imgname = 'current-exo-data/plot_habitable.png'
    # plot(alfven_data, condition, color_conditions, title, imgname, errorbars=True, name_data=None)
    

if __name__ == '__main__':
    main()
    