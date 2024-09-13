import numpy as np
import pandas as pd


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
def taucVK(VK):
    x = np.power(10., AVK + BVK * VK)

    if isinstance(x, pd.Series):
        x.loc[np.fabs(VK - 4.05) > 2.95] = np.nan
        return x
    
    # VK is guaranteed positive given W18+ relationship domain
    if np.fabs(VK - 4.05) <= 2.95:
        return x
    else:
        return np.nan


def dtaucVK(VK, dVK):
    return taucVK(VK) * LN10 * np.sqrt(np.power(dAVK, 2.) + 
                                         np.power(VK * dBVK, 2.) + 
                                         np.power(BVK, 2.) *
                                         np.power(dVK, 2.))


def RoVK(Prot, VK):
    return np.fabs(Prot) / taucVK(VK)


def dRoVK(Prot, VK, dProt, dVK):
    return RoVK(Prot, VK) * np.sqrt(np.power(dProt / np.fabs(Prot), 2.) +
                                      np.power(
                                        dtaucVK(VK, dVK) /
                                        taucVK(VK), 2.))


# wright et al 2018 (eqn 6): valid for range 0.08 < M/Msol < 1.36
#M in solar masses
def taucM(M):
    x = np.power(10., AM + BM * M + CM * np.power(M, 2.))

    if isinstance(x, pd.Series):
        x.loc[np.fabs(M - 0.72) > 0.64] = np.nan
        return x

    if np.fabs(M - 0.72) <= 0.64:
        return x
    else:
        return np.nan


def dtaucM(M, dM):
    return taucM(M) * LN10 * np.sqrt(
        np.power(dAM, 2.) +
        np.power(M * dBM, 2.) +
        np.power(np.power(M, 2.) * dCM, 2.) +
        np.power((2 * M * CM + BM) * dM, 2.))
    

def RoM(Prot, M):
    return np.fabs(Prot) / taucM(M)


def dRoM(Prot, M, dProt, dM):
    return RoM(Prot, M) * np.sqrt(np.power(dProt / np.fabs(Prot), 2.) + 
                                  np.power(dtaucM(M, dM) / taucM(M), 2.))


def choose_Tauc(TaucVK: pd.Series, TaucM: pd.Series) -> pd.Series:
    
    where_rovk_notnull = TaucVK.notnull()
    where_rom_notnull = TaucM.notnull()
    
    Ro = np.nan + np.zeros_like(TaucVK)
    Ro[where_rom_notnull] = TaucM[where_rom_notnull]
    Ro[where_rovk_notnull] = TaucVK[where_rovk_notnull]

    return pd.Series(Ro, name="Tauc", index=TaucVK.index)


def choose_eTauc(TaucVK: pd.Series, TaucM: pd.Series, dTaucVK: pd.Series,
              dTaucM: pd.Series) -> pd.Series:
    where_taucvk_notnull = TaucVK.notnull()
    where_taucm_notnull = TaucM.notnull()
    
    dTauc = np.nan + np.zeros_like(TaucVK)
    dTauc[where_taucm_notnull] = dTaucM[where_taucm_notnull]
    dTauc[where_taucvk_notnull] = dTaucVK[where_taucvk_notnull]

    return pd.Series(dTauc, name="e_Tauc", index=TaucVK.index)


def choose_Ro(RoVK: pd.Series, RoM: pd.Series) -> pd.Series:
    
    where_rovk_notnull = RoVK.notnull()
    where_rom_notnull = RoM.notnull()
    
    Ro = np.nan + np.zeros_like(RoVK)
    Ro[where_rom_notnull] = RoM[where_rom_notnull]
    Ro[where_rovk_notnull] = RoVK[where_rovk_notnull]

    return pd.Series(Ro, name="Ro", index=RoVK.index)


def choose_eRo(RoVK: pd.Series, RoM: pd.Series, dRoVK: pd.Series,
              dRoM: pd.Series) -> pd.Series:
    where_rovk_notnull = RoVK.notnull()
    where_rom_notnull = RoM.notnull()
    
    dRo = np.nan + np.zeros_like(RoVK)
    dRo[where_rom_notnull] = dRoM[where_rom_notnull]
    dRo[where_rovk_notnull] = dRoVK[where_rovk_notnull]

    return pd.Series(dRo, name="e_Ro", index=RoVK.index)


r_sol = 6.957e8     # meters per solar radius
au = 1.496e11       # meters per au
prot_sol = 25.
ro_sol = 1.85
dro_sol = 0.26
ra_sol = 20 * r_sol / au

# ra_sol = 0.0451 # 9.7Rsun
# ra_sol = 0.1383 # 695700km in AU
dra_sol = 0.2 * ra_sol  # 20% error (cycle variation)
# r_sol = 1.
s = -1.38
ds = 0.14
r = -0.16
dr = 0.13
# r = r - dr
# r = r + dr
# all values scaled to solar values at maximum

# using Vidotto et al. 2014 p 6 for B_V
# get R_A from S03 eqn 13
# radius scaling (s = -1.38, B)
def ra_schrijver(ro):
    return ra_sol * np.real(np.power(ro / ro_sol, s * r))


def dra_schrijver(ro, dro):
    return ra_schrijver(ro) * np.sqrt(
        np.square(dra_sol / ra_sol) +
        np.square(s * r * dro / ro) +
        np.square(s * r * dro_sol / ro_sol) +
        np.square(r * np.log(ro / ro_sol) * ds) +
        np.square((s * np.log(ro / ro_sol)) * dr))

p_f21 = -1.74
dp_f21 = 0.0097
c_f21 = 28.4

def lx_farrish(ro):
    return p_f21 * np.log10(ro / ro_sol) + c_f21

def dlx_farrish(ro, dro):
    return lx_farrish(ro) * np.sqrt(
        np.square(dp_f21 * np.log10(ro / ro_sol)) +
        np.square(p_f21 * dro / ro / LN10)
    )

def measured_uncertainties(nasa_exo: pd.DataFrame) -> pd.DataFrame:
    x = nasa_exo
    nasa_exo["e_pl_bmasse"] = \
        np.max([x["pl_bmasseerr1"],np.fabs(x["pl_bmasseerr2"])], axis=0)
    nasa_exo["e_pl_bmassj"] = \
        np.max([x["pl_bmassjerr1"],np.fabs(x["pl_bmassjerr2"])], axis=0)
    nasa_exo["e_pl_orbsmax"] = \
        np.max([x["pl_orbsmaxerr1"],np.fabs(x["pl_orbsmaxerr2"])], axis=0)
    nasa_exo["e_pl_orbeccen"] = \
        np.max([x["pl_orbeccenerr1"],np.fabs(x["pl_orbeccenerr2"])], axis=0)
    nasa_exo["e_pl_rade"] = \
        np.max([x["pl_radeerr1"],np.fabs(x["pl_radeerr2"])], axis=0)
    nasa_exo["e_pl_radj"] = \
        np.max([x["pl_radjerr1"],np.fabs(x["pl_radjerr2"])], axis=0)
    nasa_exo["e_st_mass"] = \
        np.max([x["st_masserr1"],np.fabs(x["st_masserr2"])], axis=0)
    nasa_exo["e_st_rad"] = \
        np.max([x["st_raderr1"],np.fabs(x["st_raderr2"])], axis=0)
    nasa_exo["e_st_age"] = \
        np.max([x["st_ageerr1"],np.fabs(x["st_ageerr2"])], axis=0)
    nasa_exo["e_sy_dist"] = \
        np.max([x["sy_disterr1"],np.fabs(x["sy_disterr2"])], axis=0)
    nasa_exo["e_sy_kmag"] = \
        np.max([x["sy_kmagerr1"],np.fabs(x["sy_kmagerr2"])], axis=0)
    nasa_exo["e_sy_vmag"] = \
        np.max([x["sy_vmagerr1"],np.fabs(x["sy_vmagerr2"])], axis=0)
    
    return nasa_exo


def estimate_rossby(prot_data: pd.DataFrame) -> pd.DataFrame:
    x = prot_data
    prot_data["VK_color"] = \
        prot_data["sy_vmag"] - prot_data["sy_kmag"]
    prot_data["e_VK_color"] = \
        np.sqrt(np.square(x["e_sy_vmag"]) + np.square(x["e_sy_kmag"]))
    prot_data["RoVK"] = \
        RoVK(x["Prot"], x["VK_color"])
    prot_data["e_RoVK"] = \
        dRoVK(x["Prot"], x["VK_color"], x["e_Prot"], x["e_VK_color"])
    prot_data["TaucVK"] = \
        taucVK(x["VK_color"])
    prot_data["e_TaucVK"] = \
        dtaucVK(x["VK_color"], x["e_VK_color"])
    prot_data["RoM"] = \
        RoM(x["Prot"], x["st_mass"])
    prot_data["e_RoM"] = \
        dRoM(x["Prot"], x["st_mass"], x["e_Prot"], x["e_st_mass"])
    prot_data["TaucM"] = \
        taucM(x["st_mass"])
    prot_data["e_TaucM"] = \
        dtaucM(x["st_mass"], x["e_st_mass"])

    # selects TaucVK over TaucM if both present
    prot_data["Tauc"] = choose_Tauc(x["TaucVK"], x["TaucM"])
    prot_data["e_Tauc"] = choose_eTauc(x["TaucVK"], x["TaucM"], x["e_TaucVK"], x["e_TaucM"])

    # selects RoVK over RoM if both present
    prot_data["Ro"] = choose_Ro(x["RoVK"], x["RoM"])
    prot_data["e_Ro"] = choose_eRo(x["RoVK"], x["RoM"], x["e_RoVK"], x["e_RoM"])

    return prot_data


def estimate_alfven(data: pd.DataFrame) -> pd.DataFrame:
    
    # Estimate value and uncertainty for periastron
    x = data

    where_no_ecc = x["pl_orbeccen"].isnull()
    where_ecc = x["pl_orbeccen"].notnull()
    x.loc[where_no_ecc, "rperi"] = x.loc[where_no_ecc, "pl_orbsmax"]
    x.loc[where_no_ecc, "e_rperi"] = x.loc[where_no_ecc, "e_pl_orbsmax"]
    x.loc[where_ecc, "rperi"] = x.loc[where_ecc, "pl_orbsmax"] * (1 - x.loc[where_ecc, "pl_orbeccen"])
    x.loc[where_ecc, "e_rperi"] = np.sqrt(np.nansum([
        np.power(x.loc[where_ecc, "e_pl_orbsmax"] * (1 - x.loc[where_ecc, "pl_orbeccen"]), 2.),
        np.power(x.loc[where_ecc, "pl_orbsmax"] * x.loc[where_ecc, "e_pl_orbeccen"], 2.)], axis=0))
    
    ra = ra_schrijver(data["Ro"])
    dra = dra_schrijver(data["Ro"], data["e_Ro"] / ra)

    lx = lx_farrish(data["Ro"])
    dlx = dlx_farrish(data["Ro"], data["e_Ro"])

    ashc = data["rperi"] / ra
    dashc = data["rperi"] / ra * np.sqrt(
        np.power(data["e_rperi"] / data["rperi"], 2.) + \
        np.power(dra, 2.))

    alfven_data = data.copy(deep=False)
    alfven_data["RA"] = ra
    alfven_data["e_RA"] = dra
    alfven_data["RASun"] = ra * au / r_sol
    alfven_data["e_RASun"] = dra * au / r_sol
    alfven_data["LX"] = lx
    alfven_data["e_LX"] = dlx
    alfven_data["ASHC"] = ashc
    alfven_data["e_ASHC"] = dashc
    
    alfven_data = alfven_data[(alfven_data["Ro"] > 0.0) & (alfven_data["ASHC"] > 0.0)]

    return alfven_data


def mass_class(pl_mass):
    if isinstance(pl_mass, pd.Series):
        mclass = -1 + np.zeros_like(pl_mass, dtype=int)
        mclass[np.fabs(pl_mass - 0.3) <= 0.2] = 0
        mclass[np.fabs(pl_mass - 1.75) <= 1.25] = 1
        mclass[np.fabs(pl_mass - 6.5) <= 3.5] = 2
        mclass[pl_mass > 10.] = 3
        return pd.Series(mclass, name="mass_class", dtype=int, index=pl_mass.index)

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
    if isinstance(pl_rad, pd.Series):
        rclass = -1 + np.zeros_like(pl_rad, dtype=int)
        rclass[np.fabs(pl_rad - 0.6) <= 0.2] = 0
        rclass[np.fabs(pl_rad - 1.2) <= 0.4] = 1
        rclass[np.fabs(pl_rad - 2.05) <= 0.45] = 2
        rclass[pl_rad > 2.5] = 3
        return pd.Series(rclass, name="rad_class", dtype=int, index=pl_rad.index)

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


def size_class_subscripts(m, r, h, ASHC) -> pd.Series:
    s = ""
    if h == 1 and ASHC > 1:
        if m == 1:
            s += "a"
            if r == 1:
                s += ",b"
        elif r == 1:

            s += "b"
    return s


def planet_classes(alfven_data: pd.DataFrame) -> pd.DataFrame:

    alfven_data.reset_index(inplace=True)

    exos_habitable = pd.read_csv("tables-merged/exos_habitable.csv")
    exos_hill23 = pd.read_csv("tables-merged/exos_hill23.csv")

    habitable_pl_names = set(exos_habitable["pl_name"]).union(set(exos_hill23["pl_name"]))

    CHZOHZests = pd.merge(alfven_data, exos_hill23, how="left", on="pl_name")
    CHZOHZests.fillna(0, inplace=True)

    alfven_data["mass_class"] = mass_class(alfven_data["pl_bmasse"])
    alfven_data["rad_class"] = rad_class(alfven_data["pl_rade"])
    alfven_data["habitable"] = alfven_data.apply(lambda r: 1 if r["pl_name"] in habitable_pl_names else 0, axis=1)
    alfven_data["CHZ"] = CHZOHZests["CHZ"].astype(np.int8)
    alfven_data["OHZ"] = CHZOHZests["OHZ"].astype(np.int8)
    alfven_data["size_class"] = alfven_data.apply(lambda r: r["mass_class"] if r["mass_class"] != -1 else r["rad_class"], axis=1)
    alfven_data["n_size_class"] = alfven_data.apply(
        lambda r: size_class_subscripts(r["mass_class"], r["rad_class"], r["habitable"], r["ASHC"]), axis=1)

    return alfven_data


def calculate_exos() -> pd.DataFrame:
    nasa_exo = pd.read_csv("tables-merged/nasa_exo.csv")
    nasa_exo = measured_uncertainties(nasa_exo)
    stars_prot = pd.read_csv("tables-merged/hosts_prot.csv")

    # List exoplanets with a host star that has a rotation period + stats
    prot_col_list = ["pl_name","hostname", "pl_letter", "Prot","e_Prot",
        "pl_bmasse", "e_pl_bmasse", "pl_bmassprov", "pl_rade", "e_pl_rade",
        "pl_bmassj", "e_pl_bmassj", "pl_radj", "e_pl_rade",
        "pl_orbsmax", "e_pl_orbsmax", "pl_orbeccen", "e_pl_orbeccen",
        "sy_vmag", "e_sy_vmag", "sy_kmag", "e_sy_kmag", "st_rad",
        "e_st_rad", "st_mass", "e_st_mass", "sy_dist", "e_sy_dist", "KOI",
        "st_teff", "st_age", "e_st_age",
        "KIC", "TIC", "GAIA", "db"]
    prot_data = pd.merge(nasa_exo, stars_prot, how="inner", on="hostname",
                         suffixes=("_x", None))[prot_col_list]

    # Estimate Ro from Wright et. al. 2018 Tc formulae
    prot_data = estimate_rossby(prot_data)

    # List exoplanets that have all of the relevant stats: Ro, a, e 
    data_col_list = ["pl_name", "hostname", "pl_letter", "pl_orbsmax",
        "e_pl_orbsmax", "pl_orbeccen", "e_pl_orbeccen", "pl_rade", "e_pl_rade",
        "pl_bmassj", "e_pl_bmassj", "pl_radj", "e_pl_rade",
        "pl_bmasse", "e_pl_bmasse", "st_rad", "e_st_rad", "RoVK", "RoM",
        "Ro", "e_RoVK", "e_RoM", "e_Ro", "KOI", "KIC", "TIC", "GAIA", "db",
        "sy_dist", "e_sy_dist", "st_mass", "e_st_mass", "sy_vmag", "e_sy_vmag",
        "sy_kmag", "e_sy_kmag", "Prot","e_Prot", "VK_color", "e_VK_color",
        "st_teff", "st_age", "e_st_age", "Tauc", "e_Tauc", "db"]
    
    # Planets must have a calculable r_p - both a and e.
    where_data = prot_data["Ro"].notnull() & prot_data["pl_orbsmax"].notnull()
    data = prot_data.loc[where_data, data_col_list].copy()

    # List exoplanets with periastron to mean AS ratio
    alfven_data = estimate_alfven(data)

    # Assign size and habitability classifications to each planet
    alfven_data = planet_classes(alfven_data)

    alfven_data.rename(columns={"db": "r_Prot"}, inplace=True)

    alfven_data.to_csv("tables-merged/alfven_data.csv", index=False)
    return alfven_data


if __name__ == "__main__":

    print_brief = True

    alfven_data = calculate_exos()

    chz_data = alfven_data[alfven_data["habitable"] == 1]
    chz_ashc_data = alfven_data[(alfven_data["ASHC"] > 1.0) & (alfven_data["habitable"] == 1)]

    cols_print_long = ["pl_name", "mass_class", "rad_class", "st_mass",
                       "st_rad", "RoVK", "e_RoVK", "RoM", "e_RoM", "Ro", "e_Ro",
                       "ASHC", "e_ASHC"]
    cols_print_brief = ["pl_name", "Ro", "e_Ro", "ASHC", "e_ASHC"]

    if print_brief:
        cols_print = cols_print_brief
    else:
        cols_print = cols_print_long
    
    print("[alfven_data]\n{}\n".format(alfven_data[cols_print].count()))
    print("[chz]\n{}\n".format(chz_data[cols_print].count()))
    print("[chz & ashc]\n{}\n".format(chz_ashc_data[cols_print].count()))

    CHZ_names = chz_data[["pl_name", "Ro", "ASHC"]]
    CHZ_names.to_csv("tables-merged/CHZ_names.csv", index=False)

    CHZ_ASHC_names = chz_ashc_data[["pl_name", "Ro", "ASHC"]]
    CHZ_ASHC_names.to_csv("tables-merged/CHZ_ASHC_names.csv", index=False)
    