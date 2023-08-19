import numpy as np
import pandas as pd
from alfven_estimates import estimate_alfven, estimate_rossby, mass_class, measured_uncertainties, rad_class

if __name__ == "__main__":

    full_sample_path = "current-exo-data/nasa_exo.csv"
    full_sample = pd.read_csv(full_sample_path)
    full_sample = measured_uncertainties(full_sample)
    full_sample["vk"] = full_sample["sy_vmag"] - full_sample["sy_kmag"]
    full_sample["mass_class"] = mass_class(full_sample["pl_bmasse"])
    full_sample["rad_class"] = rad_class(full_sample["pl_rade"])

    # pl data
    m = full_sample["st_mass"]
    vk = full_sample["vk"]# CHZ PLANET CUT

    "db" in full_sample.columns

    # chz flag
    exos_habitable = pd.read_csv("current-exo-data/exos_habitable.csv")
    exos_hill23 = pd.read_csv("current-exo-data/exos_hill23.csv")

    # habitable_pl_names = set(exos_habitable.pl_name)
    habitable_pl_names = set(exos_habitable["pl_name"]).union(set(exos_hill23["pl_name"]))
    full_sample["habitable"] = full_sample.apply(lambda r: 1 if r["pl_name"] in habitable_pl_names else 0, axis=1)

    print("CHZ - [{}] UPR\nCHZ - [{}] Hill23+\nCHZ - [{}] All".format(len(exos_habitable), len(exos_hill23), len(habitable_pl_names)))

    # Planet Class
    mclass = full_sample["mass_class"]
    rclass = full_sample["rad_class"]

    print("MASS CLASS")
    print("EXOS - [{}] No class".format(mclass[mclass == -1].count()))
    print("EXOS - [{}] Subterran (0.1 <= M <= 0.5)".format(mclass[mclass == 0].count()))
    print("EXOS - [{}] Terran (0.5 < M <= 3.0)".format(mclass[mclass == 1].count()))
    print("EXOS - [{}] Superterran (3.0 < M <= 10.0)".format(mclass[mclass == 2].count()))
    print("EXOS - [{}] Giant (M > 10.0)".format(mclass[mclass == 3].count()))

    print("RADIUS CLASS")
    print("EXOS - [{}] No class".format(rclass[rclass == -1].count()))
    print("EXOS - [{}] Subterran (0.4 <= R <= 0.8)".format(rclass[rclass == 0].count()))
    print("EXOS - [{}] Terran (0.8 < R <= 1.6)".format(rclass[rclass == 1].count()))
    print("EXOS - [{}] Superterran (1.6 < R <= 2.5)".format(rclass[rclass == 2].count()))
    print("EXOS - [{}] Giant (R > 2.5)".format(rclass[rclass == 3].count()))

    # ORBITAL DATA CUT

    print("EXOS - [{}] semi-major axis (a)\nEXOS - [{}] eccentricity (e)".format(
        full_sample["pl_orbsmax"].count(),
        full_sample["pl_orbeccen"].count()))
    print("EXOS - [{}] uncertainty in semi-major axis (da)\nEXOS - [{}] uncertainty in eccentricity (de)".format(
        full_sample["pl_orbsmaxerr"].count(),
        full_sample["pl_orbeccenerr"].count()))

    # STELLAR/INTERIOR CONVECTION TYPE CUT

    # planets with stellar data to calculate tauc
    print("EXOS - [{}] M*\nEXOS - [{}] V-K".format(m.count(), vk.count()))
    # planets with stellar data satisfying partially- or fully-convective stellar interior constraint from W18+ tauc
    print("EXOS - [{}] 0.08 <= M* <= 1.36".format(m[np.fabs(m - 0.72) <= 0.64].count()))
    print("EXOS - [{}] 1.1 <= V-K <= 7.0".format(vk[np.fabs(vk - 4.05) <= 2.95].count()))

    # st data
    full_sample_st = full_sample.drop_duplicates(subset="hostname", keep="first").drop(["pl_name", "pl_letter"], axis="columns")

    m_st = full_sample_st["st_mass"]
    vk_st = full_sample_st["vk"]

    assert len(pd.unique(full_sample["hostname"])) == len(full_sample_st)
    print("HOSTS - [{}]".format(len(full_sample_st)))

    # stars with calculable tauc
    print("HOSTS - [{}] M* \nHOSTS - [{}] V-K".format(m_st.count(), vk_st.count()))
    # stars satisfying partially- or fully-convective stellar interior constraint from W18+ tauc
    print("HOSTS - [{}] 0.08 <= M* <= 1.36".format(m_st[np.fabs(m_st - 0.72) <= 0.64].count()))
    print("HOSTS - [{}] 1.1 <= V-K <= 7.0".format(vk_st[np.fabs(vk_st - 4.05) <= 2.95].count()))

    # ROTATION PERIOD CUT

    hosts_path = "current-exo-data/hosts_prot.csv"
    hosts = pd.read_csv(hosts_path)
    prot_sample = pd.merge(full_sample, hosts, on="hostname")
    prot_sample_st = pd.merge(full_sample_st, hosts, on="hostname")

    assert hosts["Prot"].count() == prot_sample_st["Prot"].count()
    assert hosts["e_Prot"].count() == prot_sample_st["e_Prot"].count()

    print(hosts.groupby(by="db")["Prot"].count())

    print("HOSTS (R) - [{}] Prot\nHOSTS (R) - [{}] e_Prot".format(
        prot_sample_st["Prot"].count(),
        prot_sample_st["e_Prot"].count()))
    print("EXOS (R) - [{}] Prot\nEXOS (R)- [{}] e_Prot".format(
        prot_sample["Prot"].count(),
        prot_sample["e_Prot"].count()))

    # STELLAR/INTERIOR CONVECTION TYPE CUT (ROTATORS)

    print("HOSTS (R) - [{}] V-K\nHOSTS (R)- [{}] M*".format(
        prot_sample_st["vk"].count(),
        prot_sample_st["st_mass"].count()))
    print("EXOS (R) - [{}] V-K\nEXOS (R) - [{}] M*".format(
        prot_sample["vk"].count(),
        prot_sample["st_mass"].count()))

    print("HOSTS (R) - [{}] dV\nHOSTS (R) - [{}] dK\nHOSTS (R) - [{}] dM*".format(
        prot_sample_st["sy_vmagerr"].count(),
        prot_sample_st["sy_kmagerr"].count(),
        prot_sample_st["st_masserr"].count()))
    print("EXOS (R) - [{}] dV\nEXOS (R) - [{}] dK\nEXOS (R) - [{}] dM*".format(
        prot_sample["sy_vmagerr"].count(),
        prot_sample["sy_kmagerr"].count(),
        prot_sample["st_masserr"].count()))

    where_err = (prot_sample["sy_vmagerr"].notnull() | prot_sample["sy_kmagerr"].notnull()) ^ \
        (prot_sample["sy_vmagerr"].notnull() & prot_sample["sy_kmagerr"].notnull())
    where_err_st = (prot_sample_st["sy_vmagerr"].notnull() | prot_sample_st["sy_kmagerr"].notnull()) ^ \
        (prot_sample_st["sy_vmagerr"].notnull() & prot_sample_st["sy_kmagerr"].notnull())

    print("HOSTS")
    print(prot_sample_st.loc[where_err_st, ["sy_vmagerr", "sy_kmagerr"]].count())
    print("EXOS")
    print(prot_sample.loc[where_err, ["sy_vmagerr", "sy_kmagerr"]].count())

    # ORBITAL DATA CUT (ROTATORS)
    print("EXOS (R) - [{}] semi-major axis (a)\nEXOS (R) - [{}] eccentricity (e)".format(
        prot_sample["pl_orbsmax"].count(),
        prot_sample["pl_orbeccen"].count()))
    print("EXOS (R) - [{}] uncertainty in semi-major axis (da)\nEXOS (R) - [{}] uncertainty in eccentricity (de)".format(
        prot_sample["pl_orbsmaxerr"].count(),
        prot_sample["pl_orbeccenerr"].count()))

    ro_sample = estimate_rossby(prot_sample)
    ro_sample_st = estimate_rossby(prot_sample_st)

    as_sample = estimate_alfven(ro_sample)
    print("EXOS (R) - [{}] MHC".format(
          as_sample[as_sample["MHC"].notnull()]["MHC"].count()))

    where_CHZ = as_sample["habitable"] == 1
    where_MHC = as_sample["MHC"] > 1

    CHZ_sample = as_sample[where_CHZ]
    MHC_sample = as_sample[where_MHC]
    CHZ_MHC_sample = as_sample[where_CHZ & where_MHC]

    print("[CHZ]\n",
          CHZ_sample[["pl_name", "MHC", "dMHC"]].count())
    print("[MHC>1]\n",
          MHC_sample[["pl_name", "MHC", "dMHC"]].count())
    print("[CHZ, MHC>1]\n",
          CHZ_MHC_sample[["pl_name", "MHC", "dMHC"]].count())

    print("[CHZ]\n",
        CHZ_sample.groupby(by="mass_class")[["MHC", "dMHC"]].count())
    print("\n\n[MHC>1]\n",
        MHC_sample.groupby(by="mass_class")[["MHC", "dMHC"]].count())
    print("\n\n[CHZ, MHC>1]\n",
        CHZ_MHC_sample.groupby(by="mass_class")[["MHC", "dMHC"]].count())

    print("[CHZ]\n",
        CHZ_sample.groupby(by="rad_class")[["MHC", "dMHC"]].count())
    print("\n\n[MHC>1]\n",
        MHC_sample.groupby(by="rad_class")[["MHC", "dMHC"]].count())
    print("\n\n[CHZ, MHC>1]\n",
        CHZ_MHC_sample.groupby(by="rad_class")[["MHC", "dMHC"]].count())
