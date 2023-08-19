import pandas as pd

def print_table(pl_name, Ro, dRo, MHC, dMHC, num):
    
    str_Ro = f"${Ro:05f}"
    if dRo is not None:
        str_Ro += f"\pm {dRo:05f}"
    str_Ro += "$"

    str_MHC = f"${MHC:05f}"
    if dMHC is not None:
        str_MHC += f"\pm {dMHC:05f}"
    str_MHC += "$"

    if num is not None:
        str_num = str(num)
    else:
        str_num = "--"
    
    print(f"{pl_name} & {str_Ro} & {str_MHC} & {str_num}\\\\")


if __name__ == "__main__":

    data_path = "current-exo-data/alfven_data.csv"
    df = pd.read_csv(data_path)

    good_pls = df[(df["habitable"] == 1) & (df["MHC"] > 1)]
    terrans = good_pls[good_pls["mass_class"] == 1].sort_values(by="MHC")
    superterrans = good_pls[good_pls["mass_class"] == 2].sort_values(by="MHC")

    print(" --- TERRAN --- ")
    for num, (_, row) in enumerate(terrans.iterrows()):
        print_table(row["pl_name"], row["Ro"], row["dRo"], row["MHC"], row["dMHC"], num + 1)

    print(" --- SUPERTERRAN --- ")
    for _, row in superterrans.iterrows():
        print_table(row["pl_name"], row["Ro"], row["dRo"], row["MHC"], row["dMHC"], None)