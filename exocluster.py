import stats.knn as knn
import numpy as np
import pandas as pd

if __name__ == "__main__":
    df = pd.read_csv("current-exo-data/alfven_data.csv")
    cols = ["st_mass", "VK_color", "st_age", "Prot"]
    temp = df[cols]
    logic = np.prod([np.array(temp[col].notnull()) for col in cols], axis=0)
    data = df.loc[logic, cols].to_numpy(dtype=np.float64)
    print(data.shape)