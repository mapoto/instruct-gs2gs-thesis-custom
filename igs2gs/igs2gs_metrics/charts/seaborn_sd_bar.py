import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from pathlib import Path

metric_type = ["ciede2000", "rmse", "lpips"]
human_models = ["dora", "ephra", "irene", "simon"]
style = ["rabbit", "elf", "stone", "red", "cowboy", "vangogh", "3d", "smile"]

dict_h = {"dora": "A", "ephra": "B", "irene": "C", "simon": "D"}
dict_s = {
    "rabbit": "P1",
    "elf": "P2",
    "stone": "P3",
    "red": "P4",
    "cowboy": "P5",
    "vangogh": "P6",
    "3d": "P7",
    "smile": "P8",
}


path = Path("igs2gs/igs2gs_metrics/charts")

for m in metric_type:
    with open(path / f"sd_{m}.csv", "a+") as f:
        f.write("conf,average,sd\n")

    for h in human_models:
        data = []
        # Unstylized
        data1 = np.genfromtxt(
            f"/media/lucky/486d4773-81cb-4c30-ae5f-8cd74b05a68a/Lucky_Thesis_Data/eval_lpips/low/{h}_elf/30000_{m}.csv",
            delimiter=",",
        )
        # compute standard deviation
        sd1 = np.std(data1)
        average1 = np.mean(data1)

        data.append(data1)
        df = pd.DataFrame({m: data1, "sP": "unstylized", "P": "unstylized"})
        with open(path / f"sd_{m}.csv", "a+") as f:
            f.write(f"{dict_h[h]}+unstylized,{average1}, {sd1}\n")

        for s in style:

            # Low Sp
            data2 = np.genfromtxt(
                f"/media/lucky/486d4773-81cb-4c30-ae5f-8cd74b05a68a/Lucky_Thesis_Data/eval_lpips/low/{h}_{s}/35000_{m}.csv",
                delimiter=",",
            )
            sd2 = np.std(data2)
            average2 = np.mean(data2)
            df_low = pd.DataFrame({m: data2, "sP": "low", "P": dict_s[s]})
            df = pd.concat([df, df_low])

            # High Sp
            data3 = np.genfromtxt(
                f"/media/lucky/486d4773-81cb-4c30-ae5f-8cd74b05a68a/Lucky_Thesis_Data/eval_lpips/high/{h}_{s}/35000_{m}.csv",
                delimiter=",",
            )
            sd3 = np.std(data3)
            average3 = np.mean(data3)
            df_high = pd.DataFrame({m: data3, "sP": "high", "P": dict_s[s]})
            df = pd.concat([df, df_high])

            with open(path / f"sd_{m}.csv", "a+") as f:
                f.write(f"{str(dict_h[h]+dict_s[s])}+low,{average2},{sd2}\n")
                f.write(f"{str(dict_h[h]+dict_s[s])}+high, {average3},{sd3}\n")

        # sns.barplot(data=df, x=m, y="P", hue="sP", errorbar="sd", gap=0.5)
        # plt.title(f"Standard Deviation of {m.upper()} Human Model: {dict_h[h]}")
        # plt.xlabel(f"{m.upper()}")
        # plt.ylabel("Stylization configurations")
        # # Save the plot
        # plt.savefig(f"igs2gs/igs2gs_metrics/charts/sd_{m}_{h}.png")
        # plt.clf()
