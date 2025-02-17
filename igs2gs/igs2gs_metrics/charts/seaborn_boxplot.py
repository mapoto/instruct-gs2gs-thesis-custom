import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

metric_type = ["ciede2000", "rmse", "lpips"]
human_models = ["ephra", "dora", "irene", "simon"]
style = ["elf", "stone", "red", "vangogh", "3d"]

dict_h = {"ephra": "B", "dora": "A", "irene": "C", "simon": "D"}
dict_s = {"rabbit": "P1", "elf": "P2", "stone": "P3", "red": "P4", "cowboy": "P5", "vangogh": "P6", "3d": "P7"}

for m in metric_type:
    for h in human_models:
        data = []
        # Unstylized
        data1 = np.genfromtxt(
            f"/media/lucky/486d4773-81cb-4c30-ae5f-8cd74b05a68a/Lucky_Thesis_Data/eval_lpips/low/{h}_elf/30000_{m}.csv",
            delimiter=",",
        )
        data.append(data1)
        df = pd.DataFrame({m: data1, "sP": "unstylized", "P": "unstylized"})

        for s in style:
            # Low Sp
            data2 = np.genfromtxt(
                f"/media/lucky/486d4773-81cb-4c30-ae5f-8cd74b05a68a/Lucky_Thesis_Data/eval_lpips/low/{h}_{s}/35000_{m}.csv",
                delimiter=",",
            )
            df_low = pd.DataFrame({m: data2, "sP": "low", "P": dict_s[s]})
            df = pd.concat([df, df_low])

            # High Sp
            data3 = np.genfromtxt(
                f"/media/lucky/486d4773-81cb-4c30-ae5f-8cd74b05a68a/Lucky_Thesis_Data/eval_lpips/high/{h}_{s}/35000_{m}.csv",
                delimiter=",",
            )
            df_high = pd.DataFrame({m: data3, "sP": "high", "P": dict_s[s]})
            df = pd.concat([df, df_high])

        sns.boxplot(data=df, x=m, y="P", hue="sP", fill=False, gap=0.5)
        plt.title(f"MVS consistency using {m.upper()} Human Model: {dict_h[h]}")
        plt.xlabel(f"{m.upper()}")
        plt.ylabel("Stylization configurations")
        # Save the plot
        plt.savefig(f"igs2gs/igs2gs_metrics/charts/{m}_{h}.png")
        plt.clf()
