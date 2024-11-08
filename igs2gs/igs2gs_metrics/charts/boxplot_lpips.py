import matplotlib.pyplot as plt
import numpy as np


baseline = np.genfromtxt(
    "/media/lucky/486d4773-81cb-4c30-ae5f-8cd74b05a68a/Lucky_Thesis_Data/eval_lpips/10-22-15-02_Ephra_turn-him-into-Tolkien-Elf_42_5.0_0.5_2.0_0.2/30001_lpips.csv",
    delimiter=",",
)
low = np.genfromtxt(
    "/media/lucky/486d4773-81cb-4c30-ae5f-8cd74b05a68a/Lucky_Thesis_Data/eval_lpips/10-22-15-02_Ephra_turn-him-into-Tolkien-Elf_42_5.0_0.5_2.0_0.2/35000_lpips.csv",
    delimiter=",",
)

high = np.genfromtxt(
    "/media/lucky/486d4773-81cb-4c30-ae5f-8cd74b05a68a/Lucky_Thesis_Data/eval_lpips/10-29-12-17_Ephra_Make-this-person-look-like-Tolkien-Elf_42_7.5_0.5_2.0_0.2/35000_lpips.csv",
    delimiter=",",
)
# Combine data1 and data2 into a list
data = [baseline, low, high]

# Create the box plot
plt.boxplot(data, labels=["Unstylized", "Low Sp", "High Sp"])

# Add title and labels
plt.title("Model A - P2")
plt.ylabel("Masked LPIPS Score")

# Display the plot
plt.show()
