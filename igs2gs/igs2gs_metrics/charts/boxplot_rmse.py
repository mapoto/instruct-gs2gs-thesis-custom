import matplotlib.pyplot as plt
import numpy as np


data1 = np.genfromtxt(
    "/media/lucky/486d4773-81cb-4c30-ae5f-8cd74b05a68a/Lucky_Thesis_Data/eval_lpips/10-25-19-32_Ephra_turn-him-into-a-rabbit_42_10.0_0.5_2.0_0.2/30000_rmse_filter.csv",
    delimiter=",",
)
data1 = data1.reshape(-1)
# data1_line = np.genfromtxt(
#     "/media/lucky/486d4773-81cb-4c30-ae5f-8cd74b05a68a/Lucky_Thesis_Data/eval_lpips/10-22-15-02_Ephra_turn-him-into-Tolkien-Elf_42_5.0_0.5_2.0_0.2/30000/lpips_line.csv",
#     delimiter=",",
# )

# data1_line = data1_line.reshape(-1)
# data1 = np.concatenate((data1, data1_line))

data2 = np.genfromtxt(
    "/media/lucky/486d4773-81cb-4c30-ae5f-8cd74b05a68a/Lucky_Thesis_Data/eval_lpips/10-25-19-32_Ephra_turn-him-into-a-rabbit_42_10.0_0.5_2.0_0.2/30000_rmse_masked.csv",
    delimiter=",",
)
data2 = data2.reshape(-1)
# data2_line = np.genfromtxt(
#     "/media/lucky/486d4773-81cb-4c30-ae5f-8cd74b05a68a/Lucky_Thesis_Data/eval_lpips/10-22-15-02_Ephra_turn-him-into-Tolkien-Elf_42_5.0_0.5_2.0_0.2/35000/lpips_line.csv",
#     delimiter=",",
# )
# data2_line = data2_line.reshape(-1)
# data2 = np.concatenate((data2, data2_line))

data3 = np.genfromtxt(
    "/media/lucky/486d4773-81cb-4c30-ae5f-8cd74b05a68a/Lucky_Thesis_Data/eval_lpips/10-25-19-32_Ephra_turn-him-into-a-rabbit_42_10.0_0.5_2.0_0.2/35000_rmse_filter.csv",
    delimiter=",",
)
data3 = data3.reshape(-1)

data4 = np.genfromtxt(
    "/media/lucky/486d4773-81cb-4c30-ae5f-8cd74b05a68a/Lucky_Thesis_Data/eval_lpips/10-25-19-32_Ephra_turn-him-into-a-rabbit_42_10.0_0.5_2.0_0.2/35000_rmse_masked.csv",
    delimiter=",",
)
data4 = data4.reshape(-1)

# Combine data1 and data2 into a list
data = [data1, data2, data3, data4]

# Create the box plot
plt.boxplot(data, labels=["30000 filtered", "30000 masked", "35000 filtered", "35000 masked"])

# Add title and labels
plt.title("Rabbit")
plt.ylabel("Values")

# Display the plot
plt.show()
