import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


df = pd.read_csv("igs2gs/igs2gs_metrics/charts/qualitative_A.csv", header=0)
print(df)


def calculate_position(prompt, sp):
    # Calculate value (and round it)
    y_position = round(df[sp][prompt])
    # Determine x_position depending on the sp
    if sp == "low":
        x_position = 1 - 1.2
    else:
        x_position = 1 + 0.12

    return (prompt, x_position, y_position)


def add_label(x_position, y_position, prompt, value):

    # Adding the text
    plt.text(
        x_position,  # x-axis position
        y_position,  # y-axis position
        f"{prompt}, {value}",  # Text
        fontsize=8,  # Text size
        color="black",  # Text color
    )


def adjust_y():
    positions = []

    # Add label of each continent at each sp
    for item in df.index:
        for sp in df.columns:

            (p, x, y) = calculate_position(item, sp)
            positions.append((p, x, y))

    new_positions = []

    # Iterate through the list of positions
    for i, position in enumerate(positions):
        text, x, y = position
        adjustment = 0
        value = y

        # Check for identical x, y in the rest of the list
        for j in range(i + 1, len(positions)):
            if positions[j][1] == x and positions[j][2] == y:
                adjustment += 0.05

    y += adjustment
    new_positions.append((text, x, y, value))

    return new_positions


# Filter data for the sps high and low
sp = ["low", "high"]
df = df[df["SP"].isin(sp)]

# Calculate average gdp, per continent, per sp
df = df.groupby(["P", "SP"])["Quality"].mean().unstack()


# Set figsize
plt.figure(figsize=(6, 8))

# Vertical lines for the sps
plt.axvline(x=sp[0], color="black", linestyle="--", linewidth=1)  # 1952
plt.axvline(x=sp[1], color="black", linestyle="--", linewidth=1)  # 1957


# Plot the line for each continent
for prompt in df.index:

    # Color depending on the evolution
    value_before = df[df.index == prompt][sp[0]][0]  # gdp/cap of the continent in 1952
    value_after = df[df.index == prompt][sp[1]][0]  # gdp/cap of the continent in 1957

    # Red if the value has decreased, green otherwise
    if value_before > value_after:
        color = "red"
    else:
        color = "green"

    # Add the line to the plot
    plt.plot(sp, df.loc[prompt], marker="o", label=prompt, color=color)

new_positions = adjust_y()

# Print the new positions
print(new_positions)

for item in new_positions:

    # Add the label
    add_label(item[1], item[2], item[0], item[3])

# Add a title ('\n' allow us to jump lines)
plt.title(f"Slope Chart: \nComparing GDP Per Capita between {sp[0]} vs {sp[1]}  \n\n\n")

plt.yticks([])  # Remove y-axis
plt.box(False)  # Remove the bounding box around plot
plt.show()  # Display the chart
