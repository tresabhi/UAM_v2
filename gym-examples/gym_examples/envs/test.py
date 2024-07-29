import matplotlib.pyplot as plt

fig, ax = plt.subplots()  # Creates a figure and a set of subplots
ax.plot([0, 1], [0, 1])  # Plot a line from (0,0) to (1,1)

print(type(fig), type(ax))
