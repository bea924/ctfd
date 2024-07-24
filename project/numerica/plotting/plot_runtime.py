import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 0 exact
# 1 lax
# 2 roe
solver_list = [0, 1, 2]
output_time = 0.2


fig, axes = plt.subplots(1, 1, figsize=(8, 6))
solver_dict = {
    0: 'Exact',
    1: "Lax",
    2: 'Roe'
}

for i, solver in enumerate(solver_list):
    # Read the .out file
    file_path = f'output/solver{solver}_t{output_time}_stats.out'
    # file_path = f'project/numerica/output/solver{solver}_t{output_time}_stats.out' # for debugger
    runtime = pd.read_csv(file_path, delim_whitespace=True, header=None)

    runtime = runtime[0].to_numpy()
    x = np.arange(0,len(runtime))

    # Plot each column in a separate subplot
    axes.plot(x, runtime, label=solver_dict[i])
    # axes.set_title(f'Plot of {solver_dict[i]}')
    axes.legend()
    axes.grid(True)


axes.set_title(f'Plot of runtime')
# Adjust layout
plt.tight_layout()

# Show the plots
plt.show()