# This file is part of the Metron AI DeepForge (https://github.com/OndrejSzekely/metron_ai_deepforge).
# Copyright (c) 2025 Ondrej Szekely (ondra.szekely@gmail.com).

import matplotlib.pyplot as plt

from usecases.assignments.samplinghuman.utils.net_utils import sinusoidal_positional_encoding

cell_num = 2
cells_total = cell_num * cell_num
d_model = 64

pe = sinusoidal_positional_encoding(d_model, cell_num)
pe = pe.view(cells_total, d_model // cells_total).numpy()

plt.figure(figsize=(12, 8))
plt.imshow(pe, cmap="coolwarm", aspect="auto", vmin=-1, vmax=1)
plt.colorbar()
plt.title("Sinusoidal Positional Encoding")
plt.xlabel("Dimension")
plt.ylabel("Position")
plt.tight_layout()
plt.savefig("positional_encoding.png")
