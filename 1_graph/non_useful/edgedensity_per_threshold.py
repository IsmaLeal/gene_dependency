from graph_functions import prep_graph
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from multiprocessing import Pool, Manager


def density(threshold, progress_list):
    g, _ = prep_graph(threshold)
    d = g.num_edges() / (g.num_vertices() * (g.num_vertices() - 1) / 2)
    progress_list.append(1)
    return d


thresholds = np.arange(0, 1, 0.02)

with Manager() as manager:
    progress_list = manager.list()
    with Pool(processes=40) as pool:
        results = [pool.apply_async(density, args=(threshold, progress_list)) for threshold in thresholds]
        with tqdm(total=len(thresholds)) as pbar:
            while len(progress_list) < len(thresholds):
                pbar.update(len(progress_list) - pbar.n)
                pbar.refresh()

densities = [result.get() for result in results]

plt.plot(thresholds, densities)
plt.title("Edge density vs threshold")
plt.xlabel('Threshold')
plt.ylabel('Edge density')
plt.grid(True)
plt.tight_layout()
plt.show()