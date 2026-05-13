"""Fig. 2 — Example cascade tree.


Picks one test-set cascade and draws it as a directed graph: the target
user at the center, with their observed-prefix neighbors as predecessors
arranged radially by retweet time.


Run:
   PYTHONPATH=. python scripts/figures/fig2_cascade_tree.py
"""


import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


from scripts.figures._utils import load_model_and_test, save, setup_matplotlib




def pick_cascade(test_ds, min_size: int = 8, max_size: int = 20):
   """Find a positive cascade with a non-trivial neighbor count."""
   for i in range(len(test_ds)):
       s = test_ds[i]
       if float(s["activated"]) < 0.5:
           continue
       nbrs = [int(u) for u in s["neighbor_nodes"].tolist() if int(u) != 0]
       if min_size <= len(nbrs) <= max_size:
           return i, s
   # Fallback: relax the size constraint and just take any positive
   for i in range(len(test_ds)):
       s = test_ds[i]
       if float(s["activated"]) >= 0.5:
           return i, s
   return 0, test_ds[0]




def main():
   setup_matplotlib()
   _, _, test_ds, _ = load_model_and_test()


   idx, sample = pick_cascade(test_ds)
   target = int(sample["target_nodes"])
   nb_ids = sample["neighbor_nodes"].tolist()
   nb_times = sample["neighbor_times"].tolist()
   neighbors = [int(u) for u in nb_ids if int(u) != 0]
   times = [float(t) for u, t in zip(nb_ids, nb_times) if int(u) != 0]


   # Build directed graph: each neighbor -> target
   G = nx.DiGraph()
   G.add_node(target, kind="target")
   for n in neighbors:
       G.add_node(n, kind="neighbor")
       G.add_edge(n, target)


   # Layout: target at origin, neighbors on a circle (angle = chronological order)
   pos = {target: (0.0, 0.0)}
   if neighbors:
       order = np.argsort(times)
       for rank, j in enumerate(order):
           angle = 2 * np.pi * rank / len(neighbors)
           pos[neighbors[j]] = (np.cos(angle), np.sin(angle))


   fig, ax = plt.subplots(figsize=(6, 6))


   # Edges
   nx.draw_networkx_edges(
       G, pos, ax=ax,
       edge_color="#999",
       arrows=True, arrowsize=14, width=1.2,
       connectionstyle="arc3,rad=0.05",
   )


   # Neighbor nodes — colour by time (earlier = darker)
   if neighbors:
       t_min, t_max = min(times), max(times)
       t_range = max(t_max - t_min, 1e-6)
       n_colors = [(t - t_min) / t_range for t in times]
       nx.draw_networkx_nodes(
           G, pos, nodelist=neighbors, ax=ax,
           node_color=n_colors, cmap="viridis",
           node_size=380, edgecolors="#222", linewidths=1.0,
       )


   # Target node
   nx.draw_networkx_nodes(
       G, pos, nodelist=[target], ax=ax,
       node_color="#d33", node_size=820,
       edgecolors="#222", linewidths=1.5,
   )


   labels = {target: f"target\n#{target}"}
   nx.draw_networkx_labels(G, pos, labels=labels, ax=ax,
                           font_size=9, font_color="white",
                           font_weight="bold")


   ax.set_title(
       f"Example cascade #{idx} — target user activated\n"
       f"({len(neighbors)} observed-prefix retweeters)",
       pad=10,
   )
   ax.set_axis_off()


   # Colourbar for time
   if neighbors:
       sm = plt.cm.ScalarMappable(
           cmap="viridis",
           norm=plt.Normalize(vmin=t_min, vmax=t_max),
       )
       sm.set_array([])
       cbar = plt.colorbar(sm, ax=ax, fraction=0.04, pad=0.04)
       cbar.set_label("retweet time (hours)", fontsize=9)


   save(fig, "fig2_cascade_tree")
   plt.close(fig)




if __name__ == "__main__":
   main()




