"""
Projet Graphes & OpenData - Réseau de bus Île-de-France
========================================================
Ce script charge les données GTFS, construit le graphe,
et génère toutes les figures nécessaires au rapport LaTeX.

Données : https://data.iledefrance-mobilites.fr
  -> Télécharger "IDFM-gtfs.zip" et dézipper dans ./data/

Dépendances : pip install networkx pandas matplotlib folium numpy
"""
from scipy.spatial import cKDTree
import os
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")  # Sans interface graphique
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from collections import Counter

# ─────────────────────────────────────────────
# 0. Configuration
# ─────────────────────────────────────────────
DATA_DIR    = "./data"          # Dossier contenant les fichiers GTFS
OUTPUT_DIR  = "./figures"       # Dossier de sortie des figures
os.makedirs(OUTPUT_DIR, exist_ok=True)

ROUTE_TYPE_BUS = 3             # Code GTFS pour les bus
N_SAMPLE_CENTRALITY = 500      # Échantillon pour betweenness (performance)


# ─────────────────────────────────────────────
# 1. Chargement des données GTFS
# ─────────────────────────────────────────────
def load_gtfs(data_dir):
    print("Chargement des données GTFS...")

    stops = pd.read_csv(f"{data_dir}/stops.txt", dtype=str)
    stops["stop_lat"] = stops["stop_lat"].astype(float)
    stops["stop_lon"] = stops["stop_lon"].astype(float)

    routes = pd.read_csv(f"{data_dir}/routes.txt", dtype=str)
    trips  = pd.read_csv(f"{data_dir}/trips.txt",  dtype=str)

    # Filtrer uniquement les bus
    bus_routes = routes[routes["route_type"] == str(ROUTE_TYPE_BUS)]
    bus_trips  = trips[trips["route_id"].isin(bus_routes["route_id"])]

    # stop_times : chargement par chunks pour économiser la RAM
    chunks = []
    for chunk in pd.read_csv(f"{data_dir}/stop_times.txt",
                             dtype=str, chunksize=500_000):
        chunk = chunk[chunk["trip_id"].isin(bus_trips["trip_id"])]
        chunks.append(chunk)
    stop_times = pd.concat(chunks, ignore_index=True)
    stop_times["stop_sequence"] = stop_times["stop_sequence"].astype(int)

    print(f"  {len(stops)} arrêts | {len(bus_routes)} lignes bus | "
          f"{len(stop_times)} horaires")
    return stops, bus_trips, stop_times


def parse_time(t):
    """Convertit '25:30:00' en secondes (gère les heures > 24h)."""
    h, m, s = t.strip().split(":")
    return int(h) * 3600 + int(m) * 60 + int(s)


# ─────────────────────────────────────────────
# 2. Construction du graphe
# ─────────────────────────────────────────────
def build_graph(stops, bus_trips, stop_times):
    print("Construction du graphe...")

    G = nx.DiGraph()

    # Ajout des nœuds
    for _, row in stops.iterrows():
        G.add_node(row["stop_id"],
                   name=row.get("stop_name", ""),
                   lat=row["stop_lat"],
                   lon=row["stop_lon"])

    # Ajout des arêtes
    st_sorted = stop_times.sort_values(["trip_id", "stop_sequence"])
    for trip_id, group in st_sorted.groupby("trip_id"):
        group = group.reset_index(drop=True)
        for i in range(len(group) - 1):
            u = group.loc[i,   "stop_id"]
            v = group.loc[i+1, "stop_id"]
            try:
                t1 = parse_time(group.loc[i,   "departure_time"])
                t2 = parse_time(group.loc[i+1, "arrival_time"])
                w  = t2 - t1
            except Exception:
                w = 60  # valeur par défaut : 1 minute
            if w > 0 and u in G and v in G:
                if G.has_edge(u, v):
                    # Moyenne des poids si l'arête existe déjà
                    G[u][v]["weight"] = (G[u][v]["weight"] + w) / 2
                else:
                    G.add_edge(u, v, weight=w)

    print(f"  Graphe : {G.number_of_nodes()} nœuds, "
          f"{G.number_of_edges()} arêtes")
    return G


# ─────────────────────────────────────────────
# 3. FIGURE 1 — Sous-graphe visualisé
# ─────────────────────────────────────────────
def fig_subgraph(G, output_dir):
    print("Figure 1 : sous-graphe...")

    # Garder les 150 nœuds de plus haut degré
    top_nodes = sorted(G.degree(), key=lambda x: x[1], reverse=True)[:150]
    top_ids   = [n for n, _ in top_nodes]
    sub = G.subgraph(top_ids).copy()

    fig, ax = plt.subplots(figsize=(10, 8))

    pos = nx.spring_layout(sub, seed=42, k=0.3)

    degrees = dict(sub.degree())
    node_sizes  = [30 + degrees[n] * 15 for n in sub.nodes()]
    node_colors = [degrees[n] for n in sub.nodes()]

    nx.draw_networkx_edges(sub, pos, ax=ax,
                           alpha=0.25, width=0.5,
                           edge_color="gray", arrows=False)
    sc = nx.draw_networkx_nodes(sub, pos, ax=ax,
                                node_size=node_sizes,
                                node_color=node_colors,
                                cmap=plt.cm.plasma, alpha=0.9)

    # Labels uniquement pour les 10 plus connectés
    top10 = {n: G.nodes[n].get("name", n)[:15]
             for n, _ in top_nodes[:10]}
    nx.draw_networkx_labels(sub, pos, labels=top10, ax=ax,
                            font_size=6, font_color="black")

    plt.colorbar(sc, ax=ax, label="Degré du nœud")
    ax.set_title("Sous-graphe du réseau de bus IDF\n"
                 "(150 arrêts les plus connectés)", fontsize=13)
    ax.axis("off")
    plt.tight_layout()
    path = f"{output_dir}/fig1_subgraph.pdf"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Sauvegardé : {path}")


# ─────────────────────────────────────────────
# 4. FIGURE 2 — Distribution des degrés
# ─────────────────────────────────────────────
def fig_degree_distribution(G, output_dir):
    print("Figure 2 : distribution des degrés...")

    degrees = [d for _, d in G.degree()]
    counter = Counter(degrees)
    x = sorted(counter.keys())
    y = [counter[k] for k in x]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Histogramme
    axes[0].bar(x, y, color="steelblue", edgecolor="white", linewidth=0.5)
    axes[0].set_xlabel("Degré (entrées + sorties)", fontsize=11)
    axes[0].set_ylabel("Nombre d'arrêts", fontsize=11)
    axes[0].set_title("Distribution des degrés", fontsize=12)
    axes[0].set_xlim(0, min(50, max(x)))
    axes[0].axvline(np.mean(degrees), color="red", linestyle="--",
                    label=f"Moyenne = {np.mean(degrees):.1f}")
    axes[0].legend()

    # Log-log (loi de puissance ?)
    axes[1].loglog(x, y, "o", color="steelblue", markersize=4, alpha=0.7)
    axes[1].set_xlabel("Degré (log)", fontsize=11)
    axes[1].set_ylabel("Nombre d'arrêts (log)", fontsize=11)
    axes[1].set_title("Distribution des degrés (échelle log-log)", fontsize=12)
    axes[1].grid(True, which="both", linestyle="--", alpha=0.5)

    plt.suptitle("Analyse de la distribution des degrés\n"
                 "Réseau de bus Île-de-France", fontsize=13)
    plt.tight_layout()
    path = f"{output_dir}/fig2_degree_distribution.pdf"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Sauvegardé : {path}")

    # Stats affichées
    print(f"  Degré moyen    : {np.mean(degrees):.2f}")
    print(f"  Degré max      : {max(degrees)}")
    print(f"  Degré médian   : {np.median(degrees):.1f}")


# ─────────────────────────────────────────────
# 5. FIGURE 3 — Centralité (betweenness)
# ─────────────────────────────────────────────
def fig_centrality(G, output_dir):
    print("Figure 3 : centralité (betweenness) — peut prendre quelques minutes...")

    # Approximation par échantillonnage
    bc = nx.betweenness_centrality(G, k=N_SAMPLE_CENTRALITY,
                                   normalized=True, weight="weight")

    # Top 15 arrêts
    top15 = sorted(bc.items(), key=lambda x: x[1], reverse=True)[:15]
    names  = [G.nodes[n].get("name", n)[:25] for n, _ in top15]
    values = [v for _, v in top15]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = cm.RdYlGn_r(np.linspace(0.1, 0.9, len(names)))
    bars = ax.barh(range(len(names)), values, color=colors)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Centralité d'intermédiarité (betweenness)", fontsize=11)
    ax.set_title("Top 15 des arrêts les plus centraux\n"
                 "Réseau de bus Île-de-France", fontsize=12)
    ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=8)
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    path = f"{output_dir}/fig3_centrality.pdf"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Sauvegardé : {path}")
    return bc


# ─────────────────────────────────────────────
# 6. FIGURE 4 — Plus court chemin (Dijkstra)
# ─────────────────────────────────────────────
def fig_shortest_path(G, output_dir,
                      start_name="Gare du Nord",
                      end_name="Gare de Lyon"):
    print(f"Figure 4 : plus court chemin ({start_name} → {end_name})...")

    # Trouver les IDs par nom
    name_to_id = {data.get("name", ""): nid
                  for nid, data in G.nodes(data=True)}

    sid = name_to_id.get(start_name)
    eid = name_to_id.get(end_name)

    if sid is None or eid is None:
        print("  Arrêts non trouvés, utilisation de nœuds aléatoires.")
        nodes = list(G.nodes())
        sid, eid = nodes[0], nodes[100]

    try:
        path = nx.dijkstra_path(G, sid, eid, weight="weight")
        length = nx.dijkstra_path_length(G, sid, eid, weight="weight")
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        print("  Gares isolées. Recherche d'un chemin alternatif dans la composante principale...")
        
        # 1. On récupère la plus grande composante
        largest_cc = max(nx.weakly_connected_components(G), key=len)
        G_sub_main = G.subgraph(largest_cc)
        
        # 2. On prend deux nœuds au hasard mais assez éloignés dans cette composante
        nodes_in_cc = list(G_sub_main.nodes())
        sid = nodes_in_cc[0]
        eid = nodes_in_cc[min(500, len(nodes_in_cc)-1)] # Un nœud un peu plus loin
        
        # Mettre à jour les noms pour la figure
        start_name = G.nodes[sid].get("name", "Stop A")
        end_name = G.nodes[eid].get("name", "Stop B")
        
        path = nx.dijkstra_path(G_sub_main, sid, eid, weight="weight")
        length = nx.dijkstra_path_length(G_sub_main, sid, eid, weight="weight")

    # Sous-graphe : voisinage des nœuds du chemin
    neighborhood = set(path)
    for n in path:
        neighborhood.update(list(G.predecessors(n))[:3])
        neighborhood.update(list(G.successors(n))[:3])
    sub = G.subgraph(neighborhood).copy()

    fig, ax = plt.subplots(figsize=(10, 7))
    pos = nx.spring_layout(sub, seed=0, k=0.5)
    print(f"DEBUG: ID Nord = {sid}, ID Lyon = {eid}")
    
    # Couleurs des nœuds
    node_colors = []
    for n in sub.nodes():
        if n == sid or n == eid:
            node_colors.append("red")
        elif n in path:
            node_colors.append("orange")
        else:
            node_colors.append("lightblue")

    nx.draw_networkx_nodes(sub, pos, ax=ax,
                           node_color=node_colors, node_size=200)
    nx.draw_networkx_edges(sub, pos, ax=ax,
                           edge_color="lightgray", alpha=0.5,
                           arrows=False, width=1)

    # Mettre en évidence le chemin
    path_edges = list(zip(path[:-1], path[1:]))
    nx.draw_networkx_edges(sub, pos, edgelist=path_edges, ax=ax,
                           edge_color="red", width=2.5, arrows=True,
                           arrowsize=15)

    # Labels des nœuds du chemin
    labels = {n: G.nodes[n].get("name", n)[:15] for n in path}
    nx.draw_networkx_labels(sub, pos, labels=labels, ax=ax,
                            font_size=7, font_color="black",
                            font_weight="bold")

    total_min = int(length // 60)
    ax.set_title(
        f"Plus court chemin : {start_name} → {end_name}\n"
        f"(Dijkstra, {len(path)} arrêts, durée estimée ≈ {total_min} min)",
        fontsize=12)
    ax.axis("off")

    # Légende
    from matplotlib.patches import Patch
    legend = [Patch(color="red",       label="Départ / Arrivée"),
              Patch(color="orange",    label="Arrêts du chemin"),
              Patch(color="lightblue", label="Arrêts voisins")]
    ax.legend(handles=legend, loc="lower right", fontsize=9)

    plt.tight_layout()
    path_fig = f"{output_dir}/fig4_shortest_path.pdf"
    plt.savefig(path_fig, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Sauvegardé : {path_fig}")


# ─────────────────────────────────────────────
# 7. FIGURE 5 — Composantes connexes
# ─────────────────────────────────────────────
def fig_connected_components(G, output_dir):
    print("Figure 5 : composantes connexes...")

    # Graphe non orienté pour les composantes faiblement connexes
    UG = G.to_undirected()
    components = sorted(nx.connected_components(UG),
                        key=len, reverse=True)

    sizes = [len(c) for c in components]
    labels_cc = [f"CC {i+1}\n({s} nœuds)" if i < 5 else ""
                 for i, s in enumerate(sizes)]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Camembert des 6 plus grandes composantes + "autres"
    top6   = sizes[:6]
    others = sum(sizes[6:])
    pie_sizes  = top6 + ([others] if others > 0 else [])
    pie_labels = [f"CC {i+1} ({s})" for i, s in enumerate(top6)]
    if others > 0:
        pie_labels.append(f"Autres ({len(components)-6} CC)")

    axes[0].pie(pie_sizes, labels=pie_labels, autopct="%1.1f%%",
                startangle=140, colors=plt.cm.tab10.colors)
    axes[0].set_title("Répartition des composantes connexes\n"
                      "(faiblement connexes)", fontsize=11)

    # Histogramme des tailles
    axes[1].hist(sizes, bins=30, color="steelblue",
                 edgecolor="white", log=True)
    axes[1].set_xlabel("Taille de la composante (nœuds)", fontsize=11)
    axes[1].set_ylabel("Nombre de composantes (log)", fontsize=11)
    axes[1].set_title("Distribution des tailles de composantes", fontsize=11)
    axes[1].grid(axis="y", linestyle="--", alpha=0.5)

    plt.suptitle(f"Analyse de la connexité — {len(components)} composantes connexes",
                 fontsize=13)
    plt.tight_layout()
    path = f"{output_dir}/fig5_components.pdf"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Sauvegardé : {path}")
    print(f"  Composante principale : {sizes[0]} nœuds "
          f"({100*sizes[0]/G.number_of_nodes():.1f}% du graphe)")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    stops, bus_trips, stop_times = load_gtfs(DATA_DIR)
    G = build_graph(stops, bus_trips, stop_times)

    fig_subgraph(G, OUTPUT_DIR)
    fig_degree_distribution(G, OUTPUT_DIR)
    fig_centrality(G, OUTPUT_DIR)
    fig_shortest_path(G, OUTPUT_DIR,
                      start_name="Gare du Nord",
                      end_name="Gare de Lyon")
    fig_connected_components(G, OUTPUT_DIR)

    print("\nToutes les figures sont dans le dossier ./figures/")
    print("Lancez maintenant la compilation LaTeX.")

# ─────────────────────────────────────────────
# 7. FIGURE 5 — Composantes connexes
# ─────────────────────────────────────────────
def fig_connected_components(G, output_dir):
    print("Figure 5 : composantes connexes...")

    # Graphe non orienté pour les composantes faiblement connexes
    UG = G.to_undirected()
    components = sorted(nx.connected_components(UG),
                        key=len, reverse=True)

    sizes = [len(c) for c in components]
    labels_cc = [f"CC {i+1}\n({s} nœuds)" if i < 5 else ""
                 for i, s in enumerate(sizes)]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Camembert des 6 plus grandes composantes + "autres"
    top6   = sizes[:6]
    others = sum(sizes[6:])
    pie_sizes  = top6 + ([others] if others > 0 else [])
    pie_labels = [f"CC {i+1} ({s})" for i, s in enumerate(top6)]
    if others > 0:
        pie_labels.append(f"Autres ({len(components)-6} CC)")

    axes[0].pie(pie_sizes, labels=pie_labels, autopct="%1.1f%%",
                startangle=140, colors=plt.cm.tab10.colors)
    axes[0].set_title("Répartition des composantes connexes\n"
                      "(faiblement connexes)", fontsize=11)

    # Histogramme des tailles
    axes[1].hist(sizes, bins=30, color="steelblue",
                 edgecolor="white", log=True)
    axes[1].set_xlabel("Taille de la composante (nœuds)", fontsize=11)
    axes[1].set_ylabel("Nombre de composantes (log)", fontsize=11)
    axes[1].set_title("Distribution des tailles de composantes", fontsize=11)
    axes[1].grid(axis="y", linestyle="--", alpha=0.5)

    plt.suptitle(f"Analyse de la connexité — {len(components)} composantes connexes",
                 fontsize=13)
    plt.tight_layout()
    path = f"{output_dir}/fig5_components.pdf"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Sauvegardé : {path}")
    print(f"  Composante principale : {sizes[0]} nœuds "
          f"({100*sizes[0]/G.number_of_nodes():.1f}% du graphe)")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    stops, bus_trips, stop_times = load_gtfs(DATA_DIR)
    G = build_graph(stops, bus_trips, stop_times)

    fig_subgraph(G, OUTPUT_DIR)
    fig_degree_distribution(G, OUTPUT_DIR)
    fig_centrality(G, OUTPUT_DIR)
    fig_shortest_path(G, OUTPUT_DIR,
                      start_name="Gare du Nord",
                      end_name="Gare de Lyon")
    fig_connected_components(G, OUTPUT_DIR)

    print("\nToutes les figures sont dans le dossier ./figures/")
    print("Lancez maintenant la compilation LaTeX.")
