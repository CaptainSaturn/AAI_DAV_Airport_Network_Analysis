import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import cartopy.crs as ccrs
import cartopy.feature as cfeature
plt.style.use('seaborn-whitegrid')
COUNTRIES = ["USA", "China", "UK", "Australia"]
COORDINATES = {"USA": [-180, -66, 16, 60],
               "China": [70, 140, 15, 54],
               "UK": [-10, 2, 50, 60],
               "Australia": [110, 155, -40, -10]}

labelIndex = {"USA": 213000,
         "China": 51100,
         "UK": 48700,
         "Australia": 10704}

widthIndex = {"USA": 0.5,
           "China": 0.75,
           "UK": 0.9,
           "Australia": 5}

# Importing the data
airports_names = pd.read_csv("Airports.csv", encoding= 'unicode_escape')
routes_2003 = pd.read_csv("Flight_Data_p1.csv", encoding= 'unicode_escape')

# set IATA airoprt codes as index column
airports_names = airports_names.set_index("id")

# Remove countries not in the assignment
routes_2003 = routes_2003[routes_2003["Source Country"].isin(COUNTRIES)].reset_index(drop=True)

# Convert weights into int
routes_2003["Weight"] = routes_2003["Weight"].astype(int)

def network_graph(country, airports_names = airports_names, routes=routes_2003, plot=True, output_g=False, output_all=False):
    country = "NAN" if country.upper() == "NA" else country
    airport_country_filter = "NAN" if country.upper() == "NA" else country
    airports_country = airports_names[airports_names["country"] == airport_country_filter]

    if country.upper() == "USA":
        airports_country = airports_country[airports_country["Lon"] != -70]
  
    routes_country = routes[routes["Source Country"] == country]
    routes_country = routes_country[routes_country["Source"].isin(airports_country.index) &
                                    routes_country["Target"].isin(airports_country.index)]

    weight_edges = routes_country[["Source", "Target", "Weight"]].values
    g = nx.DiGraph()
    g.add_weighted_edges_from(weight_edges)
    print("type of g is: ",type(g))

    pos = {airport: (v["Lon"], v["Lat "]) for airport, v in
           airports_country.to_dict('index').items()}

    deg = nx.degree(g, weight='weight')
    all_sizes = [deg[iata] for iata in g.nodes]
    sizes = [(((deg[iata] - min(all_sizes)) * (300 - 17)) / (max(all_sizes) - min(all_sizes))) + 1 for iata in g.nodes]

    labels = {iata: iata if deg[iata] >= labelIndex[country] else ''
              for iata in g.nodes}

    all_weights = [data['weight'] for node1, node2, data in g.edges(data=True)]
    edge_width = [(((weight - min(all_weights)) * (widthIndex[country] - 0.075)) / (max(all_weights) - min(all_weights))) + 0.075
                  for weight in all_weights]

    if plot:
        crs = ccrs.PlateCarree()
        fig, ax = plt.subplots(
            1, 1, figsize=(17, 8),
            subplot_kw=dict(projection=crs))
        ax.coastlines()
        ax.add_feature(cfeature.BORDERS)
        ax.add_feature(cfeature.LAND)
        ax.add_feature(cfeature.OCEAN, alpha = 0.33)
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        ax.add_feature(cfeature.RIVERS, alpha = 0.33)
        ax.add_feature(cfeature.LAKES, alpha = 0.33)
        ax.set_extent(COORDINATES[country])
        ax.gridlines()
        nx.draw_networkx(g, ax=ax,
                         font_size=17,
                         alpha=.5,
                         width=edge_width,
                         node_size=sizes,
                         labels=labels,
                         pos=pos,
                         node_color=sizes,
                         cmap=plt.cm.plasma)
        plt.title(str(country) + r' Airline Network Graph for 01.07.2003')
        plt.ylabel(r"Logscale of weighted degree")
        plt.xlabel(r"Descending rank")
        plt.show()

    if output_all:
        return airports_country, routes_country, g, weight_edges, pos, deg, sizes, labels, all_weights, edge_width
    if output_g:
        return g


# plot f degree distribution 
# Degree distribution (x-axis: descending rank, y-axis: logscale of weighted degree).  
def degree_distribution(deg, country):
    if len(deg) > 1:
        degree_sequence_all = sorted([d for n, d in deg[2]], reverse=True)

        plt.semilogy(degree_sequence_all, marker=".", color = "r", markersize=2)
        plt.legend(loc="best")
    else:
        degree_sequence = sorted([d for n, d in deg[0]], reverse=True)
        plt.semilogy(degree_sequence, marker=".", color = "r", markersize=2)

    plt.title(str(country) + r' Degree Distribution Plot for 01.07.2003')
    plt.ylabel(r"Logscale of weighted degree")
    plt.xlabel(r"Descending rank")
    plt.show()


# degree vs betweenness distr
def degree_betweenness(G, deg, country):
    # init the plot
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.loglog()

    if len(deg) > 1:

        for k in range(len(deg)):
            b = nx.betweenness_centrality(G[k], normalized=False)
            x = [deg[k][iata] for iata in G[k].nodes]
            y = [b[iata] for iata in G[k].nodes]
            labels = [iata for iata in G[k].nodes]

            plt.scatter(x, y, alpha=0.75, color = "r")
            for i in range(len(labels)):
                ax.annotate(labels[i], (x[i], y[i]), size = 8)
    else:
        b = nx.betweenness_centrality(G, normalized=False)
        x = [deg[iata] for iata in G.nodes]
        y = [b[iata] for iata in G.nodes]
        labels = [iata for iata in G.nodes]

        fig = plt.figure(figsize=(8, 8), dpi=300)
        ax = fig.add_subplot(1, 1, 1)
        ax.loglog()
        plt.scatter(x, y, alpha=0.75, color = "r")
        for i in range(len(labels)):
            ax.annotate(labels[i], (x[i], y[i]), size = 8)

    plt.ylim(0.1, 10000)
    plt.legend(loc="best")
    plt.title(str(country) + r' Degree vs Betweenness for 01.07.2003')
    plt.ylabel(r"Betweenness")
    plt.xlabel(r"Degree")
    plt.show()


# assortativity
def assort(G):
    r = nx.degree_pearson_correlation_coefficient(G, weight="weight")
    print("assortititiitvy is: ",r)
    return r

def assortitivity(G):
    node_degree, neighbor_degree = zip(*nx.average_degree_connectivity(G, weight = 'weight').items())

    fig, ax = plt.subplots(1,1,figsize = (7,7))
    plt.scatter(node_degree, neighbor_degree, marker = 'o', color = "r")
    plt.xlabel("Node Degree")
    plt.ylabel("Nearest Neighbors Degree")
    plt.title(str(country) + r" Nearest Neighbors Degree for 01.07.2003")
    plt.show()

# core community size
def core_community(G, country):
    # does what is explained on slide 51 of AVDC_2019-2022_AIAS_Lecture_Graph & Visualisation.pdf
    fig, ax = plt.subplots(1, 1, figsize=(17, 8))
    time_label = ["01.07.2003"]
    color = ["#4C72B0"]

    s = []

    for k in range(len(G)):
        try:
            core = nx.k_core(G[k], core_number=nx.core_number(G[k]))
        except nx.exception.NetworkXError:
            g = G[k]
            g.remove_edges_from(list(nx.selfloop_edges(g)))
            core = nx.k_core(g, core_number=nx.core_number(g))
        s.append(len(core))

        nx.draw_networkx(core, ax=ax, label=time_label[0], alpha=0.50, node_color=color[0], edge_color=color[0])
    plt.legend(loc="best")
    plt.show()

    return s

def core_community_plot(G):
    d = pd.DataFrame([(n, v) for (n, v) in sorted(G.degree, key = lambda x: x[1], reverse = True)]).rename(columns = {0:'id', 1:'degree'}, inplace = False)
    conn_hi_deg = []
    for id in d['id']:
        pseudo_var = 0
        for i in G.neighbors(id):
            if d[d['id'] == i].iloc[0]['degree'] > d[d['id'] == id].iloc[0]['degree']:
                pseudo_var += 1
        conn_hi_deg.append(pseudo_var)
        
    fig, ax = plt.subplots(1, 1 ,figsize = (10, 5))
    ax.plot(conn_hi_deg, color = "r", marker = '.', alpha = 0.5)
    plt.ylabel("No. of connections to higher degree nodes")
    plt.xlabel("Ranked Node number")
    plt.title(str(country) + r" Core Community Plot for 01.07.2003")
    plt.show()    

if __name__ == "__main__":
    country = "UK"  
    g_old = network_graph(country, routes=routes_2003, output_g=True)
    print("Type of g_old is: ", type(g_old))
    
    g_new = network_graph(country, routes=routes_2003, output_g=True)
    print("Type of g_new is: ", type(g_new))
          
    g_all = network_graph(country, routes=routes_2003, output_g=True)
    print("Type of g_all is: ", type(g_all))

    deg = [nx.degree(g_old, weight='weight'), nx.degree(g_new, weight='weight'), nx.degree(g_all, weight='weight')]
    G = [g_old, g_new, g_all]
    print("Type of G is: ", type(G))

    #assortitivity(g_all)

    #degree_distribution(deg, country)

    # if deg and G is a list of old, new, and all than plot the three on one graph
    degree_betweenness(G, deg, country)

    # r_old = assort(g_old)
    # r_new = assort(g_new)
    #r_all = assort(g_all)
    # print(f"{country} & {r_old} & {r_new} % {r_all} \\\\")
    

    #s = core_community(G, country)
    
    #core_community_plot(g_all)
   
print("Done")

