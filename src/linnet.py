import geopandas as gpd
import pandas as pd
import numpy as np
import networkx as nx
import momepy
import matplotlib.pyplot as plt
from scipy.spatial import distance
from scipy import integrate
from tqdm import tqdm
import warnings
from itertools import product
from src.utils import powerexponential, matern
import shapely

warnings.filterwarnings("ignore")

def distR(segu, tu, segv, tv, L):
    """
    Calculate pairwise resistance distance between two points
    Args:
        segu (int) : segment u
        tu (float) : position on segment u
        segv (int) : segment v
        tv (float) : position on segment v
        L (Linnet) : Linnet object
    Returns:
        float : resistance distance

    Source : R function by Moller, J., & Rasmussen (2024)
    """
    # Source : Moller, J., & Rasmussen (2024)
    Inverse_Lap = np.linalg.inv(L.calcLaplacian()) # Inverse of Laplacian Matrix
    Sigajaj = Inverse_Lap[L._from[segu], L._from[segu]]
    Sigajbj = Inverse_Lap[L._from[segu], L._to[segu]]
    Sigbjbj = Inverse_Lap[L._to[segu], L._to[segu]]
    Sigaiai = Inverse_Lap[L._from[segv], L._from[segv]]
    Sigaibi = Inverse_Lap[L._from[segv], L._to[segv]]
    Sigbibi = Inverse_Lap[L._to[segv], L._to[segv]]
    Sigajai = Inverse_Lap[L._from[segu], L._from[segv]]
    Sigajbi = Inverse_Lap[L._from[segu], L._to[segv]]
    Sigbjai = Inverse_Lap[L._to[segu], L._from[segv]]
    Sigbjbi = Inverse_Lap[L._to[segu], L._to[segv]]
    li = L._len[segv]
    lj = L._len[segu]
    dist = ((1-tu)**2 * Sigajaj + tu**2 * Sigbjbj + 2*tu*(1-tu) * Sigajbj
          + (1-tv)**2 * Sigaiai + tv**2 * Sigbibi + 2*tv*(1-tv) * Sigaibi
          - 2*(1-tu)*(1-tv) * Sigajai - 2*(1-tu)*tv * Sigajbi
          - 2*tu*(1-tv) * Sigbjai - 2*tu*tv * Sigbjbi
          + tv*(1-tv)*li + tu*(1-tu)*lj
          - 2*(segu == segv)*min(tu*(1-tv),tv*(1-tu))*li)

    return dist

def pairdistR(X, L, X2=None):
    """
    Calculate pairwise resistance distance between all pairs of points
    Args:
        X (gpd.GeoSeries) : GeoSeries of points
        X2 (gpd.GeoSeries) : GeoSeries of points (default=None)
        L (Linnet) : Linnet object

    Returns:
        np.array : pairwise resistance distance matrix
    """
    # Source : Moller, J., & Rasmussen (2024) 
    # X : Geoseries/GeoDataFrame of points
    X = snap_points(L, X)
    if X2 is None:
        X2 = X
    else:
        X2 = snap_points(L, X2)
    n, m = len(X), len(X2)
    Inverse_Lap = np.linalg.inv(L.calcLaplacian()) # Inverse of Laplacian Matrix
    segu = X['edge_name'].values
    tu = X['tp'].values
    segv = X2['edge_name'].values
    tv = X2['tp'].values

    # segu outer segv 

    onesu = np.ones(n)
    onesv = np.ones(m)
    Sigajaj = np.outer(np.diagonal(Inverse_Lap)[L._from[segu]], onesv)
    Sigajbj = np.outer(Inverse_Lap[L._from[segu], L._to[segu]], onesv)
    Sigbjbj = np.outer(np.diagonal(Inverse_Lap)[L._to[segu]], onesv)
    Sigaiai = np.outer(onesu, np.diagonal(Inverse_Lap)[L._from[segv]])
    Sigaibi = np.outer(onesu, Inverse_Lap[L._from[segv], L._to[segv]])
    Sigbibi = np.outer(onesu, np.diagonal(Inverse_Lap)[L._to[segv]])
    Sigajai = Inverse_Lap[np.array(L._from[segu])[:,None], L._from[segv]]
    Sigajbi = Inverse_Lap[np.array(L._from[segu])[:,None], L._to[segv]]
    Sigbjai = Inverse_Lap[np.array(L._to[segu])[:,None], L._from[segv]]
    Sigbjbi = Inverse_Lap[np.array(L._to[segu])[:,None], L._to[segv]]
    li = np.outer(onesu, L._len[segv])
    liv = L._len[segv]
    lj = np.outer(L._len[segu], onesv)
    tum = np.outer(tu, onesv)
    tvm = np.outer(onesu, tv)

    dist = ((1-tum)**2 * Sigajaj + tum**2 * Sigbjbj + 2*tum*(1-tum) * Sigajbj
            + (1-tvm)**2 * Sigaiai + tvm**2 * Sigbibi + 2*tvm*(1-tvm) * Sigaibi
            - 2*(1-tum)*(1-tvm) * Sigajai - 2*(1-tum)*tvm * Sigajbi
            - 2*tum*(1-tvm) * Sigbjai - 2*tum*tvm * Sigbjbi
            + tvm*(1-tvm)*li + tum*(1-tum)*lj
            - 2*np.equal.outer(segu,segv)*np.minimum(np.outer(tu,1-tv),np.outer(1-tu,tv))*np.outer(onesu,liv))

    return dist * (dist > 0)

def snap_points(L, points):
    """
    Snap points to the nearest edge on the network
    Args:
        L (Linnet) : Linnet object
        points (gpd.GeoSeries) : GeoSeries of points
    Returns:
        gpd.GeoDataFrame : snapped points
    """
    
    snapped_df = pd.DataFrame(points.geometry.apply(lambda x: L.snap(x)).tolist(), columns=['snap', 'edge'])
    snapped_df['node_start'] = snapped_df['edge'].apply(lambda x: x.node_start)
    snapped_df['node_end'] = snapped_df['edge'].apply(lambda x: x.node_end)
    snapped_df['edge_name'] = snapped_df['edge'].apply(lambda x: x.name)
    snapped_df['edge_len'] = snapped_df['edge'].apply(lambda x: x.mm_len)
    snapped_df.drop(columns=['edge'], inplace=True)
    snapped_df['d_start'] = snapped_df.apply(lambda x: x.snap.distance(L.nodes.geometry[x.node_start]), axis=1)
    snapped_df['d_end'] = snapped_df.apply(lambda x: x.snap.distance(L.nodes.geometry[x.node_end]), axis=1)
    snapped_df['tp'] = snapped_df['d_start'] / snapped_df['edge_len'] # position on the edge

    return gpd.GeoDataFrame(snapped_df, geometry='snap')

def snap_to_linnet(L, points):
    """
    Snap points to the nearest edge on the network and add the snapped points to the node set
    Args:
        L (Linnet) : Linnet object
        points (gpd.GeoSeries) : GeoSeries of points
    Returns:
        gpd.GeoDataFrame : snapped points
    """
    snapped = snap_points(L, points)
    


class Linnet(nx.MultiGraph):
    """
    Create a networkx MultiGraph object from a GeoDataFrame of LINESTRINGs
    Attributes:
        nodes (GeoDataFrame) : GeoDataFrame of nodes
        edges (GeoDataFrame) : GeoDataFrame of edges
        sw (libpysal.weights.W) : spatial weights object
    """
    def __init__(self, edges):
        super().__init__()
        assert isinstance(edges, (gpd.GeoSeries, gpd.GeoDataFrame)), "Edges must be a GeoSeries or GeoDataFrame object"
        self.graph = momepy.gdf_to_nx(edges)
        nodes, edges, _ = momepy.nx_to_gdf(self.graph, points=True, lines=True, spatial_weights=True)
        self.nodes = nodes
        self.edges = edges
        self._from = self.edges['node_start']
        self._to = self.edges['node_end']
        self._len = self.edges['mm_len']
        self.shortest_path = nx.floyd_warshall_numpy(self.graph, weight='mm_len')
        self.adjacency = nx.adjacency_matrix(self.graph).toarray()

    def plot(self, ax=None, **kwargs):
        """
        Plot the network
        Args:
            ax (matplotlib.axes.Axes) : axes to plot
            **kwargs : keyword arguments to pass to GeoDataFrame.plot
        """
        if ax is None:
            fig, ax = plt.subplots()
        self.edges.plot(ax=ax, **kwargs)
        self.nodes.plot(ax=ax, color='tab:red', markersize=10)
        return ax
    
    def count_points(self, points):
        """
        Count the number of points on the network
        Create 'count' column in the edges and nodes GeoDataFrame
        """
        cnts = self.edges.sjoin_nearest(points, how='right').groupby('index_left').size()
        cnts.name = 'count'
        self.edges = self.edges.join(cnts).fillna(0)

    def snap_points(self, points):
        """
        Snap points to the nearest edge on the network
        Args:
            points (gpd.GeoSeries) : GeoSeries of points
        Returns:
            gpd.GeoDataFrame : snapped points
        """
        return snap_points(self, points)

    def projection(self, points):
        """
        Get projection matrix P for a set of points
        Args:
            points (gpd.GeoSeries) : GeoSeries of points
        Returns:
            np.array : projection matrix (npoints x nvertices)
        """
        snapped = self.snap_points(points)
        P = np.zeros((snapped.shape[0], self.nodes.shape[0]))

        node_start_indices = snapped['node_start'].values
        node_end_indices = snapped['node_end'].values
        tp_values = snapped['tp'].values

        P[np.arange(snapped.shape[0]), node_start_indices] = 1 - tp_values
        P[np.arange(snapped.shape[0]), node_end_indices] = tp_values

        return P
    
    def get_dual_lengths(self):
        """
        Get the dual discretization lengths
        Returns:
            np.array : dual lengths (num of nodes)
        """
        _nodes = self.nodes.copy()
        _nodes['start_sum'] = self.edges.groupby('node_start')['mm_len'].sum()
        _nodes['end_sum'] = self.edges.groupby('node_end')['mm_len'].sum()
        _nodes['sum'] = _nodes['start_sum'].fillna(0) + _nodes['end_sum'].fillna(0)
        
        return _nodes['sum'].values / 2


    def calcLaplacian(self):
        L = - self.adjacency * (1 / self.shortest_path)
        L[np.diag_indices_from(L)] = 0
        L[np.diag_indices_from(L)] = -L.sum(axis=1)
        L[0, 0] += 1
        return L
    
    def discretize(self, length):
        """
        Return the equally spaced points along the network
        Args:
            length (float) : maximum distance between two points
            points (gpd.GeoSeries) : GeoSeries of point patterns
        Returns:
            discretized_Linnet : discretized network (Linnet object)
        """
        vs = []
        edgenum = []
        for edge in self.edges.itertuples():
            n = np.ceil(edge.mm_len / length).astype(int) + 1
            interpolates = [edge.geometry.interpolate(i, normalized=True) for i in np.linspace(0, 1, n)]
            # to vertices
            interpolates_lines = [shapely.geometry.LineString([interpolates[i], interpolates[i+1]]) for i in range(n-1)]
            vs.extend(interpolates_lines)
            edgenum.extend([edge.Index] * (n-1))

            # if return_nodes:
            #     last_nodeid = nodes['nodeID'].max()
            #     to_append = pd.DataFrame({
            #         'nodeID': np.arange(last_nodeid+1, last_nodeid+n-1),
            #         'geometry': interpolates[1:-1] # n-2
            #     })
            #     nodes = pd.concat([nodes, to_append], ignore_index=True)

        # to edges
        edges = gpd.GeoDataFrame(geometry=vs)
        edges['edge_num'] = edgenum

        # if return_nodes:
        #     return edges, nodes
        # else:
        #     return edges
        return Linnet(edges)
    
    def _return_edge_vals(self):
        return list(zip(self.edges.node_start, self.edges.node_end, self.edges.mm_len))
    


    # def d_G(self, source, target):
    #     """
    #     Return the shortest path distance(geodesic) between two nodes
    #     Args:
    #         source : source node index
    #         target : target node index
    #     Returns:
    #         list : shortest path
    #     """
    #     # return nx.shortest_path_length(self.graph, source=source, target=target, weight='mm_len')
    #     return self.shortest_path[source][target]

    # def conductance(self, source, target):
    #     """
    #     Return the conductance between two nodes
    #     Args:
    #         source : source node index
    #         target : target node index
    #     Returns:
    #         float : conductance function value
    #     """
    #     # If source and target are neighbors, conductance is 1/d_G
    #     ## Check if source and target are neighbors

    #     if self.adjacency[source, target] == 1:
    #         return 1/self.d_G(source, target)
    #     else:
    #         # Conductance is 0
    #         return 0

    # def _c(self, source):
    #     """
    #     Return the c(u) function of a node
    #     Args:
    #         source : source node index
    #     Returns:
    #         float : c(u) function value
    #     """
    #     # Return the sum of conductance between origin and all other neighbors
    #     c_u = 0
    #     for neighbor in np.where(self.adjacency[source] == 1)[0]:
    #         c_u += self.conductance(source, neighbor)
    #     return c_u

    # def L_uv(self, source, target):
    #     """
    #     Return the L(u,v) function between two nodes
    #     Args:
    #         source : source node index
    #         target : target node index
    #     Returns:
    #         float : L(u,v) function value
    #     """
    #     if source == target == self.origin:
    #         return 1 + self._c(source)
    #     elif source == target:
    #         return self._c(source)
    #     else:
    #         return (-1) * self.conductance(source, target)

    # def _L_inv(self, return_matrix=False):
    #     """
    #     Return the inverse of the L(u,v) matrix
    #     Returns:
    #         np.array : inverse of L(u,v) matrix (L**-1)
    #     """
    #     V = len(self.graph)
    #     L = np.zeros((V, V))
    #     for i in range(V):
    #         for j in range(V):
    #             L[i, j] = self.L_uv(i, j)
            
    #     assert np.all(np.linalg.eigvals(L) > 0), "L(u,v) matrix is not positive definite"

    #     self.L_inv = np.linalg.inv(L)
    #     if return_matrix:
    #         return self.L_inv

    def snap(self, point):
        """
        Snap a point to the nearest edge on the network
        Args:
            point (Point) : point to snap
        Returns:
            (Point, GeoDataFrame) : snapped point, nearest edge (GeoDataFrame)
        """
        nidx = self.edges.sindex.nearest(point)[1][0]
        nedge = self.edges.loc[nidx]
        snap = nedge.geometry.interpolate(nedge.geometry.project(point))
        return snap, nedge


    def geodesic_metric(self, source, target):
        """
        Return the geodesic metric between two points on the network
        Args:
            source : source point
            target : target point
        Returns:
            float : geodesic metric
        """
        # Snap source and target to nearest edges
        source_snap, source_edge = self.snap(source)
        target_snap, target_edge = self.snap(target)

        if source_edge.name == target_edge.name: # on the same edge
            return source_snap.distance(target_snap)

        # Get the shortest path
        source_l, source_u = source_edge.node_start, source_edge.node_end
        target_l, target_u = target_edge.node_start, target_edge.node_end

        # dist to nearest edges
        source_d_start = source_snap.distance(self.nodes.geometry[source_l])
        source_d_end = source_snap.distance(self.nodes.geometry[source_u])
        target_d_start = target_snap.distance(self.nodes.geometry[target_l])
        target_d_end = target_snap.distance(self.nodes.geometry[target_u])


        # Get the shortest path
        res = [
            self.shortest_path[source_l][target_l] + source_d_start + target_d_start,
            self.shortest_path[source_l][target_u] + source_d_start + target_d_end,
            self.shortest_path[source_u][target_l] + source_d_end + target_d_start,
            self.shortest_path[source_u][target_u] + source_d_end + target_d_end
        ]
        return min(res)

def split_line(edge, point):
    edge_start, edge_end = edge.boundary.geoms

    if point == edge_start or point == edge_end:
        return gpd.GeoSeries([edge])

    line1 = shapely.geometry.LineString([edge_start, point])
    line2 = shapely.geometry.LineString([point, edge_end])

    return [line1, line2]

def snap(edges, point):
    """
    Snap a point to the nearest edge
    Args:
        edges (GeoSeries) : GeoSeries of LINESTRINGs
    Returns:
        (Point, GeoDataFrame) : snapped point, nearest edge (GeoDataFrame)
    """
    nidx = edges.sindex.nearest(point)[1][0]
    nedge = edges.loc[nidx]
    snap = nedge.geometry.interpolate(nedge.geometry.project(point))
    return snap, nedge

### K function on the linear network

def Ang_correction(interevent_dist, adj, dGmat, tk):
    # Ang's correction
    n = interevent_dist.shape[0]
    # For each event, get the number of perimeters using the interevent distance
    res = np.zeros_like(interevent_dist)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            r = interevent_dist[i, j]
            mask = tk[i] <= r
            if not np.any(mask):
                res[i,j] = 2
            _from = np.where(mask)[0]
            edges = set([tuple(sorted([u, v])) for u in _from for v in np.where(adj[u])[0]])
            for vk, vkp in edges:
                if tk[i, vkp] >= r:
                    res[i, j] += 1
                else:
                    c = dGmat[vk, vkp] - (r-tk[i, vkp]) - (r-tk[i, vk])
                    if c > 0:
                        res[i, j] += 2
                    elif c == 0:
                        res[i, j] += 1
    return res

def interevent_dist(points, linnet):
    n = points.shape[0]
    iedist = np.zeros((n, n))
    dGmat = linnet.shortest_path
    snapped = snap_points(linnet, points)
    d_start = snapped.d_start.values[:, np.newaxis]
    d_end = snapped.d_end.values[:, np.newaxis]
    edge_names = snapped.edge_name.values

    for i in range(n):
        for j in range(i+1, n):
            # If the points are on the same edge
            if edge_names[i] == edge_names[j]:
                iedist[i,j] = snapped.snap[i].distance(snapped.snap[j])
                continue

            iedist[i, j] = min(
                dGmat[snapped.node_start[i], snapped.node_start[j]] + d_start[i] + d_start[j],
                dGmat[snapped.node_start[i], snapped.node_end[j]] + d_start[i] + d_end[j],
                dGmat[snapped.node_end[i], snapped.node_start[j]] + d_end[i] + d_start[j],
                dGmat[snapped.node_end[i], snapped.node_end[j]] + d_end[i] + d_end[j]
            )
    iedist = iedist + iedist.T
    return iedist

def K_est_linnet_inhom(points, intensities, linnet, r_values, 
correction=True):
    """
    K function estimation for the linear network
    
    Source: Rakshit, S., et al. (2019). Efficient Code for Second Order Analysis of Events on a Linear Network.
    """
    n = points.shape[0]
    iedist = np.zeros((n, n))
    dGmat = linnet.shortest_path
    adj = linnet.adjacency
    snapped = snap_points(linnet, points)
    d_start = snapped.d_start.values[:, np.newaxis]
    d_end = snapped.d_end.values[:, np.newaxis]
    fromstart = dGmat[snapped.node_start, :] + d_start
    fromend = dGmat[snapped.node_end, :] + d_end
    tk = np.minimum(fromstart, fromend)

    for i in range(n):
        for j in range(i+1, n):
            # If the points are on the same edge
            if snapped.edge_name[i] == snapped.edge_name[j]:
                iedist[i,j] = snapped.snap[i].distance(snapped.snap[j])
                continue

            iedist[i, j] = min(
                dGmat[snapped.node_start[i], snapped.node_start[j]] + d_start[i] + d_start[j],
                dGmat[snapped.node_start[i], snapped.node_end[j]] + d_start[i] + d_end[j],
                dGmat[snapped.node_end[i], snapped.node_start[j]] + d_end[i] + d_start[j],
                dGmat[snapped.node_end[i], snapped.node_end[j]] + d_end[i] + d_end[j]
            )
    iedist = iedist + iedist.T
    

    if correction:
        # Ang's correction
        corrections = Ang_correction(interevent_dist=iedist, adj=adj, dGmat=dGmat, tk=tk)
        corrections[np.diag_indices(n)] = 1
    else:
        corrections = np.ones_like(intensities)

    int_outer = np.outer(intensities, intensities)
    corrections = corrections * int_outer
    np.fill_diagonal(iedist, np.inf)

    denom = np.sum(1/intensities)

    # Vectorized K calculation
    K_r = np.sum(
        (iedist[:, :, None] <= r_values) / corrections[:, :,None], axis=(0, 1)
    )
    K_r = K_r / denom

    return K_r

def K_est_linnet_hom(points, linnet, r_values, 
correction=True):
    """
    K function estimation for the linear network (homogeneous intensity)
    
    Source: Rakshit, S., et al. (2019). Efficient Code for Second Order Analysis of Events on a Linear Network.
    """
    n = points.shape[0]
    iedist = np.zeros((n, n))
    dGmat = linnet.shortest_path
    adj = linnet.adjacency
    snapped = snap_points(linnet, points)
    d_start = snapped.d_start.values[:, np.newaxis]
    d_end = snapped.d_end.values[:, np.newaxis]
    fromstart = dGmat[snapped.node_start, :] + d_start
    fromend = dGmat[snapped.node_end, :] + d_end
    tk = np.minimum(fromstart, fromend)
    snaps = snapped.snap.values

    for i in range(n):
        for j in range(i+1, n):
            # If the points are on the same edge
            if snapped.edge_name[i] == snapped.edge_name[j]:
                iedist[i,j] = snaps[i].distance(snaps[j])
                continue

            iedist[i, j] = min(
                dGmat[snapped.node_start[i], snapped.node_start[j]] + d_start[i] + d_start[j],
                dGmat[snapped.node_start[i], snapped.node_end[j]] + d_start[i] + d_end[j],
                dGmat[snapped.node_end[i], snapped.node_start[j]] + d_end[i] + d_start[j],
                dGmat[snapped.node_end[i], snapped.node_end[j]] + d_end[i] + d_end[j]
            )
    iedist = iedist + iedist.T

    if correction:
        # Ang's correction
        corrections = Ang_correction(interevent_dist=iedist, adj=adj, dGmat=dGmat, tk=tk)
        corrections[np.diag_indices(n)] = 1
    else:
        corrections = np.ones((n, n))

    np.fill_diagonal(iedist, np.inf)

    nom = linnet.edges.mm_len.sum() / n / (n-1)

    # Vectorized K calculation
    K_r = np.sum(
        (iedist[:, :, None] <= r_values) / corrections[:, :,None], axis=(0, 1)
    )
    K_r = K_r * nom

    return K_r