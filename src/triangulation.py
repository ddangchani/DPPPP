import numpy as np
import matplotlib.tri as mtri
from shapely.geometry import Polygon, MultiPolygon
import pymc as pm
from pytensor.compile.ops import as_op
import pytensor.tensor as pt

def create_triangulation(nx, ny, window=[0, 1, 0, 1]):
    """
    Create a triangulation object for a regular grid.

    Parameters
    ----------
    nx : int
        Number of grid points in x direction.
    ny : int
        Number of grid points in y direction.
    window : [xmin, xmax, ymin, ymax]
        Window of the grid (default is [0, 1, 0, 1]).

    Returns
    -------
    triang : matplotlib.tri.Triangulation object
        Triangulation object for the grid.
    """
    # Create grid points
    xmin, xmax, ymin, ymax = window
    x = np.linspace(xmin, xmax, nx) 
    y = np.linspace(ymin, ymax, ny)
    x, y = np.meshgrid(x, y)
    x = x.flatten()
    y = y.flatten()

    # Create triangles
    triangles = []
    for i in range(ny - 1):
        for j in range(nx - 1):
            p1 = i * nx + j
            p2 = p1 + 1
            p3 = p1 + nx
            p4 = p3 + 1
            # Add two triangles for each square
            triangles.append([p1, p2, p4])
            triangles.append([p1, p4, p3])

    # Create triangulation object
    triang = mtri.Triangulation(x, y, triangles)

    return triang

def calculate_centroids(mesh):
    x = mesh.x
    y = mesh.y
    centroids = np.array([
        [x[tri].mean(), y[tri].mean()] for tri in mesh.triangles
    ])
    return centroids

def get_dual_points(mesh, index): 
    """Returns the points of the dual mesh nearest to the point in the mesh given by the index.

    Parameters:
        mesh: matplotlib.tri.Triangulation
            Input mesh.
        index: int
            Index of the point in the input mesh for which to calculate the nearest points of the dual mesh.
    """
    # Find the cells where the given index appears
    _idxs = np.where(mesh.triangles == index)[0]
    # Find the centroids of all the cells of _idxs
    centroids = calculate_centroids(mesh)
    # Find the centroids of the cells where the given index appears
    _centroids = centroids[_idxs]
    return _centroids

def get_dual_edges_boundary(mesh, index, window=[0, 1, 0, 1]):
    # Select edges containing the given index
    edges_with_index = mesh.edges[np.any(mesh.edges == index, axis=1)]
    midpoints = (mesh.x[edges_with_index[:, 0]] + mesh.x[edges_with_index[:, 1]]) / 2
    midpoints = np.vstack((
        midpoints,
        (mesh.y[edges_with_index[:, 0]] + mesh.y[edges_with_index[:, 1]]) / 2
    )).T

    return midpoints[(midpoints[:, 0] == window[0]) | (midpoints[:, 0] == window[1]) | (midpoints[:, 1] == window[2]) | (midpoints[:, 1] == window[3])] 

# Return only the midpoints that are on the boundary

def get_dual(mesh, window=[0, 1, 0, 1]):
    """
    Returns the dual mesh of the input mesh.

    Parameters:
        mesh: matplotlib.tri.Triangulation
            Input mesh.

    Returns:
        MultiPolygon
            Voronoi dual mesh.
    """
    ## For each type of cell do the following
    # Find the centroids of all the triangles
    centroids = calculate_centroids(mesh)
    # Create the dual cells
    dual_cells = []

    for i, pt in enumerate(zip(mesh.x, mesh.y)):
        dual_pts = get_dual_points(mesh, i)
        # if point is one of the 4 corners, add itself to the dual points
        if pt[0] in [window[0], window[1]] and pt[1] in [window[2], window[3]]:
            dual_pts = np.vstack((dual_pts, pt))
        if len(dual_pts) < 4:
            # Add the midpoints of the edges that are on the boundary
            dual_pts = np.vstack((dual_pts, get_dual_edges_boundary(mesh, i, window)))
        # Connect the centroids
        dual_cells.append(Polygon(dual_pts).convex_hull)

    return MultiPolygon(dual_cells)

def proj(points, mesh):
    """
    Compute the basis function values at the points.

    Parameters
    ----------
    points : numpy.ndarray
        Points at which to compute the basis functions.
    mesh : matplotlib.tri.Triangulation
        Triangulation object for the grid.

    Returns
    -------
    basis : numpy.ndarray
        Basis function matrix (N x n) where N is the number of points and n is the number of grid points.
    """
    basis = np.zeros((points.shape[0], mesh.x.shape[0]))
    for i in range(mesh.x.shape[0]):
        gridvals = np.zeros_like(mesh.x)
        gridvals[i] = 1
        interp = mtri.LinearTriInterpolator(mesh, gridvals)
        basis[:, i] = interp(points[:, 0], points[:, 1])

    return basis

def get_areas(multipolygon):
    # Check if input is MultiPolygon
    if not isinstance(multipolygon, MultiPolygon):
        raise TypeError("Input must be a MultiPolygon object")
    
    # Extract areas from each polygon in the MultiPolygon
    areas = np.array([polygon.area for polygon in multipolygon.geoms])
    
    return areas

def loglik(observation, mesh, weights):
    """
    Compute the log-likelihood of the triangulation model.

    Parameters
    ----------
    observation : numpy.ndarray
        Observation data.
    mesh : matplotlib.tri.Triangulation
        Triangulation object for the grid.
    weights : numpy.ndarray
        Weights for the vertices of the grid.

    Returns
    -------
    loglik : float
        Log-likelihood of the model.
    """
    # Get the number of basis functions
    n_basis = mesh.x.shape[0]

    # Project the observation data
    P = proj(observation, mesh)

    # Get the dual mesh
    dual_mesh = get_dual(mesh)

    # Get the areas of the dual mesh
    alpha_0 = get_areas(dual_mesh)

    # Construct the vectors for the log-likelihood
    n_obs = observation.shape[0]
    y = np.hstack([np.zeros(n_basis), np.ones(n_obs)]).reshape(1, -1)
    alpha = np.hstack([alpha_0, np.zeros(n_obs)]).reshape(1, -1)
    logeta = np.hstack([weights, weights @ P.T]).reshape(1, -1)

    # Compute the log-likelihood
    loglik = np.sum(y * logeta - alpha * np.exp(logeta))

    return loglik

def tessellation(nx, ny, window):
    xmin, xmax, ymin, ymax = window
    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)
    # Return MultiPolygon
    return MultiPolygon([Polygon([(x[i], y[j]), (x[i+1], y[j]), (x[i+1], y[j+1]), (x[i], y[j+1])]) for i in range(nx-1) for j in range(ny-1)])

def get_dual_rec(window, nx, ny):
    # Get dual rectangles
    xmin, xmax, ymin, ymax = window
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    dx = xs[1] - xs[0]
    dy = ys[1] - ys[0]
    xs[1:] -= dx / 2
    ys[1:] -= dy / 2
    xs = np.append(xs, xmax)
    ys = np.append(ys, ymax)

    return MultiPolygon([Polygon([(xs[i], ys[j]), (xs[i+1], ys[j]), (xs[i+1], ys[j+1]), (xs[i], ys[j+1])]) for i in range(nx) for j in range(ny)])
