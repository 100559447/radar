# Required imports
import numpy as np
from Location import Location
from Boundaries import Boundaries
from Radar import Radar
from tqdm import tqdm
import networkx as nx

# Constant that avoids setting cells to have an associated cost of zero
EPSILON = 1e-4


class Map:
    """ Class that models the map for the simulation """

    def __init__(self,
                 boundaries: Boundaries,
                 height: np.int32,
                 width: np.int32,
                 radars: np.array = None):
        self.boundaries = boundaries  # Boundaries of the map
        self.height = height  # Number of coordinates in the y-axis
        self.width = width  # Number of coordinates int the x-axis
        self.radars = radars  # List containing the radars (objects)

    def generate_radars(self, n_radars: np.int32) -> None:
        """ Generates n-radars randomly and inserts them into the radars list """
        # Select random coordinates inside the boundaries of the map
        lat_range = np.linspace(start=self.boundaries.min_lat, stop=self.boundaries.max_lat, num=self.height)
        lon_range = np.linspace(start=self.boundaries.min_lon, stop=self.boundaries.max_lon, num=self.width)
        rand_lats = np.random.choice(a=lat_range, size=n_radars, replace=False)
        rand_lons = np.random.choice(a=lon_range, size=n_radars, replace=False)
        self.radars = []  # Initialize 'radars' as an empty list

        # Loop for each radar that must be generated
        for i in range(n_radars):
            # Create a new radar
            new_radar = Radar(location=Location(latitude=rand_lats[i], longitude=rand_lons[i]),
                              transmission_power=np.random.uniform(low=1, high=1000000),
                              antenna_gain=np.random.uniform(low=10, high=50),
                              wavelength=np.random.uniform(low=0.001, high=10.0),
                              cross_section=np.random.uniform(low=0.1, high=10.0),
                              minimum_signal=np.random.uniform(low=1e-10, high=1e-15),
                              total_loss=np.random.randint(low=1, high=10),
                              covariance=None)

            # Insert the new radar
            self.radars.append(new_radar)
        return

    def get_radars_locations_numpy(self) -> np.array:
        """ Returns an array with the coordiantes (lat, lon) of each radar registered in the map """
        locations = np.zeros(shape=(len(self.radars), 2), dtype=np.float32)
        for i in range(len(self.radars)):
            locations[i] = self.radars[i].location.to_numpy()
        return locations

    def compute_detection_map(self) -> np.array:
        """ Computes the detection probability map using radar signals and MinMax normalization """
        detection_map = np.zeros((self.height, self.width), dtype=np.float32)

        # 1. Generate coordinate grid
        lat_range = np.linspace(start=self.boundaries.min_lat, stop=self.boundaries.max_lat, num=self.height)
        lon_range = np.linspace(start=self.boundaries.min_lon, stop=self.boundaries.max_lon, num=self.width)

        # 2. Compute max detection level for each cell
        for i in range(self.height):
            for j in range(self.width):
                lat = lat_range[i]
                lon = lon_range[j]

                # Compute Ψ* (max detection level from all radars)
                max_detection = max(radar.compute_detection_level(lat, lon) for radar in self.radars)
                detection_map[i, j] = max_detection

        # 3. Apply MinMax scaling with ε
        psi_min = detection_map.min()
        psi_max = detection_map.max()

        if psi_max == psi_min:
            scaled = np.full_like(detection_map, fill_value=EPSILON)
        else:
            scaled = ((detection_map - psi_min) / (psi_max - psi_min)) * (1 - EPSILON) + EPSILON

        return scaled

    def build_graph(self, prob_map: np.array, threshold: float) -> nx.DiGraph:
        """ Builds a directed graph where nodes are valid grid cells and edges connect adjacent cells """
        G = nx.DiGraph()
        H, W = prob_map.shape

        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right

        for i in range(H):
            for j in range(W):
                if prob_map[i, j] <= threshold:
                    G.add_node((i, j))

                    # Try to connect to each neighbor
                    for di, dj in directions:
                        ni, nj = i + di, j + dj

                        # Check bounds
                        if 0 <= ni < H and 0 <= nj < W:
                            if prob_map[ni, nj] <= threshold:
                                G.add_edge((i, j), (ni, nj), weight=prob_map[ni, nj])

        return G