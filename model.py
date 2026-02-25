import time
import pandas as pd
import numpy as np
pd.set_option("display.max_rows", None)
pd.set_option("display.float_format", "{:.6f}".format)

IO_FEATURES = ["Flow Packets/s", "Flow Bytes/s", "Average Packet Size"]
OUT_FEATURES = ["Total Fwd Packet", "Fwd Packets/s"]  # "Dst Port",
IN_FEATURES = ["Total Bwd packets", "Bwd Packets/s"]  # "Src Port",
ALL_IP_FEATURES = IO_FEATURES + OUT_FEATURES + IN_FEATURES

start = time.perf_counter()
data = pd.read_csv("./thursday.csv")
after_read = time.perf_counter()
print(f"Dataset read time: {after_read-start:.6f}s")


def preprocess_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.dropna()
    df = df.drop(columns=[c for c in df.columns if df[c].nunique() == 1])
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    split_cols = ["id", "Flow ID", "Attempted Category", "Label"]
    #print(df.info())
    #print(df.mean(numeric_only=True))
    return (df.drop(columns=split_cols), df.loc[:, split_cols])

def display_ip_io_fv(fv: np.ndarray):
    width = 20
    print("".join(f"{h:<{width}}" for h in ALL_IP_FEATURES))
    print("".join(f"{v:<{width}.3f}" for v in np.round(fv, 3)))


class FlowNode:
    def __init__(self, fv: np.ndarray, feature_len: int, src: str, dst: str):
        self._src: str = src
        self._dst: str = dst
        self._fv: np.ndarray = fv
        self._degree: int = 0
        self._mean_fv: np.ndarray = np.zeros(feature_len, dtype=np.float64)

    def update_fv(self, x: np.ndarray):
        self._degree += 1
        self._mean_fv += (x - self._mean_fv) / self._degree

    def structure(self):
        print(f"Src: {self._src}")
        print(f"Dst: {self._dst}")
        print(self._fv)



class IpNode:
    def __init__(self, ip: str):
        self._ip: str = ip
        self._degree: int = 0
        self._mean_fv: np.ndarray = np.zeros(len(ALL_IP_FEATURES), dtype=np.float64)

    def update_fv(self, x: np.ndarray):
        self._degree += 1
        self._mean_fv += (x - self._mean_fv) / self._degree

    def structure(self):
        print(f"\nIP: {self._ip}")
        display_ip_io_fv(self._mean_fv) 


    def mean_feature_vector(self):
        return self._mean_fv

class IpGraph:
    def __init__(self):
        self.ip_map: dict[str, int] = {}
        self._nodes: list[IpNode] = []
        self._edges: list[set[int]] = []

    def swap_io_features(self, features: np.ndarray) -> np.ndarray:
        OUT = slice(len(IO_FEATURES), len(IO_FEATURES) + len(OUT_FEATURES))
        IN = slice(len(IO_FEATURES) + len(OUT_FEATURES), len(ALL_IP_FEATURES))
        swapped = features.copy()
        swapped[OUT], swapped[IN] = swapped[IN].copy(), swapped[OUT].copy()
        return swapped

    def get_id_from_ip(self, ip: str) -> int:
        if ip not in self.ip_map:
            id = len(self._nodes)
            self._nodes.append(IpNode(ip))
            self._edges.append(set())
            self.ip_map[ip] = id
        return self.ip_map[ip]

    def add_edge(self, src: str, dst: str, features: np.ndarray):
        src_id = self.get_id_from_ip(src)
        dst_id = self.get_id_from_ip(dst)
        self._edges[src_id].add(dst_id)
        self._edges[dst_id].add(src_id)
        self._nodes[src_id].update_fv(features)
        inverted_features = self.swap_io_features(features)
        #print("" + "".join(f"{v:<{20}}" for v in np.round(features, 3)))
        #print("" + "".join(f"{v:<{20}}" for v in np.round(inverted_features, 3)))
        self._nodes[dst_id].update_fv(inverted_features)

    def initialize(self, data: pd.DataFrame):
        features = data[ALL_IP_FEATURES].to_numpy(dtype=np.float64)
        flow_ips = data[["Src IP", "Dst IP"]].to_numpy()
        start = time.perf_counter()
        for i in range(len(features)):
            #print(f"src: {flow_ips[i, 0]}, dst: {flow_ips[i, 1]}")
            self.add_edge(flow_ips[i, 0], flow_ips[i, 1], features[i])
        end = time.perf_counter()
        print(f"Graph built in {end-start:.6f}s")

    def structure(self):
        print("Nodes: ", len(self._nodes))
        edges = sum([len(edge) for edge in self._edges])
        print("Edges: ", edges/2)

    def get_neighbours_by_ip(self, ip: str) -> set[int]:
        id = self.ip_map[ip]
        return self._edges[id]

    def get_neighbours_by_id(self, id: int) -> set[int]:
        if id > len(self._nodes)-1:
            raise ValueError("id does not exist")
        return self._edges[id]

    def structure_of(self, id: int):
        if id > len(self._nodes)-1:
            raise ValueError("id does not exist")
        self._nodes[id].structure()

# residual = observed - predicted
    def find_anomalies(self):
        for id in self.ip_map.values():
            if id > 10:
                break
            mean_neighbour_features: np.ndarray = np.zeros(len(ALL_IP_FEATURES), dtype=np.float64)
            node = self._nodes[id]
            edges = self._edges[id]
            for (edge, count) in enumerate(edges, start=1):
                mean_neighbour_features += (self._nodes[edge].mean_feature_vector() - mean_neighbour_features) / count

            print("\n\nNEIGHBOURS")
            display_ip_io_fv(mean_neighbour_features)
            residual = node.mean_feature_vector() - mean_neighbour_features
            print("RESIDUAL")
            display_ip_io_fv(residual)





clean_data, unused = preprocess_data(data)

graph = IpGraph()
graph.initialize(clean_data)

graph.structure()
graph.structure_of(2)
print(graph.get_neighbours_by_id(2))
graph.find_anomalies()
