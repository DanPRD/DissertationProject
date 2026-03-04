import time
import pandas as pd
import numpy as np
from collections import defaultdict
import pickle
import matplotlib.pyplot as plt
import scipy.stats as stats
pd.set_option("display.max_rows", None)
pd.set_option("display.float_format", "{:.6f}".format)

IO_FEATURES = ["Flow Packets/s", "Flow Bytes/s", "Average Packet Size"]
OUT_FEATURES = ["Total Fwd Packet", "Fwd Packets/s"]  # "Dst Port",
IN_FEATURES = ["Total Bwd packets", "Bwd Packets/s"]  # "Src Port",
ALL_IP_FEATURES = IO_FEATURES + OUT_FEATURES + IN_FEATURES

TIME_BIN_SIZE_S = 0.3
TIME_BIN_SIZE = TIME_BIN_SIZE_S * 1000000
ENERGY_THRESHOLD = 1

start = time.perf_counter()
data = pd.read_csv("./thursday.csv")
after_read = time.perf_counter()
print(f"Dataset read time: {after_read-start:.6f}s")

# 'Packet Length Mean' and 'Average Packet Size' seem to just be the same value with very minor changes, like 0.01 difference
def preprocess_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    df = df.dropna()
    df = df.drop(columns=[c for c in df.columns if df[c].nunique() == 1])
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    split_cols = ["id", "Flow ID", "Attempted Category", "Label", "Src IP", "Dst IP", "Timestamp"]

    identification_data = df[split_cols].copy()

    df = df.drop(columns=split_cols)
    benign = df[(identification_data["Label"] == "BENIGN") | (identification_data["Attempted Category"] != -1)]
    identification_data = identification_data.set_index("id")

    mins = benign.min()
    denom = (benign.max() - mins).replace(0, 1)
    df = (df - mins) / denom
    # original = scaled * denom + mins
    print(df.info())
    print(df.mean(numeric_only=True))
    return (df, identification_data, mins, denom)

def display_ip_io_fv(fv: np.ndarray):
    width = 20
    print("".join(f"{h:<{width}}" for h in ALL_IP_FEATURES))
    print("".join(f"{v:<{width}.3f}" for v in np.round(fv, 3)))



def to_microseconds(timestamp: str) -> int:
    t = timestamp[11:]
    return (
        int(t[0:2]) * 3_600_000_000 +
        int(t[3:5]) *   60_000_000 +
        int(t[6:8]) *    1_000_000 +
        int(t[9:15])
    )


class FlowNode:
    def __init__(self, flow_id: int, fv: np.ndarray, src: str, dst: str):
        self.flow_id: int = flow_id
        self.src: str = src
        self.dst: str = dst
        self.fv: np.ndarray = fv
        self.degree: int = 0
        self.mean_fv: np.ndarray = np.zeros(len(fv), dtype=np.float64)


    def update_fv(self, x: np.ndarray):
        self.degree += 1
        self.mean_fv += (x - self.mean_fv) / self.degree

    def structure(self):
        print(f"Src: {self.src}")
        print(f"Dst: {self.dst}")
        for (header, feature, mean_feature) in zip(feature_idx_map, self.fv, self.mean_fv):
            print(f"{header}: {feature} ; {mean_feature}")


class FlowGraph:
    def __init__(self, feature_len: int, timestart: int):
        self._feature_len: int = feature_len
        self._nodes: dict[int, FlowNode] = {}
        self._edges: defaultdict[int, set[int]] = defaultdict(set)
        self._src_grp: defaultdict[str, set[int]] = defaultdict(set)
        self._dst_grp: defaultdict[str, set[int]] = defaultdict(set)
        self._fullmatch_grp: defaultdict[tuple[str, str, int, float], set[int]] = defaultdict(set)
        self._timestart = timestart


    def insert_flow(self, flow_id: int, src: str, dst: str, timestamp: str, features: np.ndarray, ):
        if flow_id not in self._nodes:

            self._nodes[flow_id] = FlowNode(flow_id, features, src, dst)
            self._src_grp[src].add(flow_id)
            self._dst_grp[dst].add(flow_id)
            time_bin = (to_microseconds(timestamp) - self._timestart) // TIME_BIN_SIZE
            self._fullmatch_grp[(src, dst, int(features[feature_idx_map["Protocol"]]), time_bin)].add(flow_id)

    def find_edges(self):
        start = time.perf_counter()

        for (key, edge_set) in self._fullmatch_grp.items():
            #print(f"{key}: ({len(edge_set)})")
            for flow_id in edge_set:
                new_edges = edge_set.difference({flow_id})
                self._edges[flow_id] = self._edges[flow_id].union(new_edges)
                for edge_id in new_edges:
                    self._nodes[flow_id].update_fv(self._nodes[edge_id].fv)
        after_read = time.perf_counter()
        print(f"Find edges time: {after_read-start:.6f}s")

    def initialize(self, data: pd.DataFrame, ids: pd.DataFrame):
        features = data.to_numpy(dtype=np.float64)
        flow_data = ids[["id", "Src IP", "Dst IP", "Timestamp"]].to_numpy()
        start = time.perf_counter()
        for i in range(len(features)):
            self.insert_flow(flow_data[i, 0], flow_data[i, 1], flow_data[i, 2], flow_data[i, 3], features[i])

        end = time.perf_counter()
        print(f"Graph built in {end-start:.6f}s")
        self.structure()
        print(f"Finding Edges")
        self.find_edges()

    def structure(self):
        print("Nodes: ", len(self._nodes))
        edges = sum([len(edge) for edge in self._edges.values()])
        print("Edges: ", edges/2)
        maxn = 0
        minn = 99999999
        for l in self._fullmatch_grp.values():
            maxn = max(maxn, len(l))
            minn = min(minn, len(l))
        print(f"dst:{len(self._dst_grp)}\nsrc:{len(self._src_grp)}\nfull:{len(self._fullmatch_grp)} max:{maxn} min:{minn}")

    def get_neighbours_by_id(self, id: int) -> set[int]:
        if id > len(self._nodes)-1:
            raise ValueError("id does not exist")
        return self._edges[id]

    def structure_of(self, id: int):
        if id not in self._nodes:
            raise ValueError("id does not exist")
        print(f"FLOW ID {id}--------------------")
        print(f"Edges ({self._nodes[id].degree}): {self._edges[id]}")
        self._nodes[id].structure()

    def compare(self, lhs_id, rhs_id):
        print(f"COMPARISON----------------------")
        for (header, lhs, rhs) in zip(feature_idx_map, self._nodes[lhs_id].fv, self._nodes[rhs_id].fv):
            print(f"{header}: {lhs} - {rhs}")


    def find_anomalies(self, identification_data: pd.DataFrame):
        start = time.perf_counter()
        print("finding anomalies")

        benign_residuals = []

        num_malicious = 0
        for id, node in self._nodes.items():
            label = identification_data.loc[id, "Label"]
            if (label == "BENIGN") | ("Attempted" in label):
                residual = node.fv - node.mean_fv
                benign_residuals.append(residual)
            else:
                num_malicious += 1

        num_benign = len(benign_residuals)




        R_benign = np.array(benign_residuals)



        #feature = R_benign[:, feature_idx_map["Flow Duration"]]
        #print(len(feature))

        #plt.hist(feature, bins=50, density=True)
        #plt.savefig("Fig.png")

        feature_var = R_benign.var(axis=0)
        feature_var[feature_var == 0] = 1e-8 # no division by 0



        anomalies: list[tuple[int, int]] = []
        benign: list[tuple[int, int]] = []
        for id in self._nodes.keys():
            node = self._nodes[id]
            edges = self._edges[id]
            if len(edges) == 0:
                continue


            residual = node.fv - node.mean_fv
            energy = np.sum((residual**2) / feature_var)
            #print("RESIDUAL------------")
            #print(f"ENERGY: {energy}")
            #print(f"SRC: {node.src}")
            #print(f"DST: {node.dst}")
            #for (header, feature) in zip(feature_idx_map, residual):
                #print(f"{header}: {feature}")
            if (energy > ENERGY_THRESHOLD):
                anomalies.append((id, energy))
            else:
                benign.append((id, energy))
        fp = 0
        fn = 0

        for (flowid, energy) in anomalies:
            label: str = identification_data.loc[flowid, "Label"]
            if (label == "BENIGN") or ("Attempted" in label):
                fp += 1
        for (flowid, energy) in benign:
            label: str = identification_data.loc[flowid, "Label"]
            if (label != "BENIGN") and ("Attempted" not in label):
                fn += 1
            #print(f"anom: {flowid} ({energy}) :: {label}")
        print(f"predicited benign, actually malicious (FN): {fn}")
        print(f"predicited malicious, actually benign (FP): {fp}")
        print(f"Total Incorrect: {fn + fp}")
        print(f"Total Correct: {len(anomalies) + len(benign) - fp - fn}")
        print(f"Total predicted anomalies (mal): {len(anomalies)}")
        print(f"Total predicted benign: {len(benign)}")
        print(f"Total Nodes: {len(self._nodes)}")
        after_read = time.perf_counter()
        print(f"Find anomalies time: {after_read-start:.6f}s")






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

    def initialize(self, data: pd.DataFrame, ids: pd.DataFrame):
        features = data[ALL_IP_FEATURES].to_numpy(dtype=np.float64)
        flow_ips = ids[["Src IP", "Dst IP"]].to_numpy()
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





clean_data, identification_data, mins, denoms = preprocess_data(data)


feature_idx_map = {}
for (idx, feature_name) in enumerate(clean_data.columns.values):
    feature_idx_map[feature_name] = idx
timestart = to_microseconds(identification_data["Timestamp"].iloc[0])
graph: FlowGraph;

load_from_file = True
if (load_from_file):
    file = open("graph.bin", "rb")
    graph = pickle.load(file)
else:
    graph = FlowGraph(len(clean_data), timestart)
    graph.initialize(clean_data, identification_data)
    pickle.dump(graph, open("graph.bin", "wb"), protocol=pickle.HIGHEST_PROTOCOL)


check = 362075
#print(f"edges: {len(graph._edges[check])}")
#graph.compare(check, graph._edges[check].pop())
#graph._nodes[check].structure()
graph.find_anomalies(identification_data)
