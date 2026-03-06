import time
import pandas as pd
import numpy as np
from collections import defaultdict
import pickle
import matplotlib.pyplot as plt
import scipy.stats as stats
import matplotlib.gridspec as gridspec
from sklearn.metrics import roc_curve, auc, f1_score
pd.set_option("display.max_rows", None)
pd.set_option("display.float_format", "{:.6f}".format)

IO_FEATURES = ["Flow Packets/s", "Flow Bytes/s", "Average Packet Size"]
OUT_FEATURES = ["Total Fwd Packet", "Fwd Packets/s"]  # "Dst Port",
IN_FEATURES = ["Total Bwd packets", "Bwd Packets/s"]  # "Src Port",
ALL_IP_FEATURES = IO_FEATURES + OUT_FEATURES + IN_FEATURES

TIME_BIN_SIZE_S = 1.5
TIME_BIN_SIZE = TIME_BIN_SIZE_S * 1000000
FEATURE_CLIP_MAX = 10
ALPHA = 0 # weight of local signal 0 = global only 1 = local only
MIN_EDGES = 3 
ENERGY_THRESHOLD_SCALAR = 1.7

start = time.perf_counter()
df_monday = pd.read_csv("./monday.csv")
data = pd.read_csv("./thursday.csv")
after_read = time.perf_counter()
print(f"Dataset read time: {after_read-start:.6f}s")

# 'Packet Length Mean' and 'Average Packet Size' seem to just be the same value with very minor changes, like 0.01 difference
def process_benign_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    df = df.dropna()
    #df = df.drop(columns=[c for c in df.columns if df[c].nunique() == 1])
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    #df.loc[df["Attempted Category"] != -1, "Label"] = "BENIGN"

    split_cols = ["id", "Flow ID", "Attempted Category", "Label", "Src IP", "Dst IP", "Timestamp"]

    identification_data = df[split_cols].copy()
    identification_data = identification_data.set_index("id", drop=False)

    df = df.drop(columns=split_cols)

    mins = df.min() 
    denom = (df.max() - mins).replace(0, 1)
    df = (df - mins) / denom # original = scaled * denom + mins 
    print(df.mean(numeric_only=True)) 
    return (df, identification_data, mins, denom)


# for ATTACK DATA
def process_test_data(df: pd.DataFrame, mins: pd.Series, denom: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.dropna()
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df.loc[df["Attempted Category"] != -1, "Label"] = "BENIGN"

    split_cols = ["id", "Flow ID", "Attempted Category", "Label", "Src IP", "Dst IP", "Timestamp"]

    identification_data = df[split_cols].copy()
    identification_data = identification_data.set_index("id", drop=False)


    df = df.drop(columns=split_cols)
    df = df.reindex(columns=mins.index, fill_value=0.0)


    df = (df - mins) / denom
    df = df.clip(lower=-FEATURE_CLIP_MAX, upper=FEATURE_CLIP_MAX)  
    print(df.mean(numeric_only=True)) 
    return (df, identification_data)



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


class BaselineGraph:
    def __init__(self, feature_len: int):
        self._feature_len = feature_len
        self._dst_fvs: defaultdict[str, list[np.ndarray]] = defaultdict(list)
        self._src_fvs: defaultdict[str, list[np.ndarray]] = defaultdict(list)
        self.dst_mean: dict[str, np.ndarray] = {}
        self.dst_var: dict[str, np.ndarray] = {}
        self.src_mean: dict[str, np.ndarray] = {}
        self.global_mean: np.ndarray = np.zeros(feature_len)
        self.global_var: np.ndarray = np.ones(feature_len)
        self.energy_threshold: float = 0.0

    def add_flow(self, src: str, dst: str, features: np.ndarray):
        self._dst_fvs[dst].append(features)
        self._src_fvs[src].append(features)

    def build(self, threshold_percentile: float = 99.5):
        all_fvs = np.vstack(list(self._dst_fvs.values()))
        self.global_mean = all_fvs.mean(axis=0)
        global_v = all_fvs.var(axis=0)
        global_v[global_v == 0] = 1e-4
        self.global_var = global_v

        # per-dst variance, keeping atleast 1% variance for features that have zero
        for dst, fvs in list(self._dst_fvs.items()):
            arr = np.array(fvs)
            self.dst_mean[dst] = arr.mean(axis=0)
            v = arr.var(axis=0)
            v = np.maximum(v, self.global_var * 0.01)
            self.dst_var[dst] = v

        for src, fvs in list(self._src_fvs.items()):
            self.src_mean[src] = np.array(fvs).mean(axis=0)


        benign_energies = []
        for dst in list(self.dst_mean.keys()):  
            mean = self.dst_mean[dst]
            var = self.dst_var[dst]
            for fv in self._dst_fvs[dst]:
                e = np.sum((fv - mean) ** 2 / var)
                benign_energies.append(e)

        log_energies = np.log10(np.array(benign_energies) + 1e-12)
        mu = log_energies.mean()
        std = log_energies.std()
 
        self.energy_threshold = 10 ** (mu + ENERGY_THRESHOLD_SCALAR * std)

        print(f"baseline built: {len(all_fvs)} flows, {len(self.dst_mean)} unique dsts")
        print(f"monday log-energy: mu={mu:.2f}, std={std:.2f}")
        print(f"energy threshold (mu+{ENERGY_THRESHOLD_SCALAR}σ): {self.energy_threshold:.2f}")

    def global_energy(self, fv: np.ndarray, dst: str) -> float:
        mean = self.dst_mean.get(dst, self.global_mean)
        var = self.dst_var.get(dst, self.global_var)

        # var must be 1% of global
        var_floored = np.maximum(var, self.global_var * 0.01)

        return float(np.sum((fv - mean) ** 2 / var_floored))

    def initialize_from_df(self, data: pd.DataFrame, ids: pd.DataFrame):
        features = data.to_numpy(dtype=np.float64)
        flow_data = ids[["Src IP", "Dst IP"]].to_numpy()
        for i in range(len(features)):
            self.add_flow(flow_data[i, 0], flow_data[i, 1], features[i])
        self.build()


class FlowGraph:
    def __init__(self, feature_len: int, timestart: int, baseline: BaselineGraph):
        self._feature_len: int = feature_len
        self._nodes: dict[int, FlowNode] = {}
        self._edges: defaultdict[int, set[int]] = defaultdict(set)
        self._src_grp: defaultdict[str, set[int]] = defaultdict(set)
        self._dst_grp: defaultdict[str, set[int]] = defaultdict(set)
        self._fullmatch_grp: defaultdict[tuple[str, str, int, float], set[int]] = defaultdict(set)
        self._timestart = timestart
        self._baseline = baseline


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
        #print(f"Finding Edges")
        #self.find_edges()

    def combined_energy(self, node: FlowNode) -> float:
        return self._baseline.global_energy(node.fv, node.dst)

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
        edge_counts = [len(self._edges[id]) for id in self._nodes]
        print(f"Nodes with 0 edges: {sum(1 for e in edge_counts if e == 0)}")
        print(f"Nodes with 1-3 edges: {sum(1 for e in edge_counts if 1 <= e <= 3)}")
        print(f"Median edges per node: {np.median(edge_counts)}")
        print(f"Mean edges per node: {np.mean(edge_counts):.2f}")

        # Diagnose WHO the zero-edge nodes are
        zero_edge_ids = [id for id in self._nodes if len(self._edges[id]) == 0]

        labels = test_identification.loc[zero_edge_ids, "Label"]
        print(f"Zero-edge nodes that are BENIGN: {(labels == 'BENIGN').sum()}")
        print(f"Zero-edge nodes that are MALICIOUS: {(labels != 'BENIGN').sum()}")

        zero_srcs = [self._nodes[id].src for id in zero_edge_ids]
        print(f"Unique src IPs in zero-edge nodes: {len(set(zero_srcs))}")
        print(f"Most common src IPs: {pd.Series(zero_srcs).value_counts().head(10)}")

        edge_counts = pd.Series({id: len(self._edges[id]) for id in self._nodes})
        mal_ids = test_identification[test_identification["Label"] != "BENIGN"].index
        ben_ids = test_identification[test_identification["Label"] == "BENIGN"].index

        print(f"\nBenign edge count stats:")
        print(edge_counts[edge_counts.index.isin(ben_ids)].describe())
        print(f"\nMalicious edge count stats:")
        print(edge_counts[edge_counts.index.isin(mal_ids)].describe())

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


    def find_anomalies(self, test_identification: pd.DataFrame):
        start = time.perf_counter()
        print("finding anomalies")

        all_energies = []
        true_is_malicious = []
        flow_ids = list(self._nodes.keys())

        for id, node in self._nodes.items():
            label = test_identification.at[id, "Label"]
            is_malicious = label != "BENIGN"
            energy = self.combined_energy(node)

            all_energies.append(energy)
            true_is_malicious.append(is_malicious)


        all_energies = np.array(all_energies)
        true_is_malicious = np.array(true_is_malicious)

        ENERGY_THRESHOLD = self._baseline.energy_threshold
        print(f"Energy Threshold: {ENERGY_THRESHOLD:.4f}")

        predictions = all_energies > ENERGY_THRESHOLD

        fp = int(( predictions & ~true_is_malicious).sum())
        fn = int((~predictions & true_is_malicious).sum())
        tp = int(( predictions & true_is_malicious).sum())
        tn = int((~predictions & ~true_is_malicious).sum())

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

        print(f"predicited benign, actually malicious (FN): {fn}")
        print(f"predicited malicious, actually benign (FP): {fp}")
        print(f"Total Incorrect: {fn + fp}")
        print(f"Total Correct: {tn + tp}")
        print(f"Total predicted anomalies (mal): {tp}")
        print(f"Total predicted benign: {tn}")
        print(f"Total Nodes: {len(self._nodes.keys())}")
        after_read = time.perf_counter()
        print(f"Find anomalies time: {after_read-start:.6f}s")


        # bunch of statistic stuff to help debug
        thresh_candidates = np.percentile(all_energies, np.linspace(0, 100, 1000))
        f1s = [f1_score(true_is_malicious, all_energies > t) for t in thresh_candidates]
        best_t = thresh_candidates[np.argmax(f1s)]
        best_f1 = max(f1s)
        print(f"F1-optimal threshold: {best_t:.4f}  F1={best_f1:.4f}  "
              f"(Monday-derived threshold F1={f1_score(true_is_malicious, predictions):.4f})")

        fpr_curve, tpr_curve, _ = roc_curve(true_is_malicious, all_energies)
        roc_auc = auc(fpr_curve, tpr_curve)

        self.plot(all_energies, true_is_malicious, ENERGY_THRESHOLD,
                   fpr_curve, tpr_curve, roc_auc, fpr, tpr)

        print(f"TP={tp}  FP={fp}  TN={tn}  FN={fn}")
        print(f"TPR={tpr:.3f}  FPR={fpr:.3f}")
        print(f"ROC AUC = {roc_auc:.4f}")
        print(f"find anomalies time: {time.perf_counter() - start:.6f}s")

        fp_ids = [flow_ids[i] for i in range(len(flow_ids))
                  if predictions[i] and not true_is_malicious[i]]

        fp_energies = all_energies[[i for i in range(len(flow_ids))
                                     if predictions[i] and not true_is_malicious[i]]]

        fp_dsts = pd.Series([graph._nodes[id].dst for id in fp_ids])
        print("FP flows by dst IP:")
        print(fp_dsts.value_counts().head(10))

        fp_dst_in_baseline = [graph._nodes[id].dst in baseline.dst_mean for id in fp_ids]
        print(f"\nFP flows whose destination was in monday baseline: {sum(fp_dst_in_baseline)}")
        print(f"FP flows whose destination was not in monday baseline: {sum(not x for x in fp_dst_in_baseline)}")

        print(f"\nFP energy stats:")
        print(f"  min:    {fp_energies.min():.1f}")
        print(f"  median: {np.median(fp_energies):.1f}")
        print(f"  max:    {fp_energies.max():.1f}")

        top_ip_to_check = str(fp_dsts.value_counts().idxmax())

        fp_ids_by_dst = [flow_ids[i] for i in range(len(flow_ids))
                         if not true_is_malicious[i] 
                         and predictions[i]
                         and graph._nodes[flow_ids[i]].dst == top_ip_to_check]

        if fp_ids_by_dst:
            fp_fvs = np.array([graph._nodes[id].fv for id in fp_ids_by_dst])
            dst_mean = baseline.dst_mean[top_ip_to_check]
            dst_var = baseline.dst_var[top_ip_to_check]

            residuals = fp_fvs - dst_mean
            per_feature_energy = (residuals**2 / dst_var).mean(axis=0)
            top_features = np.argsort(per_feature_energy)[::-1][:10]

            feature_names = list(test_data.columns)
            print(f"top energy-contributing features for {top_ip_to_check} FPs:")
            for idx in top_features:
                print(f"  {feature_names[idx]}: mean_contribution={per_feature_energy[idx]:.1f} fp_mean={fp_fvs[:,idx].mean():.4f}  baseline_mean={dst_mean[idx]:.4f}")

    def plot(self, all_energies, true_is_malicious, threshold, fpr_curve, tpr_curve, roc_auc, fpr_op, tpr_op):
        benign_energy = all_energies[~true_is_malicious]
        malicious_energy = all_energies[true_is_malicious]

        fig = plt.figure(figsize=(16, 12))
        gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.3)

        ax1 = fig.add_subplot(gs[0, 0])
        ax1.hist(benign_energy, bins=100, alpha=0.6, color="blue", label=f"benign (n={len(benign_energy)})")
        ax1.hist(malicious_energy, bins=100, alpha=0.6, color="red", label=f"malicious (n={len(malicious_energy)})")
        ax1.axvline(threshold, color="black", linestyle="--", label=f"threshold = {threshold:.2f}")
        ax1.set_xscale("log")
        ax1.set_yscale("log")
        ax1.set_xlabel("energy")
        ax1.set_ylabel("count")
        ax1.set_title("energy distribution (log-log)"); ax1.legend()

        ax2 = fig.add_subplot(gs[0, 1])
        for e, c, lbl in [(benign_energy, "blue", "benign"), (malicious_energy, "red", "malicious")]:
            s = np.sort(e)
            ax2.plot(s, np.arange(1, len(s)+1)/len(s), color=c, label=lbl)

        ax2.axvline(threshold, color="black", linestyle="--")
        ax2.set_xscale("log"); ax2.set_xlabel("Energy")
        ax2.set_ylabel("cumulative proportion")
        ax2.set_title("ECDF"); ax2.legend()

        ax3 = fig.add_subplot(gs[1, 0])
        ax3.violinplot([np.log10(benign_energy + 1e-12), np.log10(malicious_energy + 1e-12)], positions=[1, 2], showmedians=True)
        ax3.set_xticks([1, 2]); ax3.set_xticklabels(["benign", "malicious"])
        ax3.axhline(np.log10(threshold), color="black", linestyle="--", label=f"threshold = {threshold:.2f}")
        ax3.set_ylabel("log₁₀(energy)"); ax3.set_title("violin"); ax3.legend()

        ax4 = fig.add_subplot(gs[1, 1])
        ax4.plot(fpr_curve, tpr_curve, color="darkorange", label=f"ROC AUC={roc_auc:.3f}")
        ax4.plot([0,1],[0,1], "navy", linestyle="--")
        ax4.scatter([fpr_op], [tpr_op], color="red", zorder=5, label=f"monday threshold\n(TPR={tpr_op:.2f}, FPR={fpr_op:.2f})")
        ax4.set_xlabel("FPR")
        ax4.set_ylabel("TPR")
        ax4.set_title("ROC Curve")
        ax4.legend()

        fig.suptitle("residual energy visualisation", fontsize=14, fontweight="bold")
        plt.savefig("energy_analysis.png", dpi=300)
        plt.close()
        print("saved diagram")




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





benign_data, benign_identification, mins, denom = process_benign_data(df_monday)
common_cols = benign_data.columns
test_data, test_identification = process_test_data(data, mins, denom)
test_data = test_data.reindex(columns=common_cols, fill_value=0.0)

print("scaled test feature ranges:")
print(test_data.abs().max().sort_values(ascending=False).head(15))

feature_idx_map = {}
for (idx, feature_name) in enumerate(test_data.columns.values):
    feature_idx_map[feature_name] = idx
timestart = to_microseconds(test_identification["Timestamp"].iloc[0])

graph: FlowGraph;
baseline: BaselineGraph;

load_from_file = False
if load_from_file:
    file = open("graph.bin", "rb")
    graph = pickle.load(file)
    file.close()
    graph.structure()
else:
    baseline = BaselineGraph(feature_len=len(benign_data.columns))
    baseline.initialize_from_df(benign_data, benign_identification)

    graph = FlowGraph(feature_len=len(benign_data.columns), timestart=timestart, baseline=baseline)
    graph.initialize(test_data, test_identification)
    pickle.dump(graph, open("graph.bin", "wb"), protocol=pickle.HIGHEST_PROTOCOL)


check = 362075
#print(f"edges: {len(graph._edges[check])}")
#graph.compare(check, graph._edges[check].pop())
#graph._nodes[check].structure()
graph.find_anomalies(test_identification)
