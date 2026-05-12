import json
from pathlib import Path
import time
import pandas as pd
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import roc_curve, auc, f1_score, fbeta_score
pd.set_option("display.max_rows", None)
pd.set_option("display.float_format", "{:.6f}".format)

IO_FEATURES = ["Flow Packets/s", "Flow Bytes/s", "Average Packet Size"]
OUT_FEATURES = ["Total Fwd Packet", "Fwd Packets/s"]  # "Dst Port",
IN_FEATURES = ["Total Bwd packets", "Bwd Packets/s"]  # "Src Port",
ALL_IP_FEATURES = IO_FEATURES + OUT_FEATURES + IN_FEATURES

TIME_BIN_SIZE_S = 180
TIME_BIN_SIZE = TIME_BIN_SIZE_S * 1000000
ALPHA = 1 # weight of local signal 0 = global only 1 = local only
MIN_EDGES = 3
ENERGY_THRESHOLD_SCALAR = 1.4
VERBOSE = True
FEATURES_TO_DISPLAY = 5
VAR_MINIMUM = 0.01
TOPK = 5
protocol_map = {6: "TCP", 17: "UDP", 1: "ICMP"}
type FlowId = np.int32

# 'Packet Length Mean' and 'Average Packet Size' seem to just be the same value with very minor changes, like 0.01 difference
# maybe remove source/dst port since it doesnt give that much info, and more often than not gives energy which isnt relevant, since nodes often use many different ports, same with protocol
def process_benign_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    df = df.dropna()
    #df = df.drop(columns=[c for c in df.columns if df[c].nunique() == 1])
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    #df.loc[df["Attempted Category"] != -1, "Label"] = "BENIGN"

    split_cols = ["id", "Flow ID", "Attempted Category", "Label", "Src IP", "Dst IP", "Timestamp", "Protocol"]

    identification_data = df[split_cols].copy()
    identification_data = identification_data.set_index("id", drop=False)
    split_cols.remove("Protocol")
    df = df.drop(columns=split_cols)

    mins = df.min() 
    denom = (df.max() - mins).replace(0, 1)
    df = (df - mins) / denom # original = scaled * denom + mins 
    return (df, identification_data, mins, denom)


# for ATTACK DATA
def process_test_data(df: pd.DataFrame, mins: pd.Series, denom: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.dropna()
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df.loc[df["Attempted Category"] != -1, "Label"] = "BENIGN"

    split_cols = ["id", "Flow ID", "Attempted Category", "Label", "Src IP", "Dst IP", "Timestamp", "Protocol"]

    identification_data = df[split_cols].copy()
    identification_data = identification_data.set_index("id", drop=False)

    split_cols.remove("Protocol")
    df = df.drop(columns=split_cols)
    df = df.reindex(columns=mins.index, fill_value=0.0)


    df = (df - mins) / denom
    #df = df.clip(lower=-FEATURE_CLIP_MAX, upper=FEATURE_CLIP_MAX)  
    return (df, identification_data)



def display_ip_io_fv(fv: np.ndarray):
    width = 20
    print("".join(f"{h:<{width}}" for h in ALL_IP_FEATURES))
    print("".join(f"{v:<{width}.3f}" for v in np.round(fv, 3)))



def to_microseconds(timestamp: str) -> int:
    return (
        int(timestamp[8:10]) * 1000000 * 60 * 60 * 24 + # day
        int(timestamp[11:13]) * 1000000 * 60 * 60+  # hours
        int(timestamp[14:16]) * 1000000 * 60 +  # minutes
        int(timestamp[17:19]) * 1000000 +  # seconds
        int(timestamp[20:26])    # microseconds
    )



def update_running_variance_batch(old_mean, old_M2, old_n, new_values):
    new_n = len(new_values)
    total_n = old_n + new_n

    if total_n == 0:
        return old_mean, old_M2, total_n, old_mean * 0

    new_mean_batch = new_values.mean(axis=0)
    new_mean = (old_mean * old_n + new_mean_batch * new_n) / total_n

    new_M2_batch = ((new_values - new_mean_batch) ** 2).sum(axis=0)

    correction = old_n * new_n / total_n * ((old_mean - new_mean_batch) ** 2)

    new_M2 = old_M2 + new_M2_batch + correction
    new_variance = new_M2 / (total_n - 1) if total_n > 1 else np.zeros_like(new_mean)

    return new_mean, new_M2, total_n, new_variance

class FlowNode:
    def __init__(self, flow_id: FlowId, fv: np.ndarray, src: str, dst: str, dstp: int):
        self.dstp = dstp
        self.flow_id: FlowId = flow_id
        self.src: str = src
        self.dst: str = dst
        self.fv: np.ndarray = fv
        self.degree: int = 0
        self.real_degree: int = 0
        self.m2: np.ndarray = np.zeros(len(fv))
        self.mean_fv: np.ndarray = np.zeros(len(fv), dtype=np.float64)
        self.peer_var: np.ndarray = np.zeros(len(fv))


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
        #self._src_fvs: defaultdict[str, list[np.ndarray]] = defaultdict(list)
        self.dst_mean: dict[str, np.ndarray] = {}
        self.dst_var: dict[str, np.ndarray] = {}
        #self.src_mean: dict[str, np.ndarray] = {}
        self.global_mean: np.ndarray = np.zeros(feature_len)
        self.global_var: np.ndarray = np.ones(feature_len)
        self.energy_threshold: float = 0.0




    def add_flow(self, src: str, dst: str, protocol: int, features: np.ndarray):
        self._dst_fvs[dst].append(features)
        #self._src_fvs[src].append(features)

    def build(self, threshold_percentile: float = 99.5):
        start = time.perf_counter()
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
            v = np.maximum(v, self.global_var * VAR_MINIMUM)
            self.dst_var[dst] = v

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
        end = time.perf_counter()
        print(f"baseline built: {len(all_fvs)} flows, {len(self.dst_mean)} unique dsts")
        print(f"baseline built in: {start-end:.6f}")
        print(f"monday log-energy: mu={mu:.2f}, std={std:.2f}")
        print(f"energy threshold (mu+{ENERGY_THRESHOLD_SCALAR}σ): {self.energy_threshold:.2f}")

    def global_energy(self, fv: np.ndarray, dst: str, protocol: int) -> float:
        mean = self.dst_mean.get(dst, self.global_mean)
        var = self.dst_var.get(dst, self.global_var)

        # var must be 1% of global
        var_floored = np.maximum(var, self.global_var * VAR_MINIMUM)

        return float(np.sum((fv - mean) ** 2 / var_floored))

    def perf_energy(self, fv: np.ndarray, dst: str, protocol: int) -> np.ndarray:
        mean = self.dst_mean.get(dst, self.global_mean)
        var = self.dst_var.get(dst, self.global_var)

        # var must be 1% of global
        var_floored = np.maximum(var, self.global_var * VAR_MINIMUM)

        return ((fv - mean) ** 2 / var_floored)


    def get_baselines(self, dst: str, protocol: int) -> tuple[np.ndarray, np.ndarray]:
        return (self.dst_mean.get(dst, self.global_mean), np.maximum(self.dst_var.get(dst, self.global_var), self.global_var * VAR_MINIMUM))

    def initialize_from_df(self, data: pd.DataFrame, ids: pd.DataFrame):
        features = data.to_numpy(dtype=np.float64)
        flow_data = ids[["Src IP", "Dst IP", "Protocol"]].to_numpy()
        for i in range(len(features)):
            self.add_flow(flow_data[i, 0], flow_data[i, 1], flow_data[i, 2], features[i])
        self.build()


class FlowGraph:
    def __init__(self, feature_len: int, timestart: int, baseline: BaselineGraph):
        self._feature_len: int = feature_len
        self._nodes: dict[FlowId, FlowNode] = {}
        self._edges: defaultdict[FlowId, set[FlowId]] = defaultdict(set)
        self._src_grp: defaultdict[str, set[FlowId]] = defaultdict(set)
        self._dst_grp: defaultdict[str, set[FlowId]] = defaultdict(set)
        self._fullmatch_grp: defaultdict[tuple[str, str, int, float], set[int]] = defaultdict(set)
        self._timestart = timestart
        self._baseline = baseline
        self.build_time = 0
        self.detect_time = 0


    def insert_flow(self, flow_id: FlowId, src: str, dst: str, timestamp: str, dstp: int, features: np.ndarray, ):
        if flow_id not in self._nodes:
            self._nodes[flow_id] = FlowNode(flow_id, features, src, dst, dstp)
            #self._src_grp[src].add(flow_id)
            #self._dst_grp[dst].add(flow_id)
            #time_bin = (to_microseconds(timestamp) - self._timestart) // TIME_BIN_SIZE
            #self._fullmatch_grp[(src, dst, int(features[feature_idx_map["Protocol"]]), time_bin)].add(flow_id)



    def find_edges(self):
        flow_id_list = list(self._nodes.keys())
        fid_to_idx = {fid:i for i, fid in enumerate(flow_id_list)}
        fvs          = np.array([self._nodes[i].fv for i in flow_id_list])

        energies = np.array([
            self._baseline.global_energy(self._nodes[i].fv, self._nodes[i].dst, 0)
            for i in flow_id_list
            ])
        is_benign = energies < self._baseline.energy_threshold
        print(f"Benign candidates: {is_benign.sum():,} / {len(flow_id_list):,}")

        def apply_group(cluster_name: str, key_fn):
            clusters: defaultdict[int, list[FlowId]] = defaultdict(list)
            for idx in flow_id_list:
                clusters[key_fn(idx)].append(idx)

            total_clusters = len(clusters)
            #print(f"\n{cluster_name}: {total_clusters:,} groups")

            for cluster_num, (cluster_key, flowids_in_cluster) in enumerate(clusters.items(), 1):
                #if cluster_num % 2500 == 0:
                    #print(f"\033[1G\033[2K{cluster_num}", end="", flush=True)
                #print(f"\033[1G\033[2K{cluster_num} ({len(flowids_in_cluster)} flows): {cluster_num:,} / {total_clusters:,} groups processed", end="", flush=True)
                #print( "", end="", flush=True )

                benign_idxs = [fid_to_idx[fid] for fid in flowids_in_cluster if is_benign[fid_to_idx[fid]]]
                if not benign_idxs:
                    continue

                for fid in flowids_in_cluster:
                    i = fid_to_idx[fid]
                    node = self._nodes[fid]
                    new_peers = [j for j in benign_idxs if j != i and flow_id_list[j] not in self._edges[fid]]
                    if not new_peers:
                        continue
                    #new_peers_mean = fvs[new_peers].mean(axis=0)
                    #new_peers_var = fvs[new_peers].var(axis=0)
                    #n_new = len(new_peers)
                    #total = node.degree + n_new
                    node.mean_fv, node.m2, node.degree, node.peer_var = update_running_variance_batch(
                        node.mean_fv,
                        node.m2,
                        node.degree,
                        fvs[new_peers]
                    )
                    #self._edges[fid].update([flow_id_list[j] for j in new_peers])
                    #node.mean_fv   = (node.mean_fv * node.degree + new_peers_mean * n_new) / total
                    #node.peer_var = (node.peer_var * node.degree + new_peers_var * n_new) / total
                    #node.degree    = total
                    node.real_degree += len(flowids_in_cluster) - 1

            print(f"\n{cluster_name}: {total_clusters:,} / {total_clusters:,} groups done")

        #apply_group("Layer1 (src+dst)", lambda i: (srcs[i], dsts[i]))
        apply_group("Layer2 (dst)",     lambda i: (self._nodes[i].dst, (to_microseconds(str(test_identification.at[i, "Timestamp"])) - self._timestart) // TIME_BIN_SIZE))
        apply_group("Layer3 (dstport)", lambda i: (self._nodes[i].fv[feature_idx_map["Dst Port"]], (to_microseconds(str(test_identification.at[i, "Timestamp"])) - self._timestart) // TIME_BIN_SIZE))


    def initialize(self, data: pd.DataFrame, ids: pd.DataFrame):
        start = time.perf_counter()
        features = data.to_numpy(dtype=np.float64)
        flow_data = ids[["id", "Src IP", "Dst IP", "Timestamp", "Protocol"]].to_numpy()
        for i in range(len(features)):
            self.insert_flow(np.int32(flow_data[i, 0]), flow_data[i, 1], flow_data[i, 2], flow_data[i, 3], flow_data[i, 4], features[i])



        print(f"Finding Edges")
        self.find_edges()
        end = time.perf_counter()
        print(f"Graph and edges built in {end-start:.6f}s")
        self.build_time = end-start

    def combined_energy(self, node: FlowNode) -> float:
        global_e = self._baseline.global_energy(node.fv, node.dst, node.dstp)
        if node.degree < MIN_EDGES:
            return global_e

        var_floored = np.maximum(node.peer_var, self._baseline.global_var * VAR_MINIMUM)
        local_residual = node.fv - node.mean_fv
        local_e = float(np.sum(local_residual ** 2 / var_floored))

        return ((1 - ALPHA) * global_e) + (ALPHA * local_e)

    def energy_parts(self, node: FlowNode) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        global_per_feature = self._baseline.perf_energy(node.fv, node.dst, node.dstp)
        global_e = float(global_per_feature.sum())

        if node.degree < MIN_EDGES:
            local_per_feature = np.zeros(self._feature_len)
            return global_per_feature, local_per_feature, global_per_feature 

        var_floored = np.maximum(node.peer_var, self._baseline.global_var * VAR_MINIMUM)
        local_residual = node.fv - node.mean_fv
        local_per_feature = (local_residual ** 2 / var_floored)
        local_e = float(local_per_feature.sum())

        combined = ((1 - ALPHA) * global_per_feature) + (ALPHA * local_per_feature)
        return global_per_feature, local_per_feature, combined

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

        labels = test_identification.loc[[int(i) for i in zero_edge_ids], "Label"]
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

    def get_neighbours_by_id(self, id: FlowId) -> set[int]:
        if id > len(self._nodes)-1:
            raise ValueError("id does not exist")
        return self._edges[id]

    def structure_of(self, id: FlowId):
        if id not in self._nodes:
            raise ValueError("id does not exist")
        print(f"FLOW ID {id}--------------------")
        print(f"Edges ({self._nodes[id].degree}): {self._edges[id]}")
        self._nodes[id].structure()

    def compare(self, lhs_id, rhs_id):
        print(f"COMPARISON----------------------")
        for (header, lhs, rhs) in zip(feature_idx_map, self._nodes[lhs_id].fv, self._nodes[rhs_id].fv):
            print(f"{header}: {lhs} - {rhs}")


    def explain_prediction(self, flow_id):
        print("\n\n" + "-"*120)
        denom_n = denom.to_numpy()
        mins_n = mins.to_numpy()
        feature_names = list(test_data.columns)
        node = self._nodes[flow_id]
        initial_features = (node.fv*denom_n) + mins_n
        features = node.fv
        baseline_mean, baseline_variance = self._baseline.get_baselines(node.dst, node.dstp)
        initial_stdev = baseline_variance * denom_n
        initial_mean = (baseline_mean*denom_n) + mins_n 
        initial_mean_fv = (node.mean_fv*denom_n) + mins_n
        difference = features - baseline_mean
        energy = self.combined_energy(node)
        global_pf, local_pf, combined = self.energy_parts(node)
        sorted_perf_energy = np.argsort(combined)[::-1]
        prediction = energy > self._baseline.energy_threshold
        label = test_identification.at[flow_id, "Label"]
        is_malicious = label != "BENIGN"
        shannon = self.shannon_entropy(combined)
        rows = [{
            "FEATURE": feature_names[idx],
            "INIT VALUE": initial_features[idx],
            "VALUE": features[idx],
            "INIT MEAN": initial_mean[idx],
            "BASE MEAN": baseline_mean[idx],
            "MEAN_FV": node.mean_fv[idx],
            "INIT MEAN_FV": initial_mean_fv[idx],
            "DIFFERENCE": difference[idx],
            "INIT STDEV": initial_stdev[idx],
            "BASE VARIANCE": baseline_variance[idx],
            "PEER VARIANCE": node.peer_var[idx],
            "Global Energy":  global_pf[idx],
            "Local Energy":   local_pf[idx],
            "ENERGY": combined[idx],

        } for idx in sorted_perf_energy]
        table = pd.DataFrame(rows)
        srcip = test_identification.at[flow_id, "Src IP"]
        dstip = test_identification.at[flow_id, "Dst IP"]
        srcp = int((features[feature_idx_map["Src Port"]]*denom["Src Port"])+mins["Src Port"])
        dstp = int((features[feature_idx_map["Dst Port"]]*denom["Dst Port"])+mins["Dst Port"])
        protocol = protocol_map.get(int((features[feature_idx_map["Protocol"]]*denom["Protocol"])+mins["Protocol"]), "N/A")
        if VERBOSE:
            print("VERBOSE-------------------------------------")
            print(table.to_string(index=False, float_format="%.8f"))
            print(f"Source Address: {srcip}:{srcp}")
            print(f"Destination Address: {dstip}:{dstp}")
            print(f"Protocol: {protocol}")
        else:
            print("SIMPLE---------------------------------------")
            print(f"Top {FEATURES_TO_DISPLAY} Feature Contributions for flow {flow_id}: ({srcip}:{srcp} -> {dstip}:{dstp}) ({protocol})")
            for (idx, row) in table.head(FEATURES_TO_DISPLAY).iterrows():
                sign_text = "increased" if row["DIFFERENCE"] >= 0 else "decreased"
                var_text: str
                if row["BASE VARIANCE"] >= 0.01:
                    var_text = "high"
                elif row["BASE VARIANCE"] >= 0.005:
                    var_text = "normal"
                else:
                    var_text = "low"
                scaled_difference = abs((row["DIFFERENCE"]*denom[row["FEATURE"]]) + mins[row["FEATURE"]])
                print(f"{row["FEATURE"]:>25} {sign_text} by {scaled_difference:,.3f} compared to the neighbour mean of {row["INIT MEAN"]:,.3f} paired with a {var_text} standard deviation of {row["INIT STDEV"]:,.6f} contributed {row["ENERGY"]:,.3f} total energy")

        print(f"Total Energy: {energy:.2f} ({combined.sum()})")
        print(f"Predicted anomalous: {prediction}")
        print(f"Real Label: {label}")
        print(f"Node Degree: {node.degree}")
        print(f"Shannon: {shannon}")
        print(f"Flow Timestamp: {test_identification.at[flow_id, "Timestamp"]}") 





    def find_anomalies(self, test_identification: pd.DataFrame):
        start = time.perf_counter()
        print("finding anomalies")
        ENERGY_THRESHOLD = self._baseline.energy_threshold
        print(f"Energy Threshold: {ENERGY_THRESHOLD:.4f}")
        all_energies = []
        l_energies = []
        g_energies = []
        topk_features = []
        true_is_malicious = []
        stat_grid = []
        shannon = []
        explanations_given = {str(label): [False, False] for label in test_identification["Label"].unique()}
        flow_ids = [int(i) for i in self._nodes.keys()]


        c = 0
        v = 0
        for id, node in self._nodes.items():

            label = str(test_identification.at[id, "Label"])
            is_malicious = label != "BENIGN"

            global_energy, feature_energy, combined_energy = self.energy_parts(node)
            topk = self.all_features(combined_energy)
            energy = self.combined_energy(node)

            shannon.append(float(self.shannon_entropy(combined_energy)))
            all_energies.append(energy)
            l_energies.append(float(feature_energy.sum()))
            g_energies.append(float(global_energy.sum()))
            topk_features.append(topk)
            true_is_malicious.append(is_malicious)


            if (not explanations_given[label][0]) and (energy > ENERGY_THRESHOLD):
                self.explain_prediction(id)
                explanations_given[label][0] = True
    
            if (not explanations_given[label][1]) and (not (energy > ENERGY_THRESHOLD)):
                self.explain_prediction(id)
                explanations_given[label][1] = True


        all_energies = np.array(all_energies)
        true_is_malicious = np.array(true_is_malicious)


        predictions = all_energies > ENERGY_THRESHOLD

        detect_time = time.perf_counter() - start
        print(f"Find Anomalies time: {detect_time:.6f}")

        attack_labels = test_identification.loc[[int(i) for i in flow_ids], "Label"].values
        unique_labels = sorted(set(attack_labels))

        print(f"\n{'Attack Type':<30} {'Total':>8} {'Detected':>9} {'Missed':>8} {'Recall/Spec':>12}")
        print("-" * 65)

        for label in unique_labels:
            mask = attack_labels == label
            label_true = true_is_malicious[mask]
            label_pred = predictions[mask]
            total = mask.sum()

            if label == "BENIGN":
                tn_l = int((~label_pred & ~label_true).sum())
                fp_l = int(( label_pred & ~label_true).sum())
                specificity = tn_l / total if total > 0 else 0
                stat_grid.append(["BENIGN", int(total), int(tn_l), int(fp_l), float(specificity)])
                print(f"{'BENIGN (specificity)':<30} {total:>8,} {tn_l:>9,} {fp_l:>8,} {specificity:>12.3f}")
            else:
                tp_l = int(( label_pred &  label_true).sum())
                fn_l = int((~label_pred &  label_true).sum())
                recall = tp_l / total if total > 0 else 0
                stat_grid.append([label, int(total), int(tp_l), int(fn_l), float(recall)])
                print(f"{label:<30} {total:>8,} {tp_l:>9,} {fn_l:>8,} {recall:>12.3f}")

        print("-" * 65)

        total_flagged = predictions.sum()
        total_malicious = true_is_malicious.sum()
        global_tp = int(( predictions &  true_is_malicious).sum())
        global_tn = int((~predictions & ~true_is_malicious).sum())
        global_fp = int(( predictions & ~true_is_malicious).sum())
        global_fn = int((~predictions &  true_is_malicious).sum())
        tpr = global_tp / (global_tp + global_fn) if (global_tp + global_fn) > 0 else 0
        fpr = global_fp / (global_fp + global_tn) if (global_fp + global_tn) > 0 else 0
        global_precision = global_tp / total_flagged if total_flagged   > 0 else 0.0
        global_recall = global_tp / total_malicious if total_malicious > 0 else 0.0
        global_f1 = (2 * global_precision * global_recall / (global_precision + global_recall) if (global_precision + global_recall) > 0 else 0.0)
        global_accuracy = (global_tp + global_tn) / len(predictions)
        global_fpr = global_fp / (global_fp + global_tn) if (global_fp + global_tn) > 0 else 0.0

        f2 = fbeta_score(true_is_malicious, predictions, beta=2)
        f0_5 = fbeta_score(true_is_malicious, predictions, beta=0.5)

        fpr_curve, tpr_curve, _ = roc_curve(true_is_malicious, all_energies)
        roc_auc = auc(fpr_curve, tpr_curve)

        print(f"\n{'GLOBAL METRICS':}")
        print(f"TP={global_tp:,}  FP={global_fp:,}  TN={global_tn:,}  FN={global_fn:,}")
        print(f"Accuracy:   {global_accuracy:.4f}")
        print(f"Precision:  {global_precision:.4f}")
        print(f"Recall:     {global_recall:.4f}")
        print(f"F1:         {global_f1:.4f}")
        print(f"F2:         {f2:.4f}  (recall-weighted)")
        print(f"F0.5:       {f0_5:.4f}  (precision-weighted)")
        print(f"FPR:        {global_fpr:.4f}")
        print(f"ROC AUC:    {roc_auc:.4f}")
        print(f"TPR:        {global_recall:.4f}")
        print("-" * 70)

        label = f"all-full-{TIME_BIN_SIZE_S}"
        snapshot = {
                "label": label,
                "tp": global_tp,
                "fp": global_fp,
                "tn": global_tn,
                "fn": global_fn,
                "accuracy": global_accuracy,
                "precision": global_precision,
                "recall": global_recall,
                "f1": global_f1,
                "f2": f2,
                "f0.5": f0_5,
                "fpr": global_fpr,
                "tpr": global_recall,
                "roc": roc_auc,
                "stat_grid": stat_grid,
                "topk_feautres": topk_features,
                "energies": all_energies.tolist(),
                "global_energies": g_energies,
                "local_energies": l_energies,
                "shannon": shannon,
                "fids": flow_ids,
                "true_is_malicious": true_is_malicious.tolist(),
                "flow_labels": [test_identification.at[int(fid), "Label"] for fid in flow_ids],
                "build_time": self.build_time,
                "detection_time": detect_time,
                }

        filepath = f"./saves/timed-all/{label}"
        with open(filepath, "w") as f:
            json.dump(snapshot, f)

    def plot_energy_change(self):
        fid: FlowId = np.int32(19)
        peer_fids = list(self._edges[fid])
        energies = []
        node = self._nodes[fid]
        node.degree = 0
        node.mean_fv = np.zeros(len(node.fv), dtype=np.float64)
        node.peer_var = np.zeros(len(node.fv))
        node.m2 = np.zeros(len(node.fv))

        for peer in peer_fids:
            node.mean_fv, node.m2, node.degree, node.peer_var = update_running_variance_batch(node.mean_fv, node.m2, node.degree, np.array([self._nodes[peer].fv]))
            energies.append(self.combined_energy(node))


        peers = list(range(1, len(energies) + 1))

        plt.figure()
        plt.plot(peers, energies)

        plt.xlabel("Number of Peers")
        plt.ylabel("Energy")
        plt.title("Energy Convergence as More Peers are Added")

        plt.ylim(0, 100)

        plt.grid(True)
        plt.tight_layout()
        plt.savefig("energy_convergance.png")
        plt.close()

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

    def topk_features(self, feature_energy: np.ndarray):
        sorted_desc = np.argsort(feature_energy)[::-1]
        total = np.sum(feature_energy)
        return [[int(i), float(feature_energy[i])] for i in sorted_desc[:TOPK]]

    def all_features(self, feature_energy: np.ndarray):
        sorted_desc = np.argsort(feature_energy)[::-1]
        return [[int(i), float(feature_energy[i])] for i in sorted_desc if feature_energy[i] > 1]


    def shannon_entropy(self, x, eps=1e-12):

        total = np.sum(x)
        if total == 0:
            return 0.0

        p = x / total
        p = p[p > 0]  # avoid log(0)

        return -np.sum(p * np.log(p + eps))


dataset_folder = "./dataset"
dataset_files = Path(dataset_folder).rglob("*.csv") 
print(dataset_files)
start = time.perf_counter()
data = pd.concat((pd.read_csv(file) for file in dataset_files if file.name != "monday.csv"), ignore_index=True)
data["id"] = data.index
df_monday = pd.read_csv(f"{dataset_folder}/monday.csv")
after_read = time.perf_counter()
print(f"Dataset read time: {after_read-start:.6f}s")

benign_data, benign_identification, mins, denom = process_benign_data(df_monday)
common_cols = benign_data.columns
test_data, test_identification = process_test_data(data, mins, denom)
test_data = test_data.reindex(columns=common_cols, fill_value=0.0)

print("\nscaled test feature maximums:")
print(test_data.abs().max().sort_values(ascending=False).head(25))
feature_idx_map = {}
for (idx, feature_name) in enumerate(test_data.columns.values):
    feature_idx_map[feature_name] = idx
timestart = to_microseconds(test_identification["Timestamp"].iloc[0])


baseline = BaselineGraph(feature_len=len(benign_data.columns))
baseline.initialize_from_df(benign_data, benign_identification)

graph = FlowGraph(feature_len=len(benign_data.columns), timestart=timestart, baseline=baseline)
graph.initialize(test_data, test_identification)


graph.find_anomalies(test_identification)
