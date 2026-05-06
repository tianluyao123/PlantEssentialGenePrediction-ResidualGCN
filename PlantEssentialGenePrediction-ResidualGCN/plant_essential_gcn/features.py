import pickle

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .config import OptimizedConfig


def _num_transfer(seq: str) -> str:
    seq = seq.upper()
    seq = seq.replace("A", "0").replace("C", "1").replace("G", "2").replace("T", "3")
    return "".join(filter(str.isdigit, seq))


def _num_transfer_loc(num_seq: str, K: int) -> list[int]:
    loc = []
    seq_len = len(num_seq)
    if seq_len < K:
        num_seq = num_seq.ljust(K, "0")
        seq_len = K
    for i in range(seq_len - K + 1):
        kmer_val = int(num_seq[i:i + K], base=4)
        loc.append(kmer_val)
    return loc


def _count_kmer_occurrence(loc_list: list[int], num_kmer: int) -> np.ndarray:
    count = np.zeros(num_kmer, dtype=int)
    for loc in loc_list:
        if loc < num_kmer:
            count[loc] += 1
    return count


def _loc_transfer_matrix(loc_list: list[int], dis: list[int], K: int, seq_length: int) -> np.ndarray:
    num_kmer = 4 ** K
    matrix = np.zeros((num_kmer, num_kmer), dtype=np.float32)
    num = 0
    dis_val = dis[0]
    valid_len = len(loc_list) - K - dis_val

    if valid_len > 0:
        for i in range(valid_len):
            idx1 = loc_list[i]
            idx2 = loc_list[i + K + dis_val]
            if idx1 < num_kmer and idx2 < num_kmer:
                matrix[idx1][idx2] += 1
        num = max(1, seq_length - 2 * K - dis_val + 1)
    return matrix / num if num != 0 else matrix


def matrix_encoding_no_bio(seq: str, K: int, d: int) -> tuple[np.ndarray, np.ndarray]:
    seq = seq.upper()
    seq_length = len(seq)
    num_seq = _num_transfer(seq) or "0" * K
    loc_list = _num_transfer_loc(num_seq, K)

    d = min(d, 4)
    dis_list = [[0], [1], [3], [5]][:d]

    num_kmer = 4 ** K
    kmer_features = []

    for dis in dis_list:
        matrix = _loc_transfer_matrix(loc_list, dis, K, seq_length)
        flattened_matrix = matrix.flatten()
        kmer_features.append(flattened_matrix)

    if kmer_features:
        main_feat = np.hstack(kmer_features) * 100
    else:
        main_feat = np.zeros(num_kmer * num_kmer * d, dtype=np.float32)

    kmer_count = _count_kmer_occurrence(loc_list, num_kmer)

    return main_feat.astype(np.float32), kmer_count


class SequenceProcessor:
    def __init__(self, max_len: int, strategy: str = "smart"):
        self.max_len = max_len
        self.strategy = strategy

    def process_sequence(self, seq_str: str) -> str:
        seq_str = seq_str.upper()
        if len(seq_str) > self.max_len:
            return self._truncate_sequence(seq_str)
        elif len(seq_str) < self.max_len:
            return self._pad_sequence(seq_str)
        else:
            return seq_str

    def _truncate_sequence(self, seq_str: str) -> str:
        if self.strategy == "smart":
            start = (len(seq_str) - self.max_len) // 2
            return seq_str[start:start + self.max_len]
        else:
            return seq_str[-self.max_len:]

    def _pad_sequence(self, seq_str: str) -> str:
        if self.strategy == "smart":
            total_pad = self.max_len - len(seq_str)
            left_pad = total_pad // 2
            return "N" * left_pad + seq_str + "N" * (total_pad - left_pad)
        else:
            return seq_str + "N" * (self.max_len - len(seq_str))


class FeatureReducer:


    def __init__(self, target_dim: int):
        self.target_dim = target_dim
        self.pca = None
        self.scaler = None
        self.is_fitted = False

    def fit_transform(self, features: np.ndarray) -> np.ndarray:
        if features.shape[1] <= self.target_dim:
            self.is_fitted = True
            return features.astype(np.float32)

        self.scaler = StandardScaler()
        features_scaled = self.scaler.fit_transform(features)

        self.pca = PCA(n_components=self.target_dim, random_state=OptimizedConfig.RANDOM_STATE)
        reduced_features = self.pca.fit_transform(features_scaled)
        self.is_fitted = True
        explained_variance = np.sum(self.pca.explained_variance_ratio_)
        print(f"  - PCA reduction: {features.shape[1]} -> {self.target_dim} dimensions, retained variance: {explained_variance:.4f}")
        return reduced_features.astype(np.float32)

    def transform(self, features: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("FeatureReducer has not been fitted on the training set.")
        if self.pca is None or features.shape[1] <= self.target_dim:
            return features.astype(np.float32)

        if self.scaler is not None:
            features = self.scaler.transform(features)
        return self.pca.transform(features).astype(np.float32)

    def save(self, filepath: str):

        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"  - Reducer saved: {filepath}")

    @classmethod
    def load(cls, filepath: str):

        with open(filepath, 'rb') as f:
            reducer = pickle.load(f)
        print(f"  - Reducer loaded: {filepath}")
        return reducer


class BioFeatureExtractor:
    def __init__(self):
        self.amino_acid_groups = {
            'hydrophobic': ['A', 'V', 'L', 'I', 'P', 'F', 'W', 'M'],
            'hydrophilic': ['R', 'N', 'D', 'Q', 'E', 'K', 'H'],
            'neutral': ['S', 'T', 'Y', 'C', 'G'],
            'aromatic': ['F', 'W', 'Y'],
            'charged_positive': ['R', 'K', 'H'],
            'charged_negative': ['D', 'E'],
            'tiny': ['A', 'G', 'S'],
            'small': ['A', 'G', 'S', 'C', 'T', 'D', 'N', 'V'],
            'large': ['R', 'K', 'E', 'Q', 'H', 'W', 'Y', 'F', 'M', 'I', 'L']
        }

        self.aa_properties = {
            'A': {'mw': 89.09, 'hydrophobicity': 0.62, 'pi': 6.0, 'volume': 88.6},
            'R': {'mw': 174.20, 'hydrophobicity': -2.53, 'pi': 10.76, 'volume': 173.4},
            'N': {'mw': 132.12, 'hydrophobicity': -0.78, 'pi': 5.41, 'volume': 114.1},
            'D': {'mw': 133.10, 'hydrophobicity': -0.90, 'pi': 2.77, 'volume': 111.1},
            'C': {'mw': 121.15, 'hydrophobicity': 0.29, 'pi': 5.07, 'volume': 108.5},
            'Q': {'mw': 146.15, 'hydrophobicity': -0.85, 'pi': 5.65, 'volume': 143.8},
            'E': {'mw': 147.13, 'hydrophobicity': -0.74, 'pi': 3.22, 'volume': 138.4},
            'G': {'mw': 75.07, 'hydrophobicity': 0.48, 'pi': 5.97, 'volume': 60.1},
            'H': {'mw': 155.16, 'hydrophobicity': -0.40, 'pi': 7.59, 'volume': 153.2},
            'I': {'mw': 131.17, 'hydrophobicity': 1.38, 'pi': 6.02, 'volume': 166.7},
            'L': {'mw': 131.17, 'hydrophobicity': 1.06, 'pi': 5.98, 'volume': 166.7},
            'K': {'mw': 146.19, 'hydrophobicity': -1.50, 'pi': 9.74, 'volume': 168.6},
            'M': {'mw': 149.21, 'hydrophobicity': 0.64, 'pi': 5.74, 'volume': 162.9},
            'F': {'mw': 165.19, 'hydrophobicity': 1.19, 'pi': 5.48, 'volume': 189.9},
            'P': {'mw': 115.13, 'hydrophobicity': 0.12, 'pi': 6.30, 'volume': 112.7},
            'S': {'mw': 105.09, 'hydrophobicity': -0.18, 'pi': 5.68, 'volume': 89.0},
            'T': {'mw': 119.12, 'hydrophobicity': -0.05, 'pi': 5.60, 'volume': 116.1},
            'W': {'mw': 204.23, 'hydrophobicity': 0.81, 'pi': 5.89, 'volume': 227.8},
            'Y': {'mw': 181.19, 'hydrophobicity': 0.26, 'pi': 5.66, 'volume': 193.6},
            'V': {'mw': 117.15, 'hydrophobicity': 1.08, 'pi': 5.96, 'volume': 140.0}
        }

    def extract_cds_features(self, cds_seq: str) -> np.ndarray:
        features = []
        if not cds_seq:
            return np.zeros(OptimizedConfig.CDS_FEATURE_DIM, dtype=np.float32)
        seq_len = len(cds_seq)

        gc_content = (cds_seq.count('G') + cds_seq.count('C')) / seq_len
        at_content = (cds_seq.count('A') + cds_seq.count('T')) / seq_len
        gc_skew = (cds_seq.count('G') - cds_seq.count('C')) / (cds_seq.count('G') + cds_seq.count('C') + 1e-8)
        at_skew = (cds_seq.count('A') - cds_seq.count('T')) / (cds_seq.count('A') + cds_seq.count('T') + 1e-8)
        features.extend([gc_content, at_content, gc_skew, at_skew])

        nucleotides = ['A', 'C', 'G', 'T']
        for nt in nucleotides:
            features.append(cds_seq.count(nt) / seq_len)

        di_nucleotides = [a + b for a in nucleotides for b in nucleotides]
        total_pairs = max(1, seq_len - 1)
        for di in di_nucleotides:
            features.append(cds_seq.count(di) / total_pairs)

        tri_nucleotides = [a + b + c for a in nucleotides for b in nucleotides for c in nucleotides][:16]
        total_triplets = max(1, seq_len - 2)
        for tri in tri_nucleotides:
            features.append(cds_seq.count(tri) / total_triplets)

        features.extend([seq_len / 1000.0, np.log(seq_len + 1), seq_len / 3000.0])

        if seq_len >= 3:
            codons = [cds_seq[i:i + 3] for i in range(0, seq_len - 2, 3)]
            codon_count = len(codons)
            if codon_count > 0:
                unique_codons = len(set(codons))
                codon_diversity = unique_codons / codon_count
                effective_codons = 1 / sum((codons.count(codon) / codon_count) ** 2 for codon in set(codons))
                start_codons = ['ATG']
                stop_codons = ['TAA', 'TAG', 'TGA']
                start_count = sum(1 for codon in codons if codon in start_codons)
                stop_count = sum(1 for codon in codons if codon in stop_codons)
                features.extend([codon_diversity, effective_codons / 61.0, start_count / codon_count,
                                 stop_count / codon_count, codon_count / 100.0])
            else:
                features.extend([0.0] * 5)
        else:
            features.extend([0.0] * 5)

        orf_features = self._extract_orf_features(cds_seq)
        features.extend(orf_features)

        repeat_features = self._extract_repeat_features(cds_seq)
        features.extend(repeat_features)

        complexity_features = self._extract_complexity_features(cds_seq)
        features.extend(complexity_features)

        target_dim = OptimizedConfig.CDS_FEATURE_DIM
        if len(features) < target_dim:
            features.extend([0.0] * (target_dim - len(features)))
        elif len(features) > target_dim:
            features = features[:target_dim]

        return np.array(features, dtype=np.float32)

    def _extract_orf_features(self, seq: str) -> list:
        if len(seq) < 6:
            return [0.0] * 5
        start_codons = ['ATG']
        stop_codons = ['TAA', 'TAG', 'TGA']
        orf_lengths = []
        orf_counts = [0, 0, 0]
        for frame in range(3):
            current_orf = 0
            in_orf = False
            frame_orf_count = 0
            for i in range(frame, len(seq) - 2, 3):
                codon = seq[i:i + 3]
                if codon in start_codons and not in_orf:
                    in_orf = True
                    current_orf = 1
                elif codon in stop_codons and in_orf:
                    in_orf = False
                    orf_lengths.append(current_orf)
                    frame_orf_count += 1
                    current_orf = 0
                elif in_orf:
                    current_orf += 1
            orf_counts[frame] = frame_orf_count

        max_orf = max(orf_lengths) if orf_lengths else 0
        avg_orf = np.mean(orf_lengths) if orf_lengths else 0
        total_orfs = len(orf_lengths)

        return [max_orf / 100.0, avg_orf / 50.0, total_orfs / 10.0,
                sum(orf_counts) / 10.0, max(orf_counts) / 5.0]

    def _extract_repeat_features(self, seq: str) -> list:
        repeat_units = ['AT', 'GC', 'AC', 'GT', 'AG', 'CT']
        total_repeats = 0
        for unit in repeat_units:
            count = 0
            i = 0
            while i < len(seq) - 1:
                if seq[i:i + 2] == unit:
                    count += 1
                    i += 2
                else:
                    i += 1
            total_repeats += count

        mono_repeats = (seq.count('A' * 3) + seq.count('T' * 3) +
                        seq.count('C' * 3) + seq.count('G' * 3))
        return [total_repeats / len(seq) if len(seq) > 0 else 0.0,
                mono_repeats / len(seq) if len(seq) > 0 else 0.0]

    def _extract_complexity_features(self, seq: str) -> list:
        if len(seq) == 0:
            return [0.0, 0.0, 0.0]

        nucleotide_counts = [seq.count(nt) for nt in ['A', 'C', 'G', 'T']]
        total = sum(nucleotide_counts)
        entropy = 0.0
        for count in nucleotide_counts:
            if count > 0:
                p = count / total
                entropy -= p * np.log2(p)
        entropy /= 2.0

        window_size = min(100, len(seq) // 3)
        gc_variation = 0.0
        if window_size > 0:
            gc_contents = []
            for i in range(0, len(seq) - window_size + 1, window_size):
                window = seq[i:i + window_size]
                gc_content = (window.count('G') + window.count('C')) / len(window)
                gc_contents.append(gc_content)
            if len(gc_contents) > 1:
                gc_variation = np.std(gc_contents)

        uniformity = len(set(seq)) / len(seq) if len(seq) > 0 else 0.0
        return [entropy, gc_variation, uniformity]

    def extract_aa_features(self, aa_seq: str) -> np.ndarray:
        features = []
        if not aa_seq or len(aa_seq) == 0:
            return np.zeros(OptimizedConfig.AA_FEATURE_DIM, dtype=np.float32)

        total = len(aa_seq)
        aa_list = 'ACDEFGHIKLMNPQRSTVWY'

        for aa in aa_list:
            features.append(aa_seq.count(aa) / total)

        for group_name, group_aa in self.amino_acid_groups.items():
            count = sum(1 for aa in aa_seq if aa in group_aa)
            features.append(count / total)

        representative_aa = ['A', 'L', 'V', 'G', 'E', 'K', 'R', 'D', 'S', 'T',
                             'I', 'F', 'Y', 'N', 'Q', 'H', 'M', 'P', 'W', 'C']
        dipeptides = [a + b for a in representative_aa for b in representative_aa][:20]
        total_pairs = max(1, len(aa_seq) - 1)
        for di in dipeptides:
            count = sum(1 for i in range(len(aa_seq) - 1) if aa_seq[i:i + 2] == di)
            features.append(count / total_pairs)

        mw_sum = hydrophobicity_sum = pi_sum = volume_sum = 0
        valid_aa_count = 0
        for aa in aa_seq:
            if aa in self.aa_properties:
                props = self.aa_properties[aa]
                mw_sum += props['mw']
                hydrophobicity_sum += props['hydrophobicity']
                pi_sum += props['pi']
                volume_sum += props['volume']
                valid_aa_count += 1

        if valid_aa_count > 0:
            features.extend([mw_sum / valid_aa_count / 1000.0,
                             hydrophobicity_sum / valid_aa_count,
                             pi_sum / valid_aa_count / 10.0,
                             volume_sum / valid_aa_count / 100.0,
                             np.std([self.aa_properties[aa]['mw'] for aa in aa_seq
                                     if aa in self.aa_properties]) / 100.0
                             if valid_aa_count > 1 else 0.0])
        else:
            features.extend([0.0] * 5)

        helix_formers = ['E', 'A', 'L', 'M', 'Q', 'K', 'R']
        sheet_formers = ['V', 'I', 'Y', 'F', 'W', 'T']
        coil_formers = ['G', 'P', 'S', 'D', 'N']
        helix_score = sum(1 for aa in aa_seq if aa in helix_formers) / len(aa_seq)
        sheet_score = sum(1 for aa in aa_seq if aa in sheet_formers) / len(aa_seq)
        coil_score = sum(1 for aa in aa_seq if aa in coil_formers) / len(aa_seq)
        features.extend([helix_score, sheet_score, coil_score])

        hydrophobic = sum(1 for aa in aa_seq if aa in self.amino_acid_groups['hydrophobic'])
        hydrophilic = sum(1 for aa in aa_seq if aa in self.amino_acid_groups['hydrophilic'])
        charged_pos = sum(1 for aa in aa_seq if aa in self.amino_acid_groups['charged_positive'])
        charged_neg = sum(1 for aa in aa_seq if aa in self.amino_acid_groups['charged_negative'])
        features.extend([hydrophobic / total, hydrophilic / total, charged_pos / total,
                         charged_neg / total, (charged_pos - charged_neg) / total])

        features.extend([len(aa_seq) / 1000.0, len(set(aa_seq)) / len(aa_seq), len(aa_seq) / 500.0])
        features.extend([aa_seq.count('C') / total, aa_seq.count('P') / total, aa_seq.count('G') / total])

        target_dim = OptimizedConfig.AA_FEATURE_DIM
        if len(features) < target_dim:
            features.extend([0.0] * (target_dim - len(features)))
        elif len(features) > target_dim:
            features = features[:target_dim]

        return np.array(features, dtype=np.float32)
