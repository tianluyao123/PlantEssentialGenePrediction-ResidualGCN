import os

import numpy as np
from Bio import SeqIO
from sklearn.model_selection import StratifiedKFold

from .features import SequenceProcessor, BioFeatureExtractor, matrix_encoding_no_bio
from .graph import build_sparse_bipartite_graph_with_bio, augment_seq


def load_kfold_dataset(config, K: int, d: int, random_state=None, test_indices=None, use_predefined_split: bool = True):
    print(f"Loading dataset (K={K}, d={d})...")
    rs = random_state if random_state is not None else config.RANDOM_STATE
    seq_processor = SequenceProcessor(config.SEQ_MAX_LEN, config.PADDING_STRATEGY)
    bio_extractor = BioFeatureExtractor()

    cv_graphs, cv_labels = [], np.array([])
    cv_feats, cv_bio_feats = np.array([]), np.array([])
    test_graphs, test_labels = [], np.array([])
    test_feats, test_bio_feats = np.array([]), np.array([])
    fold_splits = []
    cv_seqs_with_aa = []

    if use_predefined_split and hasattr(config, 'ESSENTIAL_TRAIN_FASTA') and os.path.exists(
            config.ESSENTIAL_TRAIN_FASTA):
        print("  - Using predefined training and test files...")

        essential_train = [(str(r.seq).upper(), 1, r.id) for r in SeqIO.parse(config.ESSENTIAL_TRAIN_FASTA, "fasta")]
        non_essential_train = [(str(r.seq).upper(), 0, r.id) for r in
                               SeqIO.parse(config.NON_ESSENTIAL_TRAIN_FASTA, "fasta")]

        essential_test = [(str(r.seq).upper(), 1, r.id) for r in SeqIO.parse(config.ESSENTIAL_TEST_FASTA, "fasta")]
        non_essential_test = [(str(r.seq).upper(), 0, r.id) for r in
                              SeqIO.parse(config.NON_ESSENTIAL_TEST_FASTA, "fasta")]

        aa_train_dict = {}
        for record in SeqIO.parse(config.ESSENTIAL_AA_TRAIN_FASTA, "fasta"):
            aa_train_dict[record.id] = str(record.seq).upper()
        for record in SeqIO.parse(config.NON_ESSENTIAL_AA_TRAIN_FASTA, "fasta"):
            aa_train_dict[record.id] = str(record.seq).upper()

        aa_test_dict = {}
        for record in SeqIO.parse(config.ESSENTIAL_AA_TEST_FASTA, "fasta"):
            aa_test_dict[record.id] = str(record.seq).upper()
        for record in SeqIO.parse(config.NON_ESSENTIAL_AA_TEST_FASTA, "fasta"):
            aa_test_dict[record.id] = str(record.seq).upper()

        train_seqs = essential_train + non_essential_train
        test_seqs = essential_test + non_essential_test

        print(f"  - Training set distribution: essential genes={len(essential_train)}, non-essential genes={len(non_essential_train)}")
        print(f"  - Test set distribution: essential genes={len(essential_test)}, non-essential genes={len(non_essential_test)}")

        cv_seqs_with_aa = [(seq, aa_train_dict.get(seq_id, ""), label, seq_id) for seq, label, seq_id in train_seqs]
        test_seqs_with_aa = [(seq, aa_test_dict.get(seq_id, ""), label, seq_id) for seq, label, seq_id in test_seqs]

        def build_graphs(seq_list, subset_name):
            graphs, main_features, bio_features, labels = [], [], [], []
            for idx, (seq_str, aa_seq, label, seq_id) in enumerate(seq_list):
                try:
                    processed_seq = seq_processor.process_sequence(seq_str)
                    main_feat, kmer_count = matrix_encoding_no_bio(processed_seq, K, d)

                    bio_feat = np.array([], dtype=np.float32)
                    if config.USE_BIO_FEATURES:
                        cds_features = bio_extractor.extract_cds_features(seq_str)
                        aa_features = bio_extractor.extract_aa_features(aa_seq)
                        bio_feat = np.concatenate([cds_features, aa_features])

                    graph = build_sparse_bipartite_graph_with_bio(main_feat, kmer_count, bio_feat, K, config)
                    graph.raw_main_feat = main_feat
                    graphs.append(graph)
                    main_features.append(main_feat)
                    bio_features.append(bio_feat)
                    labels.append(label)

                    if (idx + 1) % 1000 == 0:
                        print(f"  - {subset_name}: built {idx + 1}/{len(seq_list)} graph samples")
                except Exception as e:
                    print(f"Warning: Graph construction failed: {e}")
                    continue
            return graphs, np.array(main_features), np.array(bio_features), np.array(labels)

        cv_graphs, cv_feats, cv_bio_feats, cv_labels = build_graphs(cv_seqs_with_aa, "training set")
        test_graphs, test_feats, test_bio_feats, test_labels = build_graphs(test_seqs_with_aa, "test set")

    if config.AUGMENT_TRAIN and len(cv_graphs) > 0:
        print("  - Augmenting positive training samples...")
        augmented = []
        for i, (graph, feat, bio_feat, label) in enumerate(zip(cv_graphs, cv_feats, cv_bio_feats, cv_labels)):
            augmented.append((graph, feat, bio_feat, label))
            if label == 1 and np.random.random() < 0.5 and i < len(cv_seqs_with_aa):
                seq_str = cv_seqs_with_aa[i][0]
                aug_seq = augment_seq(seq_str, config.AUGMENT_PROB)
                processed_seq = seq_processor.process_sequence(aug_seq)
                main_feat, kmer_count = matrix_encoding_no_bio(processed_seq, K, d)
                bio_feat = cv_bio_feats[i] if i < len(cv_bio_feats) else np.array([])

                aug_graph = build_sparse_bipartite_graph_with_bio(main_feat, kmer_count, bio_feat, K, config)
                aug_graph.raw_main_feat = main_feat
                augmented.append((aug_graph, main_feat, bio_feat, label))

        cv_graphs = [x[0] for x in augmented]
        cv_feats = np.array([x[1] for x in augmented])
        cv_bio_feats = np.array([x[2] for x in augmented])
        cv_labels = np.array([x[3] for x in augmented])
        print(f"  - Augmented training size: {len(cv_graphs)}")

    kf = StratifiedKFold(n_splits=config.N_FOLDS, shuffle=True, random_state=rs)
    fold_splits = list(kf.split(X=cv_graphs, y=cv_labels))

    print(f"  - cross-validation set: {len(cv_graphs)} samples, test set: {len(test_graphs)} samples")
    return (cv_graphs, cv_labels, cv_feats, cv_bio_feats,
            test_graphs, test_labels, test_feats, test_bio_feats,
            fold_splits, list(range(len(test_labels))))
