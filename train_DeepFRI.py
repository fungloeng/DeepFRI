import csv
import json
import pickle
import os

import argparse
import numpy as np

from deepfrier.DeepFRI import DeepFRI
from deepfrier.utils import seq2onehot
from deepfrier.utils import load_GO_annot, load_EC_annot


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-gcd', '--gc_dims', type=int, default=[512, 512, 512], nargs='+', help="Dimensions of GraphConv layers.")
    parser.add_argument('-fcd', '--fc_dims', type=int, default=[1024], nargs='+', help="Dimensions of fully connected layers (after GraphConv layers).")
    parser.add_argument('-drop', '--dropout', type=float, default=0.3, help="Dropout rate.")
    parser.add_argument('-l2', '--l2_reg', type=float, default=1e-4, help="L2 regularization coefficient.")
    parser.add_argument('-lr', type=float, default=0.0002, help="Initial learning rate.")
    parser.add_argument('-gc', '--gc_layer', type=str, choices=['GraphConv', 'MultiGraphConv', 'SAGEConv', 'ChebConv', 'GAT', 'NoGraphConv'],
                        help="Graph Conv layer.")
    parser.add_argument('-e', '--epochs', type=int, default=200, help="Number of epochs to train.")
    parser.add_argument('-bs', '--batch_size', type=int, default=64, help="Batch size.")
    parser.add_argument('-pd', '--pad_len', type=int, help="Padd length (max len of protein sequences in train set).")
    parser.add_argument('-ont', '--ontology', type=str, default='mf', choices=['mf', 'bp', 'cc', 'pf', 'ec'], help="Ontology.")
    parser.add_argument('-lm', '--lm_model_name', type=str, help="Path to the pretraned LSTM-Language Model.")
    parser.add_argument('--cmap_type', type=str, default='ca', choices=['ca', 'cb'], help="Contact maps type.")
    parser.add_argument('--cmap_thresh', type=float, default=10.0, help="Distance cutoff for thresholding contact maps.")
    parser.add_argument('--model_name', type=str, default='GCN-PDB_MF', help="Name of the GCN model.")
    parser.add_argument('--train_tfrecord_fn', type=str, default="/mnt/ceph/users/vgligorijevic/ContactMaps/TFRecords/PDB_GO_train", help="Train tfrecords.")
    parser.add_argument('--valid_tfrecord_fn', type=str, default="/mnt/ceph/users/vgligorijevic/ContactMaps/TFRecords/PDB_GO_valid", help="Valid tfrecords.")
    parser.add_argument('--annot_fn', type=str, default="./preprocessing/data/nrPDB-GO_2019.06.18_annot.tsv", help="File (*tsv) with GO term annotations.")
    parser.add_argument('--test_list', type=str, default="./preprocessing/data/nrPDB-GO_2019.06.18_test.csv", help="File with test PDB chains.")
    parser.add_argument('--test_npz_dir', type=str, default=None, help="Directory containing test set npz files. If not specified, uses hardcoded path.")

    args = parser.parse_args()
    print (args)

    train_tfrecord_fn = args.train_tfrecord_fn + '*'
    valid_tfrecord_fn = args.valid_tfrecord_fn + '*'

    # load annotations
    if args.ontology == 'ec':
        prot2annot, goterms, gonames, counts = load_EC_annot(args.annot_fn)
    else:
        prot2annot, goterms, gonames, counts = load_GO_annot(args.annot_fn)
    goterms = goterms[args.ontology]
    gonames = gonames[args.ontology]
    output_dim = len(goterms)

    # computing weights for imbalanced go classes
    class_sizes = counts[args.ontology]
    mean_class_size = np.mean(class_sizes)
    # Avoid division by zero: replace zeros with 1 to prevent warnings
    class_sizes_safe = np.where(class_sizes == 0, 1, class_sizes)
    pos_weights = mean_class_size / class_sizes_safe
    pos_weights = np.maximum(1.0, np.minimum(10.0, pos_weights))
    pos_weights = np.concatenate([pos_weights.reshape((len(pos_weights), 1)), pos_weights.reshape((len(pos_weights), 1))], axis=-1)
    pos_weights = {i: {0: pos_weights[i, 0], 1: pos_weights[i, 1]} for i in range(output_dim)}

    print ("### Training model: ", args.model_name, " on ", output_dim, " GO terms.")
    model = DeepFRI(output_dim=output_dim, n_channels=26, gc_dims=args.gc_dims, fc_dims=args.fc_dims,
                    lr=args.lr, drop=args.dropout, l2_reg=args.l2_reg, gc_layer=args.gc_layer,
                    lm_model_name=args.lm_model_name, model_name_prefix=args.model_name)

    model.train(train_tfrecord_fn, valid_tfrecord_fn, epochs=args.epochs, batch_size=args.batch_size, pad_len=args.pad_len,
                cmap_type=args.cmap_type, cmap_thresh=args.cmap_thresh, ont=args.ontology, class_weight=None)

    # save models
    model.save_model()
    model.plot_losses()
    # model.load_model()

    # save model params to json
    with open(args.model_name + "_model_params.json", 'w') as fw:
        out_params = vars(args)
        out_params['goterms'] = goterms
        out_params['gonames'] = gonames
        json.dump(out_params, fw, indent=1)

    Y_pred = []
    Y_true = []
    proteins = []
    # Use provided test_npz_dir or fall back to hardcoded path
    if args.test_npz_dir is not None:
        path = args.test_npz_dir
        if not path.endswith('/') and not path.endswith('\\'):
            path = path + '/'
    else:
        path = '/mnt/home/vgligorijevic/Projects/NewMethods/Contact_maps/DeepFRIer2/preprocessing/data/annot_pdb_chains_npz/'
    with open(args.test_list, mode='r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        next(csv_reader, None)  # header
        skipped_count = 0
        for row in csv_reader:
            prot = row[0]
            npz_file = path + prot + '.npz'
            
            # 检查文件是否存在，如果不存在则跳过
            if not os.path.isfile(npz_file):
                print(f"Warning: NPZ file not found, skipping {prot}: {npz_file}")
                skipped_count += 1
                continue
            
            cmap = np.load(npz_file)
            # Properly convert seqres to string (handle numpy arrays correctly)
            seqres = cmap['seqres']
            if isinstance(seqres, np.ndarray) and seqres.ndim == 1 and seqres.dtype.kind in ("U", "S"):
                sequence = "".join(seqres.tolist())
            elif isinstance(seqres, np.ndarray) and seqres.ndim == 0:
                v = seqres.item()
                sequence = v if isinstance(v, str) else "".join(list(v))
            elif isinstance(seqres, (list, tuple)):
                sequence = "".join([str(x) for x in seqres])
            elif isinstance(seqres, str):
                sequence = seqres
            else:
                sequence = str(seqres)
            
            # Filter to allowed amino acid characters
            allowed = set("ACDEFGHIKLMNPQRSTVWYBXZOU-")
            sequence = "".join([c for c in sequence.upper() if c in allowed])
            
            if len(sequence) == 0:
                print(f"Warning: {prot} has empty sequence, skipping")
                skipped_count += 1
                continue
            
            Ca_dist = cmap['C_alpha']
            
            # Check dimension mismatch between sequence and contact map
            seq_len = len(sequence)
            cmap_size = Ca_dist.shape[0]
            
            if seq_len != cmap_size:
                print(f"Warning: {prot} dimension mismatch: sequence length={seq_len}, contact map size={cmap_size}")
                # Use the smaller dimension to ensure consistency
                min_size = min(seq_len, cmap_size)
                sequence = sequence[:min_size]
                Ca_dist = Ca_dist[:min_size, :min_size]
                seq_len = min_size  # Update seq_len after truncation
                print(f"  Truncated to size {min_size}")

            A = np.double(Ca_dist < args.cmap_thresh)
            S = seq2onehot(sequence)

            # Pad to pad_len if needed
            current_seq_len = len(sequence)  # Use actual sequence length after any truncation
            if current_seq_len < args.pad_len:
                # Pad sequence
                pad_seq = np.zeros((args.pad_len, S.shape[1]), dtype=S.dtype)
                pad_seq[:current_seq_len, :] = S
                S = pad_seq
                # Pad contact map
                pad_A = np.zeros((args.pad_len, args.pad_len), dtype=A.dtype)
                pad_A[:current_seq_len, :current_seq_len] = A
                A = pad_A
            elif current_seq_len > args.pad_len:
                # Truncate to pad_len
                print(f"Warning: {prot} sequence length {current_seq_len} > pad_len {args.pad_len}, truncating")
                S = S[:args.pad_len, :]
                A = A[:args.pad_len, :args.pad_len]

            # ##
            S = S.reshape(1, *S.shape)
            A = A.reshape(1, *A.shape)

            # results
            proteins.append(prot)
            Y_pred.append(model.predict([A, S]).reshape(1, output_dim))
            Y_true.append(prot2annot[prot][args.ontology].reshape(1, output_dim))
        
        if skipped_count > 0:
            print(f"### Skipped {skipped_count} proteins due to missing NPZ files")

    pickle.dump({'proteins': np.asarray(proteins),
                 'Y_pred': np.concatenate(Y_pred, axis=0),
                 'Y_true': np.concatenate(Y_true, axis=0),
                 'ontology': args.ontology,
                 'goterms': goterms,
                 'gonames': gonames},
                open(args.model_name + '_results.pckl', 'wb'))
