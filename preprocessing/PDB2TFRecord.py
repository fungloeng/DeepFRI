import csv
import os.path
import argparse
import re

import numpy as np
import tensorflow as tf
import multiprocessing


def seq2onehot(seq):
    """Create 26-dim embedding"""
    chars = ['-', 'D', 'G', 'U', 'L', 'N', 'T', 'K', 'H', 'Y', 'W', 'C', 'P',
             'V', 'S', 'O', 'I', 'E', 'F', 'X', 'Q', 'A', 'B', 'Z', 'R', 'M']
    vocab_size = len(chars)
    vocab_embed = dict(zip(chars, range(vocab_size)))
    # Use '-' (index 0) as default for unknown characters
    default_char = '-'
    default_idx = vocab_embed[default_char]

    # Convert vocab to one-hot
    vocab_one_hot = np.zeros((vocab_size, vocab_size), int)
    for _, val in vocab_embed.items():
        vocab_one_hot[val, val] = 1

    # Map each character to its index, using default for unknown chars
    embed_x = [vocab_embed.get(v, default_idx) for v in seq]
    seqs_x = np.array([vocab_one_hot[j, :] for j in embed_x])

    return seqs_x


def load_list(fname):
    """
    Load PDB chains
    """
    pdb_chain_list = []
    fRead = open(fname, 'r')
    for line in fRead:
        pdb_chain_list.append(line.strip())
    fRead.close()

    return pdb_chain_list


LABEL_TO_KEY = {
    'molecular_function': 'molecular_function',
    'biological_process': 'biological_process',
    'cellular_component': 'cellular_component',
    'primary_function': 'primary_function'
}

ONTOLOGY_ALIASES = {
    'mf': 'molecular_function',
    'bp': 'biological_process',
    'cc': 'cellular_component',
    'pf': 'primary_function'
}

ONT_FEATURE_MAP = {
    'molecular_function': 'mf_labels',
    'biological_process': 'bp_labels',
    'cellular_component': 'cc_labels',
    'primary_function': 'pf_labels'
}


def _parse_label(text):
    match = re.search(r'\(([^)]+)\)', text)
    if match:
        return match.group(1).strip().lower()
    return text.strip().lower()


def load_GO_annot(filename, selected_onts=None):
    """ Load GO annotations (supports optional PF section) """
    onts = list(LABEL_TO_KEY.values())
    if selected_onts is not None:
        selected_onts = [ONTOLOGY_ALIASES.get(ont, ont) for ont in selected_onts]
        selected_onts = [ont for ont in selected_onts if ont in onts]
        if not selected_onts:
            selected_onts = onts
    active_onts = selected_onts or onts
    prot2annot = {}
    goterms = {ont: [] for ont in onts}
    gonames = {ont: [] for ont in onts}
    with open(filename, mode='r') as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')

        # 读取所有 GO term sections
        while True:
            row = next(reader)
            if not row:
                continue
            if row[0].startswith("### PDB-chain"):
                header_row = row
                break
            if row[0].startswith("### GO-terms"):
                label = _parse_label(row[0])
                key = LABEL_TO_KEY.get(label)
                terms = next(reader)
                next(reader, None)  # skip GO-names header
                names = next(reader)
                if key:
                    goterms[key] = terms
                    gonames[key] = names
                continue

        # 每一列对应的 ont key
        column_keys = [LABEL_TO_KEY.get(_parse_label(col), None) for col in header_row[1:]]

        for row in reader:
            if not row:
                continue
            prot = row[0]
            prot2annot[prot] = {ont: np.zeros(len(goterms[ont]), dtype=np.uint8) for ont in active_onts}
            for value, key in zip(row[1:], column_keys):
                if key is None or key not in active_onts or len(goterms[key]) == 0 or value == '':
                    continue
                # Handle missing GO terms gracefully
                indices = []
                for goterm in value.split(','):
                    goterm = goterm.strip()
                    if goterm and goterm in goterms[key]:
                        indices.append(goterms[key].index(goterm))
                    elif goterm:
                        # Log warning for missing GO terms but continue processing
                        print(f"Warning: GO term '{goterm}' not found in {key} list for protein {prot}. Skipping.")
                if indices:
                    prot2annot[prot][key][indices] = 1.0

    goterms = {ont: goterms[ont] for ont in active_onts}
    gonames = {ont: gonames[ont] for ont in active_onts}
    return prot2annot, goterms, gonames


def load_EC_annot(filename):
    """ Load EC annotations """
    prot2annot = {}
    ec_numbers = []
    with open(filename, mode='r') as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')

        # molecular function
        next(reader, None)  # skip the headers
        ec_numbers = next(reader)
        next(reader, None)  # skip the headers
        for row in reader:
            prot, prot_ec_numbers = row[0], row[1]
            ec_indices = [ec_numbers.index(ec_num) for ec_num in prot_ec_numbers.split(',')]
            prot2annot[prot] = np.zeros(len(ec_numbers), dtype=np.int64)
            prot2annot[prot][ec_indices] = 1.0
    return prot2annot, ec_numbers


class GenerateTFRecord(object):
    def __init__(self, prot_list, prot2annot, ec, npz_dir, tfrecord_fn, num_shards=30, records_per_shard=500, active_onts=None):
        self.prot_list = prot_list
        self.prot2annot = prot2annot
        self.ec = ec
        self.npz_dir = npz_dir
        self.tfrecord_fn = tfrecord_fn
        if not self.ec:
            self.active_onts = active_onts or list(ONT_FEATURE_MAP.keys())
        else:
            self.active_onts = []
        if records_per_shard <= 0:
            records_per_shard = 500

        auto_shards = max(1, int(np.ceil(len(prot_list) / records_per_shard)))
        self.num_shards = max(num_shards, auto_shards)
        if self.num_shards != num_shards:
            print(f"[Info] Adjusted shard count from {num_shards} to {self.num_shards} based on target {records_per_shard} records/shard")

        shard_size = max(1, len(prot_list)//self.num_shards)
        indices = [(i*(shard_size), (i+1)*(shard_size)) for i in range(0, self.num_shards)]
        indices[-1] = (indices[-1][0], len(prot_list))
        self.indices = indices

    def _bytes_feature(self, value):
        """Returns a bytes_list from a string / byte."""
        if not isinstance(value, list):
            value = [value]
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

    def _float_feature(self, value):
        """Returns a float_list from a float / double."""
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))

    def _dtype_feature(self):
        return lambda array: tf.train.Feature(int64_list=tf.train.Int64List(value=array.astype(np.int64, copy=False)))

    def _int_feature(self, value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def _serialize_example(self, prot_id, sequence, ca_dist_matrix, cb_dist_matrix):
        labels = self._dtype_feature()

        d_feature = {}
        # load appropriate tf.train.Featur class depending on dtype
        d_feature['prot_id'] = self._bytes_feature(prot_id.encode())
        d_feature['seq_1hot'] = self._float_feature(seq2onehot(sequence).reshape(-1))
        d_feature['L'] = self._int_feature(len(sequence))

        if self.ec:
            d_feature['ec_labels'] = labels(self.prot2annot[prot_id])
        else:
            for ont in self.active_onts:
                feature_key = ONT_FEATURE_MAP.get(ont)
                if feature_key and ont in self.prot2annot[prot_id]:
                    d_feature[feature_key] = labels(self.prot2annot[prot_id][ont])

        # Store distance matrices as float16 bytes to reduce TFRecord size; decode at read time.
        d_feature['ca_dist_matrix'] = self._bytes_feature(
            ca_dist_matrix.astype(np.float16, copy=False).tobytes()
        )
        d_feature['cb_dist_matrix'] = self._bytes_feature(
            cb_dist_matrix.astype(np.float16, copy=False).tobytes()
        )

        example = tf.train.Example(features=tf.train.Features(feature=d_feature))
        return example.SerializeToString()

    def _convert_numpy_folder(self, idx):
        tfrecord_fn = self.tfrecord_fn + '_%0.2d-of-%0.2d.tfrecords' % (idx, self.num_shards)
        writer = tf.io.TFRecordWriter(tfrecord_fn)
        tmp_prot_list = self.prot_list[self.indices[idx][0]:self.indices[idx][1]]
        
        print ("### Serializing %d examples into %s" % (len(tmp_prot_list), tfrecord_fn))
        
        success_count = 0
        skip_count = 0
        error_count = 0

        for i, prot in enumerate(tmp_prot_list):
            if i % 500 == 0:
                print ("### Iter = %d/%d (success=%d, skip=%d, error=%d)" % (i, len(tmp_prot_list), success_count, skip_count, error_count))
            
            pdb_file = os.path.join(self.npz_dir, prot + '.npz')
            if not os.path.isfile(pdb_file):
                # Try alternative paths
                pdb_file_alt = self.npz_dir + '/' + prot + '.npz'
                if not os.path.isfile(pdb_file_alt):
                    skip_count += 1
                    if i < 10:  # Only print first few missing files
                        print(f"Warning: {prot} npz file not found")
                    continue
                pdb_file = pdb_file_alt
            
            try:
                # Use context manager to ensure file is closed immediately
                with np.load(pdb_file) as cmap:
                    # Check if protein has annotation
                    if prot not in self.prot2annot:
                        skip_count += 1
                        if i < 10:
                            print(f"Warning: {prot} not found in annotations, skipping")
                        continue
                    
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
                    
                    # Filter to allowed amino acid characters and convert to uppercase
                    allowed = set("ACDEFGHIKLMNPQRSTVWYBXZOU-")
                    sequence = "".join([c for c in sequence.upper() if c in allowed])
                    
                    if len(sequence) == 0:
                        skip_count += 1
                        if i < 10:
                            print(f"Warning: {prot} has empty sequence after filtering, skipping")
                        continue
                    
                    # Extract arrays while file is open; convert to float16 to reduce memory footprint
                    ca_dist_matrix = np.asarray(cmap['C_alpha'], dtype=np.float16)
                    if 'C_beta' in cmap:
                        cb_dist_matrix = np.asarray(cmap['C_beta'], dtype=np.float16)
                    else:
                        cb_dist_matrix = ca_dist_matrix  # share memory to avoid doubling usage
                        if i < 5:  # Only print warning for first few files
                            print(f"Warning: {prot} missing C_beta, using C_alpha as substitute")
                
                # File is now closed, serialize and write
                example = self._serialize_example(prot, sequence, ca_dist_matrix, cb_dist_matrix)
                writer.write(example)
                success_count += 1
                
                # Explicitly delete large arrays to free memory immediately
                del sequence, ca_dist_matrix, cb_dist_matrix, example
                
            except KeyError as e:
                error_count += 1
                if i < 10:
                    print(f"Error: {prot} missing required key in npz: {e}")
            except Exception as e:
                error_count += 1
                if i < 10:
                    print(f"Error processing {prot}: {e}")
                # Continue processing other proteins even if one fails
        
        writer.close()
        print ("Writing {} done! (success={}, skip={}, error={})".format(tfrecord_fn, success_count, skip_count, error_count))

    def run(self, num_threads):
        # Limit number of threads to avoid OOM
        # Each thread processes one shard, and each shard loads many npz files
        # Use fewer threads if we have many shards to reduce memory pressure
        effective_threads = min(num_threads, self.num_shards, 4)  # Cap at 4 to reduce memory usage
        
        if effective_threads < num_threads:
            print(f"Warning: Reducing threads from {num_threads} to {effective_threads} to avoid memory issues")
        
        pool = multiprocessing.Pool(processes=effective_threads)
        shards = [idx for idx in range(0, self.num_shards)]
        
        try:
            pool.map(self._convert_numpy_folder, shards)
        finally:
            pool.close()
            pool.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-annot', type=str, default='./data/nrPDB-EC_2020.04_annot.tsv', help="Input file (*.tsv) with preprocessed annotations.")
    parser.add_argument('-ec', help="Use EC annotations.", action="store_true")
    parser.add_argument('-prot_list', type=str, default='./data/nrPDB-GO_2019.06.18_train.txt',
                        help="Input file (*.txt) with a set of protein IDs with distMAps in npz_dir.")
    parser.add_argument('-npz_dir', type=str, default='./data/annot_pdb_chains_npz/',
                        help="Directory with distance maps saved in *.npz format to be loaded.")
    parser.add_argument('-num_threads', type=int, default=20, help="Number of threads (CPUs) to use in the computation.")
    parser.add_argument('-num_shards', type=int, default=20, help="Number of tfrecord files per protein set.")
    parser.add_argument('--records_per_shard', type=int, default=400,
                        help="Target number of records per shard; actual shard count will be max(num_shards, len(prot_list)/records_per_shard).")
    parser.add_argument('--ontology', type=str, default='all', choices=['mf', 'bp', 'cc', 'pf', 'all'],
                        help="Ontology focus for TFRecord generation; reduces memory usage when set.")
    parser.add_argument('-tfr_prefix', type=str, default='/mnt/ceph/users/vgligorijevic/ContactMaps/TFRecords/PDB_GO_train',
                        help="Directory with tfrecord files for model training.")
    args = parser.parse_args()

    prot_list = load_list(args.prot_list)
    ont_arg = args.ontology.lower()
    if ont_arg == 'all':
        selected_onts = None
    else:
        selected_onts = [ONTOLOGY_ALIASES.get(ont_arg, ont_arg)]

    if args.ec:
        prot2annot, _ = load_EC_annot(args.annot)
        active_onts = []
    else:
        prot2annot, _, _ = load_GO_annot(args.annot, selected_onts=selected_onts)
        active_onts = selected_onts or list(ONT_FEATURE_MAP.keys())

    tfr = GenerateTFRecord(
        prot_list,
        prot2annot,
        args.ec,
        args.npz_dir,
        args.tfr_prefix,
        num_shards=args.num_shards,
        records_per_shard=args.records_per_shard,
        active_onts=active_onts
    )
    tfr.run(num_threads=args.num_threads)
