# filename: PDB2TFRecord_ultra_safe.py
# purpose: An ultra-memory-safe version for converting NPZ to TFRecord.

import csv, os, argparse, numpy as np, tensorflow as tf
from multiprocessing import Pool

# --- Global dict for annotations, loaded once in the main process ---
PROT2ANNOT_GLOBAL = {}

def seq2onehot(seq):
    chars = ['-', 'D', 'G', 'U', 'L', 'N', 'T', 'K', 'H', 'Y', 'W', 'C', 'P', 'V', 'S', 'O', 'I', 'E', 'F', 'X', 'Q', 'A', 'B', 'Z', 'R', 'M']
    vocab_embed = {c: i for i, c in enumerate(chars)}
    vocab_one_hot = np.eye(len(chars), dtype=np.float32)
    indices = [vocab_embed.get(res, vocab_embed['-']) for res in seq]
    return vocab_one_hot[indices]

def _bytes_feature(value):
    if isinstance(value, type(tf.constant(0))): value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def serialize_example(prot_id, cmap, ec_mode):
    global PROT2ANNOT_GLOBAL
    if prot_id not in PROT2ANNOT_GLOBAL: return None
    
    sequence = str(cmap['seqres'])
    feature = {
        'prot_id': _bytes_feature(prot_id.encode('utf-8')),
        'seq_1hot': _float_feature(seq2onehot(sequence).ravel()),
        'L': _int64_feature([len(sequence)]),
        'C_alpha': _float_feature(cmap.get('C_alpha', cmap.get('dist')).ravel()),
    }
    
    annot = PROT2ANNOT_GLOBAL[prot_id]
    if ec_mode:
        feature['ec_labels'] = _int64_feature(annot)
    else:
        feature['mf_labels'] = _int64_feature(annot['molecular_function'])
        feature['bp_labels'] = _int64_feature(annot['biological_process'])
        feature['cc_labels'] = _int64_feature(annot['cellular_component'])
    
    return tf.train.Example(features=tf.train.Features(feature=feature)).SerializeToString()

def process_chunk(chunk_info):
    shard_idx, prot_ids_chunk, npz_dir, tfrecord_prefix, ec_mode = chunk_info
    output_path = f"{tfrecord_prefix}_{shard_idx:02d}.tfrecord"
    
    count, skipped_npz, skipped_annot = 0, 0, 0
    with tf.io.TFRecordWriter(output_path) as writer:
        for prot_id in prot_ids_chunk:
            npz_file = os.path.join(npz_dir, f"{prot_id}.npz")
            if not os.path.exists(npz_file):
                skipped_npz += 1
                continue
            
            serialized_example = serialize_example(prot_id, np.load(npz_file), ec_mode)
            if serialized_example:
                writer.write(serialized_example)
                count += 1
            else:
                skipped_annot += 1
                
    print(f"Shard {shard_idx}: Wrote {count} records. Skipped: {skipped_npz} (no npz), {skipped_annot} (no annot).")
    return count

def load_GO_annot(filename):
    onts = ['molecular_function', 'biological_process', 'cellular_component']
    prot2annot, goterms, gonames = {}, {ont: [] for ont in onts}, {ont: [] for ont in onts}
    with open(filename, mode='r') as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')
        for i in range(3):
            next(reader, None); goterms[onts[i]] = next(reader)
            next(reader, None); gonames[onts[i]] = next(reader)
        next(reader, None)
        for row in reader:
            prot, prot_goterms = row[0], row[1:]
            prot2annot[prot] = {ont: [] for ont in onts}
            for i in range(3):
                indices = [goterms[onts[i]].index(go) for go in prot_goterms[i].split(',') if go]
                prot2annot[prot][onts[i]] = np.zeros(len(goterms[onts[i]]), dtype=np.int64)
                prot2annot[prot][onts[i]][indices] = 1
    return prot2annot, goterms, gonames

def load_list(fname):
    with open(fname, 'r') as f: return [line.strip() for line in f]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-annot', type=str, required=True)
    parser.add_argument('-ec', action="store_true")
    parser.add_argument('-prot_list', type=str, required=True)
    parser.add_argument('-npz_dir', type=str, required=True)
    parser.add_argument('-num_threads', type=int, default=1)
    parser.add_argument('-num_shards', type=int, default=1)
    parser.add_argument('-tfr_prefix', type=str, required=True)
    args = parser.parse_args()

    print("Loading annotations (this may take a moment for large files)...")
    PROT2ANNOT_GLOBAL, _, _ = load_GO_annot(args.annot)
    print(f"Loaded annotations for {len(PROT2ANNOT_GLOBAL)} proteins.")

    prot_list = load_list(args.prot_list)
    
    num_shards = min(args.num_shards, len(prot_list))
    chunks = np.array_split(prot_list, num_shards)
    
    tasks = [(i, chunk.tolist(), args.npz_dir, args.tfr_prefix, args.ec) for i, chunk in enumerate(chunks)]

    print(f"Starting conversion with {args.num_threads} worker processes...")
    with Pool(processes=args.num_threads) as pool:
        results = pool.map(process_chunk, tasks)
    
    total_records = sum(results)
    print("\n-------------------------------------")
    print(f"Conversion complete. Total proteins processed: {total_records}")
    print(f"Generated {len(results)} TFRecord shards with prefix: {args.tfr_prefix}")
    print("-------------------------------------")