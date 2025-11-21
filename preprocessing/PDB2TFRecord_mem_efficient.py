# filename: PDB2TFRecord_mem_efficient.py
# purpose: A memory-efficient version for converting NPZ to TFRecord.

import csv
import os
import argparse
import numpy as np
import tensorflow as tf
from multiprocessing import Pool, cpu_count

# --- Global variables for worker processes ---
# We use global variables to avoid passing large objects between processes
GLOBAL_DATA = {}

def seq2onehot(seq):
    chars = ['-', 'D', 'G', 'U', 'L', 'N', 'T', 'K', 'H', 'Y', 'W', 'C', 'P',
             'V', 'S', 'O', 'I', 'E', 'F', 'X', 'Q', 'A', 'B', 'Z', 'R', 'M']
    vocab_size = len(chars)
    vocab_embed = dict(zip(chars, range(vocab_size)))
    vocab_one_hot = np.zeros((vocab_size, vocab_size), int)
    for _, val in vocab_embed.items():
        vocab_one_hot[val, val] = 1
    embed_x = [vocab_embed.get(v, vocab_embed['-']) for v in seq] # Use .get for safety
    return np.array([vocab_one_hot[j, :] for j in embed_x])

def _bytes_feature(value):
    if isinstance(value, type(tf.constant(0))): value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _serialize_example(prot_id, cmap, prot2annot, ec_mode):
    sequence = str(cmap['seqres'])
    
    feature = {
        'prot_id': _bytes_feature(prot_id.encode('utf-8')),
        'seq_1hot': _float_feature(seq2onehot(sequence).reshape(-1)),
        'L': _int64_feature([len(sequence)]), # Note: must be a list
        'C_alpha': _float_feature(cmap['C_alpha'].reshape(-1)),
    }

    if ec_mode:
        feature['ec_labels'] = _int64_feature(prot2annot[prot_id])
    else:
        feature['mf_labels'] = _int64_feature(prot2annot[prot_id]['molecular_function'])
        feature['bp_labels'] = _int64_feature(prot2annot[prot_id]['biological_process'])
        feature['cc_labels'] = _int64_feature(prot2annot[prot_id]['cellular_component'])
    
    return tf.train.Example(features=tf.train.Features(feature=feature)).SerializeToString()

def worker_init(annot_fn, ec_mode):
    """
    Initializer for each worker process.
    Loads the large annotation file ONCE per process.
    """
    print(f"Worker process {os.getpid()} initializing...")
    global GLOBAL_DATA
    if ec_mode:
        prot2annot, _ = load_EC_annot(annot_fn)
    else:
        prot2annot, _, _ = load_GO_annot(annot_fn)
    GLOBAL_DATA['prot2annot'] = prot2annot

def process_chunk(chunk_info):
    """
    Function executed by each worker.
    Processes a chunk of protein IDs.
    """
    shard_idx, prot_ids_chunk, npz_dir, tfrecord_prefix, ec_mode = chunk_info
    global GLOBAL_DATA
    prot2annot = GLOBAL_DATA['prot2annot']
    
    output_path = f"{tfrecord_prefix}_{shard_idx:02d}.tfrecord"
    
    count = 0
    with tf.io.TFRecordWriter(output_path) as writer:
        for prot_id in prot_ids_chunk:
            npz_file = os.path.join(npz_dir, f"{prot_id}.npz")
            if os.path.exists(npz_file) and prot_id in prot2annot:
                try:
                    cmap = np.load(npz_file)
                    example = _serialize_example(prot_id, cmap, prot2annot, ec_mode)
                    writer.write(example)
                    count += 1
                except Exception as e:
                    print(f"Warning: Failed to process {prot_id}. Error: {e}")
            
    print(f"Shard {shard_idx}: Wrote {count} records to {output_path}")
    return count

# (load_GO_annot and load_EC_annot are copied from the original script)
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

def load_EC_annot(filename):
    # Implementation from your script
    pass

def load_list(fname):
    with open(fname, 'r') as f:
        return [line.strip() for line in f]

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

    print("Loading protein list...")
    prot_list = load_list(args.prot_list)
    
    # Split protein list into chunks for each shard
    num_shards = min(args.num_shards, len(prot_list))
    chunks = np.array_split(prot_list, num_shards)
    
    # Create a list of tasks for the pool
    tasks = []
    for i, chunk in enumerate(chunks):
        tasks.append((i, chunk, args.npz_dir, args.tfr_prefix, args.ec))

    print(f"Starting conversion with {args.num_threads} worker processes...")
    # Create a pool with an initializer
    with Pool(processes=args.num_threads, initializer=worker_init, initargs=(args.annot, args.ec)) as pool:
        results = pool.map(process_chunk, tasks)
    
    total_records = sum(results)
    print("\n-------------------------------------")
    print(f"Conversion complete.")
    print(f"Total proteins processed: {total_records}")
    print(f"Generated {len(results)} TFRecord shards with prefix: {args.tfr_prefix}")
    print("-------------------------------------")