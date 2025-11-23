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

    # Convert vocab to one-hot
    vocab_one_hot = np.zeros((vocab_size, vocab_size), int)
    for _, val in vocab_embed.items():
        vocab_one_hot[val, val] = 1

    embed_x = [vocab_embed[v] for v in seq]
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


def _parse_label(text):
    match = re.search(r'\(([^)]+)\)', text)
    if match:
        return match.group(1).strip().lower()
    return text.strip().lower()


def load_GO_annot(filename):
    """ Load GO annotations (supports optional PF section) """
    onts = list(LABEL_TO_KEY.values())
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
            prot2annot[prot] = {ont: np.zeros(len(goterms[ont]), dtype=np.int64) for ont in onts}
            for value, key in zip(row[1:], column_keys):
                if key is None or len(goterms[key]) == 0 or value == '':
                    continue
                indices = [goterms[key].index(goterm) for goterm in value.split(',') if goterm]
                if indices:
                    prot2annot[prot][key][indices] = 1.0
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
    def __init__(self, prot_list, prot2annot, ec, npz_dir, tfrecord_fn, num_shards=30):
        self.prot_list = prot_list
        self.prot2annot = prot2annot
        self.ec = ec
        self.npz_dir = npz_dir
        self.tfrecord_fn = tfrecord_fn
        self.num_shards = num_shards

        shard_size = len(prot_list)//num_shards
        indices = [(i*(shard_size), (i+1)*(shard_size)) for i in range(0, num_shards)]
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
        return lambda array: tf.train.Feature(int64_list=tf.train.Int64List(value=array))

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
            d_feature['mf_labels'] = labels(self.prot2annot[prot_id]['molecular_function'])
            d_feature['bp_labels'] = labels(self.prot2annot[prot_id]['biological_process'])
            d_feature['cc_labels'] = labels(self.prot2annot[prot_id]['cellular_component'])
            # 如果存在 primary_function（注释文件包含PF部分），也序列化 pf_labels
            # 这确保了训练PF模型时所有TFRecord记录都包含pf_labels
            if 'primary_function' in self.prot2annot[prot_id] and len(self.prot2annot[prot_id]['primary_function']) > 0:
                d_feature['pf_labels'] = labels(self.prot2annot[prot_id]['primary_function'])

        d_feature['ca_dist_matrix'] = self._float_feature(ca_dist_matrix.reshape(-1))
        d_feature['cb_dist_matrix'] = self._float_feature(cb_dist_matrix.reshape(-1))

        example = tf.train.Example(features=tf.train.Features(feature=d_feature))
        return example.SerializeToString()

    def _convert_numpy_folder(self, idx):
        tfrecord_fn = self.tfrecord_fn + '_%0.2d-of-%0.2d.tfrecords' % (idx, self.num_shards)
        # writer = tf.python_io.TFRecordWriter(tfrecord_fn)
        writer = tf.io.TFRecordWriter(tfrecord_fn)
        print ("### Serializing %d examples into %s" % (len(self.prot_list), tfrecord_fn))

        tmp_prot_list = self.prot_list[self.indices[idx][0]:self.indices[idx][1]]

        for i, prot in enumerate(tmp_prot_list):
            if i % 500 == 0:
                print ("### Iter = %d/%d" % (i, len(tmp_prot_list)))
            pdb_file = self.npz_dir + '/' + prot + '.npz'
            if os.path.isfile(pdb_file):
                cmap = np.load(pdb_file)
                sequence = str(cmap['seqres'])
                ca_dist_matrix = cmap['C_alpha']
                cb_dist_matrix = cmap['C_beta']

                example = self._serialize_example(prot, sequence, ca_dist_matrix, cb_dist_matrix)
                writer.write(example)
            else:
                print (pdb_file)
        print ("Writing {} done!".format(tfrecord_fn))

    def run(self, num_threads):
        pool = multiprocessing.Pool(processes=num_threads)
        shards = [idx for idx in range(0, self.num_shards)]
        pool.map(self._convert_numpy_folder, shards)


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
    parser.add_argument('-tfr_prefix', type=str, default='/mnt/ceph/users/vgligorijevic/ContactMaps/TFRecords/PDB_GO_train',
                        help="Directory with tfrecord files for model training.")
    args = parser.parse_args()

    prot_list = load_list(args.prot_list)
    if args.ec:
        prot2annot, _ = load_EC_annot(args.annot)
    else:
        prot2annot, _, _ = load_GO_annot(args.annot)

    tfr = GenerateTFRecord(prot_list, prot2annot, args.ec, args.npz_dir, args.tfr_prefix, num_shards=args.num_shards)
    tfr.run(num_threads=args.num_threads)
