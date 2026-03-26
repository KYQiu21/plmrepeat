import os
import argparse
import pickle

import torch
from Bio import SeqIO

from plmrepeat.base import AllRepeatDector
from plmrepeat.utils import select_length, save_result


def create_arg_parser():
    parser = argparse.ArgumentParser(
        description="pLM-Repeat utilizes protein embeddings to detect repeats"
    )

    parser.add_argument(
        '--emb',
        type=str,
        required=True,
        help='path to the residue-wise embedding torch-readable file of the query protein'
    )
    parser.add_argument(
        '--seq',
        type=str,
        required=True,
        help='path to the fasta file of the query protein'
    )
    parser.add_argument(
        '--out',
        type=str,
        required=True,
        help='path to the output directory'
    )

    parser.add_argument(
        '--win_len',
        default=15,
        type=int,
        help='length of sliding window in sub-path selection'
    )
    parser.add_argument(
        '--min_span',
        default=15,
        type=int,
        help='minimum span length in traceback'
    )
    parser.add_argument(
        '--ssa-score',
        default=0.3,
        type=float,
        help='alignment score threshold for self-sequence alignment'
    )
    parser.add_argument(
        '--ssa-sigma-factor',
        default=2.0,
        type=float,
        help='sigma factor for self-sequence alignment'
    )

    parser.add_argument(
        '--scan-ssa-score',
        default=0.2,
        type=float,
        help='alignment score threshold for repeat extraction'
    )
    parser.add_argument(
        '--scan-sigma-factor',
        default=2.0,
        type=float,
        help='sigma factor for repeat extraction'
    )
    parser.add_argument(
        '--weighted-repeat-emb',
        action='store_true',
        help='whether to use re-weighted embeddings for the representative repeat'
    )

    parser.add_argument(
        '--transitivity',
        action='store_true',
        help='whether to use transitivity'
    )
    parser.add_argument(
        '--transitive-score',
        default=0.0,
        type=float,
        help='alignment score threshold for transitive traces'
    )
    parser.add_argument(
        '--repeat-len',
        default=10,
        type=int,
        help='minimum length of detected repeats'
    )
    parser.add_argument(
        '--repeat-avglen',
        default=10,
        type=int,
        help='minimum average length of detected repeats except for the longest one'
    )
    parser.add_argument(
        '--repeat-score',
        default=10,
        type=int,
        help='minimum score of detected repeats'
    )
    parser.add_argument(
        '--overlap-threshold',
        default=0.3,
        type=float,
        help='maximum overlap threshold of detected traces to accept'
    )
    parser.add_argument(
        '--metrics',
        default='repeat_total_len',
        type=str,
        help='metrics used to select repeat length'
    )
    parser.add_argument(
        '--draw',
        action='store_true',
        help='whether to draw intermediate results'
    )

    return parser


def main():
    parser = create_arg_parser()
    args = parser.parse_args()

    emb = torch.load(args.emb).float().numpy()
    seq = [str(record.seq) for record in SeqIO.parse(args.seq, 'fasta')][0]
    save_path = args.out
    os.makedirs(save_path, exist_ok=True)

    assert len(seq) == emb.shape[0], "Sequence length and embedding length do not match."

    repeat_job = AllRepeatDector(
        emb=emb,
        seq=seq,
        win_len=args.win_len,
        min_span=args.min_span,
        ssa_score_threshold=args.ssa_score,
        ssa_sigma_factor=args.ssa_sigma_factor,
        scan_score_threshold=args.scan_ssa_score,
        scan_sigma_factor=args.scan_sigma_factor,
        scan_with_weighted=args.weighted_repeat_emb,
        overlap_threshold=args.overlap_threshold,
        use_transitivity=args.transitivity,
    )

    print("transitivity", args.transitivity)

    repeat_job.search_all_repeat()

    result = repeat_job.all_repeat_info_dict
    result = {
        repeat_type: repeat_info
        for repeat_type, repeat_info in result.items()
        if len(repeat_info) > 0
    }

    with open(os.path.join(save_path, 'all_result.pkl'), 'wb') as f:
        pickle.dump(result, f)

    for repeat_idx, single_repeat_result in result.items():
        select_result_info_dict, repeat_range_list = select_length(
            single_repeat_result,
            metrics=args.metrics
        )

        save_result(
            select_result_info_dict,
            repeat_idx,
            draw=args.draw,
            save_path=save_path,
            repeat_idx=repeat_idx
        )

        if len(select_result_info_dict) >= 2:
            print(f"Detected type {repeat_idx} repeat ranges: {repeat_range_list}")
        else:
            print("No repeats detected!")


if __name__ == '__main__':
    main()
