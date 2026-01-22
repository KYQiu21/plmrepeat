import json
import os
import pickle
from Bio import SeqIO
import torch
from base import AllRepeatDector
from utils import select_length

pdb_list = [str(record.id) for record in SeqIO.parse(
    '/Users/kaiyuqiu/Documents/plmrepeat/benchmark/dataset/repeatsdb_seq/repeatsdb_df_pdb_chain_seq_0.9_sort.fasta',
    'fasta')]
seq_list = [str(record.seq) for record in SeqIO.parse(
    '/Users/kaiyuqiu/Documents/plmrepeat/benchmark/dataset/repeatsdb_seq/repeatsdb_df_pdb_chain_seq_0.9_sort.fasta',
    'fasta')]
idx_list = [str(i) for i in range(len(pdb_list))]
idx_pdb_dict = dict(zip(idx_list, pdb_list))
pdb_idx_dict = dict(zip(pdb_list, idx_list))
idx_seq_dict = dict(zip(idx_list, seq_list))
pdb_seq_dict = dict(zip(pdb_list, seq_list))

with open('/Users/kaiyuqiu/Documents/plmrepeat/benchmark/dataset/repeatsdb_fold_class_dict.pkl', 'rb') as f:
    repeatsdb_pdb_fold_dict = pickle.load(f)

print(pdb_idx_dict['2QJ6_A'])
entry = str(1120)
emb_file = f'/Users/kaiyuqiu/Documents/plmrepeat/benchmark/dataset/repeatsdb_df_pdb_chain_seq_0.9_emb/{entry}.emb'
emb = torch.load(emb_file).float().numpy()
seq = idx_seq_dict[entry]
print(seq)

emb = torch.load('/Users/kaiyuqiu/Documents/plmrepeat/1w6s.emb').float().numpy()
seq = [str(record.seq) for record in SeqIO.parse('/Users/kaiyuqiu/Documents/plmrepeat/1W6S.fasta', 'fasta')][0]

print(emb.shape, len(seq), idx_pdb_dict[entry])

RepeatJob = AllRepeatDector(emb=emb, seq=seq, win_len=15, min_span=15,
                            ssa_score_threshold=0.3, ssa_sigma_factor=2.0,
                            scan_score_threshold=0.2, scan_sigma_factor=2.0, scan_with_weighted=False,
                            overlap_threshold=0.3, use_transitivity=True, draw_during_procedure=True)
RepeatJob.search_all_repeat()
# print(RepeatJob.all_repeat_info_dict)
# if RepeatJob.high_ssa_hit and len(RepeatJob.all_repeat_info_dict[0]) == 0:


for repeat_class, repeat_class_info in RepeatJob.all_repeat_info_dict.items():
    for repeat_len, repeat_len_info in repeat_class_info.items():
        print(repeat_len_info['repeat_len'], repeat_len_info['repeat_total_len'], repeat_len_info['trace_cluster_range_list'], repeat_len_info['avg_coverage_to_rep'])
        for name, seq in repeat_len_info['repeat_seq_dict'].items():
            print(seq)

result = RepeatJob.all_repeat_info_dict
result = {repeat_type: repeat_info for repeat_type, repeat_info in result.items() if len(repeat_info) > 0}

# report repeat results
for repeat_idx, repeat_result in result.items():
    single_repeat_result = result[repeat_idx]
    select_result_info_dict, repeat_range_list = select_length(single_repeat_result, metrics='avg_coverage_to_rep')

    if len(select_result_info_dict) >= 2:
        print(f"Detected type {repeat_idx} repeat ranges: ", repeat_range_list)
    else:
        print("No repeats detected!")


# print(RepeatJob.all_repeat_info_dict, RepeatJob.high_ssa_hit)

# full benchmark
# already_list = os.listdir('/Users/kaiyuqiu/Documents/plmrepeat/benchmark/result/plmrepeat/repeatsdb_0.3_new/result_scan0.2/')
# already_list = [file.split('.pkl')[0] for file in already_list]
# print("already_list", len(already_list), already_list)
# wait_pdb_list = [entry for entry in pdb_list if entry not in already_list]
# filter_list = ['5JPQ_M', '3JB9_r', '3RYT_A']
# # '3RYT_A' is an abnormal case
# wait_pdb_list = [entry for entry in wait_pdb_list if entry not in filter_list]
# print("wait_pdb_list", len(wait_pdb_list), wait_pdb_list)
# wait_idx_list = [pdb_idx_dict[entry] for entry in wait_pdb_list]
# # wait_idx_list = []
# # for entry in wait_pdb_list:
# #     if repeatsdb_pdb_fold_dict[entry] == '3.2' and len(pdb_seq_dict[entry]) > 600:
# #         pass
# #     if repeatsdb_pdb_fold_dict[entry] == '3.3' and len(pdb_seq_dict[entry]) > 600:
# #         pass
# #     else:
# #         wait_idx_list.append(pdb_idx_dict[entry])
# print("wait_idx_list", len(wait_idx_list), wait_idx_list)
#
# easy_target_list = []
# for idx in wait_idx_list:
#     pdb = idx_pdb_dict[idx]
#     emb_file = f'/Users/kaiyuqiu/Documents/plmrepeat/benchmark/dataset/repeatsdb_df_pdb_chain_seq_0.9_emb/{idx}.emb'
#     emb = torch.load(emb_file).float().numpy()
#     seq = idx_seq_dict[idx]
#     print(emb.shape, len(seq), idx_pdb_dict[idx], idx)
#
#     RepeatJob = AllRepeatDector(emb, seq, win_len=15, min_span=15, ssa_score_threshold=0.3, ssa_sigma_factor=2.0,
#                                 scan_score_threshold=0.2, scan_sigma_factor=2.0, scan_with_weighted=True, overlap_threshold=0.3, draw_during_procedure=False,
#                                 use_transitivity=True)
#     RepeatJob.search_all_repeat()
#
#     # improve scan score threshold to avoid missing easy targets
#     # scan_score_threshold_list = [0.4, 0.45, 0.5]
#     # if RepeatJob.high_ssa_hit and len(RepeatJob.all_repeat_info_dict[0]) == 0:
#     #     easy_target_list.append(idx)
#     #
#     #     RepeatJob = AllRepeatDector(emb, seq, win_len=15, min_span=15, ssa_score_threshold=0.3, scan_score_threshold=0.4, overlap_threshold=0.3, draw_during_procedure=False)
#     #     RepeatJob.search_all_repeat()
#
#     # save result dict
#     result = RepeatJob.all_repeat_info_dict
#     with open(f'/Users/kaiyuqiu/Documents/plmrepeat/benchmark/result/plmrepeat/repeatsdb_0.3_new/result_scan0.2/{pdb}.pkl', 'wb') as f:
#         pickle.dump(result, f)