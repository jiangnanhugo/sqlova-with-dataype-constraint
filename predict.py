#!/usr/bin/env python

# Use existing model to predict sql from tables and questions.
#
# For example, you can get a pretrained model from https://github.com/naver/sqlova/releases:
#    https://github.com/naver/sqlova/releases/download/SQLova-parameters/model_bert_best.pt
#    https://github.com/naver/sqlova/releases/download/SQLova-parameters/model_best.pt
#
# Make sure you also have the following support files (see README for where to get them):
#    - bert_config_uncased_*.json
#    - vocab_uncased_*.txt
#
# Finally, you need some data - some files called:
#    - <split>.db
#    - <split>.jsonl
#    - <split>.tables.jsonl
#    - <split>_tok.jsonl         # derived using annotate_ws.py
# You can play with the existing train/dev/test splits, or make your own with
# the add_csv.py and add_question.py utilities.
#
# Once you have all that, you are ready to predict, using:
#   python predict.py \
#     --bert_type_abb uL \       # need to match the architecture of the model you are using
#     --model_file <path to models>/model_best.pt            \
#     --bert_model_file <path to models>/model_bert_best.pt  \
#     --bert_path <path to bert_config/vocab>  \
#     --result_path <where to place results>                 \
#     --data_path <path to db/jsonl/tables.jsonl>            \
#     --split <split>
#
# Results will be in a file called results_<split>.jsonl in the result_path.

import argparse, os
from sqlnet.dbengine import DBEngine
from sqlova.utils.utils_wikisql import *
from train import construct_hyper_param, get_models

# This is a stripped down version of the test() method in train.py - identical, except:
#   - does not attempt to measure accuracy and indeed does not expect the data to be labelled.
#   - saves plain text sql queries.
#
def predict(data_loader, data_table, model, model_bert, bert_config, tokenizer,
            max_seq_length,
            num_target_layers, detail=False, st_pos=0, cnt_tot=1, EG=False, beam_size=4,
            path_db=None, dset_name='test', constraint=True):

    model.eval()
    model_bert.eval()

    ave_loss = 0
    cnt = 0
    cnt_sc = 0
    cnt_sa = 0
    cnt_wn = 0
    cnt_wc = 0
    cnt_wo = 0
    cnt_wv = 0
    cnt_wvi = 0
    cnt_lx = 0
    cnt_x = 0

    cnt_list = []

    engine = DBEngine(os.path.join(path_db, f"{dset_name}.db"))
    results = []
    for iB, t in enumerate(data_loader):
        cnt += len(t)
        nlu, nlu_t, sql_i, sql_q, sql_t, tb, hs_t, hds = get_fields(t, data_table, no_hs_t=True, no_sql_t=True)
        g_sc, g_sa, g_wn, g_wc, g_wo, g_wv = get_g(sql_i)
        g_wvi_corenlp = get_g_wvi_corenlp(t)
        wemb_n, wemb_h, l_n, l_hpu, l_hs, \
        nlu_tt, t_to_tt_idx, tt_to_t_idx \
            = get_wemb_bert(bert_config, model_bert, tokenizer, nlu_t, hds, max_seq_length,
                            num_out_layers_n=num_target_layers, num_out_layers_h=num_target_layers)
        try:
            g_wvi = get_g_wvi_bert_from_g_wvi_corenlp(t_to_tt_idx, g_wvi_corenlp)
            g_wv_str, g_wv_str_wp = convert_pr_wvi_to_string(g_wvi, nlu_t, nlu_tt, tt_to_t_idx, nlu)

        except:
            # Exception happens when where-condition is not found in nlu_tt.
            # In this case, that train example is not used.
            # During test, that example considered as wrongly answered.
            for b in range(len(nlu)):
                results1 = {}
                results1["error"] = "Skip happened"
                results1["nlu"] = nlu[b]
                results1["table_id"] = tb[b]["id"]
                results.append(results1)
            continue

        if not EG:
            # No Execution guided decoding
            s_sc, s_sa, s_wn, s_wc, s_wo, s_wv = model(wemb_n, l_n, wemb_h, l_hpu, l_hs, constraint=constraint, tb=tb)
            pr_sc, pr_sa, pr_wn, pr_wc, pr_wo, pr_wvi = pred_sw_se(s_sc, s_sa, s_wn, s_wc, s_wo, s_wv, )
            pr_wv_str, pr_wv_str_wp = convert_pr_wvi_to_string(pr_wvi, nlu_t, nlu_tt, tt_to_t_idx, nlu)
            pr_sql_i = generate_sql_i(pr_sc, pr_sa, pr_wn, pr_wc, pr_wo, pr_wv_str, nlu)
        else:
            # Execution guided decoding
            prob_sca, prob_w, prob_wn_w, pr_sc, pr_sa, pr_wn, pr_sql_i = model.beam_forward(wemb_n, l_n, wemb_h, l_hpu,
                                                                                            l_hs, engine, tb,
                                                                                            nlu_t, nlu_tt,
                                                                                            tt_to_t_idx, nlu,
                                                                                            beam_size=beam_size,
                                                                                            constraint=constraint)
            # sort and generate
            pr_wc, pr_wo, pr_wv, pr_sql_i = sort_and_generate_pr_w(pr_sql_i)
            # Following variables are just for consistency with no-EG case.
            pr_wvi = None # not used
            pr_wv_str=None
            pr_wv_str_wp=None

        g_sql_q = generate_sql_q(sql_i, tb)
        pr_sql_q = generate_sql_q(pr_sql_i, tb)

        for b, (pr_sql_i1, pr_sql_q1) in enumerate(zip(pr_sql_i, pr_sql_q)):
            results1 = {}
            results1["query"] = pr_sql_i1
            results1["table_id"] = tb[b]["id"]
            results1["nlu"] = nlu[b]
            results1["sql"] = pr_sql_q1
            results.append(results1)

        cnt_sc1_list, cnt_sa1_list, cnt_wn1_list, \
        cnt_wc1_list, cnt_wo1_list, \
        cnt_wvi1_list, cnt_wv1_list = get_cnt_sw_list(g_sc, g_sa, g_wn, g_wc, g_wo, g_wvi,
                                                      pr_sc, pr_sa, pr_wn, pr_wc, pr_wo, pr_wvi,
                                                      sql_i, pr_sql_i,
                                                      mode='test')
        cnt_lx1_list = get_cnt_lx_list(cnt_sc1_list, cnt_sa1_list, cnt_wn1_list, cnt_wc1_list,
                                       cnt_wo1_list, cnt_wv1_list)

        # Execution accura y test
        cnt_x1_list = []
        # lx stands for logical form accuracy

        # Execution accuracy test.
        cnt_x1_list, g_ans, pr_ans = get_cnt_x_list(engine, tb, g_sc, g_sa, sql_i, pr_sc, pr_sa, pr_sql_i)

        # stat
        # ave_loss += loss.item()

        # count
        cnt_sc += sum(cnt_sc1_list)
        cnt_sa += sum(cnt_sa1_list)
        cnt_wn += sum(cnt_wn1_list)
        cnt_wc += sum(cnt_wc1_list)
        cnt_wo += sum(cnt_wo1_list)
        cnt_wv += sum(cnt_wv1_list)
        cnt_wvi += sum(cnt_wvi1_list)
        cnt_lx += sum(cnt_lx1_list)
        cnt_x += sum(cnt_x1_list)

        current_cnt = [cnt_tot, cnt, cnt_sc, cnt_sa, cnt_wn, cnt_wc, cnt_wo, cnt_wv, cnt_wvi, cnt_lx, cnt_x]
        cnt_list1 = [cnt_sc1_list, cnt_sa1_list, cnt_wn1_list, cnt_wc1_list, cnt_wo1_list, cnt_wv1_list, cnt_lx1_list,
                     cnt_x1_list]
        cnt_list.append(cnt_list1)
        # report
        # if detail:
        #     report_detail(hds, nlu,
        #                   g_sc, g_sa, g_wn, g_wc, g_wo, g_wv, g_wv_str, g_sql_q, g_ans,
        #                   pr_sc, pr_sa, pr_wn, pr_wc, pr_wo, pr_wv_str, pr_sql_q, pr_ans,
        #                   cnt_list1, current_cnt)
        print('Current epoch: processed %d batches' % iB, end='\r', flush=True)
    print('')

    ave_loss /= cnt
    acc_sc = cnt_sc / cnt
    acc_sa = cnt_sa / cnt
    acc_wn = cnt_wn / cnt
    acc_wc = cnt_wc / cnt
    acc_wo = cnt_wo / cnt
    acc_wvi = cnt_wvi / cnt
    acc_wv = cnt_wv / cnt
    acc_lx = cnt_lx / cnt
    acc_x = cnt_x / cnt

    acc = [ave_loss, acc_sc, acc_sa, acc_wn, acc_wc, acc_wo, acc_wvi, acc_wv, acc_lx, acc_x]
    return acc, results, cnt_list


## Set up hyper parameters and paths
parser = argparse.ArgumentParser()
parser.add_argument("--model_file", default='./saved/model_best.pt', help='model file to use (e.g. model_best.pt)')
parser.add_argument("--bert_model_file", default='./saved/model_bert_best.pt', help='bert model file to use (e.g. model_bert_best.pt)')
parser.add_argument("--bert_path", default='./data/wikisql_tok', help='path to bert files (bert_config*.json etc)')
parser.add_argument("--data_path", default='./data/wikisql_tok', help='path to *.jsonl and *.db files')
parser.add_argument("--split", default='test', help='prefix of jsonl and db files (e.g. dev)')
parser.add_argument("--result_path", default='./saved', help='directory in which to place results')
# parser.set_defaults(constraint=True)
# parser.add_argument('--no-constr',
#                         dest='constraint',
#                         action='store_false',
#                         help="If present, Execution guided decoding is used in test.")
# parser.set_defaults(EG=True)
# parser.add_argument("--no-eg", dest='EG', action='store_false', help='directory in which to place results')

args = construct_hyper_param(parser)

BERT_PT_PATH = args.bert_path
path_save_for_evaluation = args.result_path

# Load pre-trained models
path_model_bert = args.bert_model_file
path_model = args.model_file
args.no_pretraining = True  # counterintuitive, but avoids loading unused models
model, model_bert, tokenizer, bert_config = get_models(args, BERT_PT_PATH, trained=True, path_model_bert=path_model_bert, path_model=path_model)

# Load data
dev_data, dev_table = load_wikisql_data(args.data_path, mode=args.split, toy_model=args.toy_model, toy_size=args.toy_size, no_hs_tok=True)
dev_loader = torch.utils.data.DataLoader(
    batch_size=args.bS,
    dataset=dev_data,
    shuffle=False,
    num_workers=1,
    collate_fn=lambda x: x  # now dictionary values are not merged!
)

# Run prediction
with torch.no_grad():
    acc_test, results, cnt_list = predict(dev_loader,
                      dev_table,
                      model,
                      model_bert,
                      bert_config,
                      tokenizer,
                      args.max_seq_length,
                      args.num_target_layers,
                      detail=False,
                      path_db=args.data_path,
                      st_pos=0,
                      dset_name=args.split, EG=args.EG,
                      constraint=args.constraint)

def print_result(acc, dname):
    ave_loss, acc_sc, acc_sa, acc_wn, acc_wc, acc_wo, acc_wvi, acc_wv, acc_lx, acc_x = acc

    print(f'{dname} results ------------')
    print(
        f" ave loss: {ave_loss}, acc_sc: {acc_sc:.3f}, acc_sa: {acc_sa:.3f}, acc_wn: {acc_wn:.3f}, \
        acc_wc: {acc_wc:.3f}, acc_wo: {acc_wo:.3f}, acc_wvi: {acc_wvi:.3f}, acc_wv: {acc_wv:.3f}, acc_lx: {acc_lx:.4f}, acc_x: {acc_x:.3f}"
    )

print_result(acc_test, 'test')
# Save results
save_for_evaluation(path_save_for_evaluation, results, args.split)
