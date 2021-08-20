#coding:utf-8
from collections import defaultdict,Counter
import time
import numpy as np
import logging
import json
from helpers.metrics import compute_hits_at_N,compute_Hits_K_by_Rank,compute_MR_MRR_by_Rank

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"       # 使用第二块GPU（从0开始）


def compute_metrics(ranks,is_print=True):
    '''计算Hits@N/MRR'''
    mr,mrr=compute_MR_MRR_by_Rank(ranks)
    hit_1=compute_Hits_K_by_Rank(ranks,topK=1)
    hit_3=compute_Hits_K_by_Rank(ranks,topK=3)
    hit_10=compute_Hits_K_by_Rank(ranks,topK=10)
    hit_30=compute_Hits_K_by_Rank(ranks,topK=50)
    if is_print:
        print("MR   MRR hit@1    hit@3   hit@10")
        print("%s\t%s\t%s\t%s\t%s"%(mr,mrr,hit_1,hit_3,hit_10))
    return mr,mrr,hit_1,hit_3,hit_10,hit_30

def rank_args(preds,labels):
    #排序
    results=np.argsort(preds,axis=-1)
    ranks=np.where(results==labels)[1]
    args=[res[:100] for res in results]
    return ranks,args

def evaluate_model(sess,trainer,test_dataHelper,batch_size=2,mini_batch=4,is_save_top=False):
    entity_embeddings, relation_embeddings, relation_embeddings_reverse = trainer.predict_embeddings(sess, test_dataHelper)
    data_dir=test_dataHelper.data_dir
    id2entity=test_dataHelper.id2entity
    id2relation=test_dataHelper.id2relation
    tail_ranks=[]
    head_ranks=[]
    rank_args_tail=[]
    rank_args_head=[]
    #predict
    gen=test_dataHelper.batch_generator(batch_size=batch_size,start_index=0,test_num=None,mini_batch=mini_batch)
    triples=[]#test_dataHelper.triples
    relations=test_dataHelper.relation2id
    start_time=time.time()
    i=0
    for batch_datas in gen:
        if batch_datas is None:
            continue
        batch_datas["entity_embedding"] = entity_embeddings
        batch_datas["relation_embedding"] = relation_embeddings
        batch_datas["relation_embedding_reverse"] = relation_embeddings_reverse
        i+=1
        batch_pred_scores,batch_pred_scores_reverse=trainer.predict_batch(sess,batch_datas)
        masks=batch_datas["masks"]
        masks_reverse=batch_datas["masks_reverse"]
        inputs=batch_datas["inputs"]
        batch_heads=inputs["head"]
        batch_tails = inputs["tail"]
        batch_triples = batch_datas["triples"]
        for head,tail,mask,preds in zip(batch_heads,batch_tails,masks,batch_pred_scores):
            preds=-preds
            # mask[head]=1
            tail_score=preds[tail]
            preds=preds+mask*(1000000)
            preds[tail]=tail_score
            # # 排序
            results = np.argsort(preds, axis=0) #[::-1]

            rank= np.where(results == tail)[0][0]
            tail_ranks.append(rank)
            rank_args_tail.append(results[:100])

        #predict head
        for head, mask, preds in zip(batch_heads, masks_reverse, batch_pred_scores_reverse):
            preds=-preds
            head_score = preds[head]
            preds = preds + mask * (10000)
            preds[head] = head_score
            # 排序
            results = np.argsort(preds, axis=0)
            rank = np.where(results == head)[0][0]
            head_ranks.append(rank)
            rank_args_head.append(results[:100])
        triples.extend(batch_triples)
    relation2ranks_tail=defaultdict(list)
    relation2ranks_head=defaultdict(list)
    relation_nums=[]
    top_candidates={}
    for triple,head_rank,tail_rank,rank_arg_head,rank_arg_tail in zip(triples,head_ranks,tail_ranks,rank_args_head,rank_args_tail):
        h,r,t=triple
        relation_nums.append(r)
        relation2ranks_tail[r].append(tail_rank)
        relation2ranks_head[r].append(head_rank)
        #top k results
        head=id2entity[h]
        relation=id2relation[r]
        tail=id2entity[t]
        top_res_tail=[id2entity[i] for i in rank_arg_tail]
        top_res_tail=" ".join(top_res_tail)
        right=int(t in rank_arg_tail[:10])
        top_candidates["%s %s %s"%(head,relation,tail)]="%s %s"%(right,top_res_tail)

        top_res_head=[id2entity[i] for i in rank_arg_head]
        top_res_head=" ".join(top_res_head)
        right=int(t in rank_arg_head[:10])
        top_candidates["%s %s_reverse %s"%(tail,relation,head)]="%s %s"%(right,top_res_head)

    if is_save_top:
        json.dump(top_candidates, open("%s_top_candidates.json" % data_dir, "w", encoding="utf-8"), indent=3,
                  ensure_ascii=False)


    relation_nums=Counter(relation_nums)
    end_time=time.time()
    print("time: %s"%(end_time-start_time))
    res={}
    for idx in range(len(relations.items())):
        relationRanks_tail=relation2ranks_tail.get(idx)
        relationRanks_head=relation2ranks_head.get(idx)
        num=relation_nums.get(idx,0)
        if relationRanks_tail:
            mr, mrr, hit_1, hit_3, hit_10, hit_30=compute_metrics(relationRanks_tail,is_print=False)
        else:
            mr, mrr, hit_1, hit_3, hit_10=0,0,0,0,0
        res[r]=[mr, mrr, hit_1, hit_3, hit_10]
        print("%s\t%s\t%s"%(idx,num,"\t".join([str(s) for s in res[r]])))

        if relationRanks_head:
            mr, mrr, hit_1, hit_3, hit_10, hit_30=compute_metrics(relationRanks_head,is_print=False)
        else:
            mr, mrr, hit_1, hit_3, hit_10=0,0,0,0,0
        res[r]=[mr, mrr, hit_1, hit_3, hit_10]
        print("%s\t%s\t%s"%(idx,num,"\t".join([str(s) for s in res[r]])))

    #compute metrics
    print("head ranking ....")
    compute_metrics(head_ranks)
    print("tail ranking ....")
    compute_metrics(tail_ranks)
    print("Total ranking ...")
    ranks=head_ranks+tail_ranks
    compute_metrics(ranks)



