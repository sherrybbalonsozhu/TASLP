#coding:utf-8
import numpy as np
from collections import defaultdict


def padding_one(data,maxlen,pad):
    data=list(data)+[pad for i in range(maxlen-len(data))]
    return data[:maxlen]

def read_word2id(path):
    '''实体到id的映射
    文件格式：每行两列，实体    id
    '''
    with open(path,encoding="utf-8") as fin:
        word2id={}
        id2word={}
        for line in fin:
            line=line.strip().split()
            assert len(line)==2
            e,id=line
            id=int(id)
            word2id[e]=id
            id2word[id]=e
        return word2id,id2word

def read_word2vec(path):
    vectors=[]
    with open(path,encoding="utf-8") as fin:
        for line in fin:
            line=line.strip()
            vec=[float(v) for v in line.split()]
            size=len(vec)
            # assert size==100
            vectors.append(vec)
    vectors=np.array(vectors)
    print("%s vector size:%s"%(path,vectors.shape))

    return vectors

def read_kb_triples(kb_path):
    '''读知识图谱中的三元组'''
    t2heads=defaultdict(dict)
    h2tails=defaultdict(dict)
    triples=[]
    with open(kb_path,encoding="utf-8") as fin:
        for line in fin:
            line=line.strip().split()
            assert len(line)==3,line
            h,r,t=line
            triples.append([h,r,t])
            t2heads[t][h]=r
            h2tails[h][t]=r
    return triples,h2tails,t2heads
def get_nhop_triples(head2tails,entities,n=2,terminate_neighbor_num=200):
    '''
    获取每个实体nhop的邻居节点
    :param head2tails:
    :param tail2heads:
    :param entities:
    :param n:
    :param in_direction: 入度邻居
    :param out_direction: 出度邻居
    :return:
    '''
    entity2neighbors={}
    # for entity in tqdm.tqdm(entities):
    for entity in entities:
        # 获取n-hop邻居节点
        queue_tail=[[(entity,r,t)] for t,r in head2tails[entity].items()]
        neighbors=[]
        neighbors_nodes=set()
        #出度的邻居节点
        while len(queue_tail)>0:
            triples=queue_tail.pop(0)
            if len(triples)>=n:
                if triples[-1] is None:
                    continue
                nhop_triple=[triples[0][0]]+[triple[1] for triple in triples]+[triples[-1][2]]
                nhop_triple=tuple(nhop_triple)
                if nhop_triple not in neighbors_nodes:
                    neighbors.append(nhop_triple)
                    neighbors_nodes.add(nhop_triple)
            else:
                last_triple=triples[-1]
                head,r,tail=last_triple
                nbs=head2tails[tail].items()
                if len(nbs)==0 or len(nbs)>terminate_neighbor_num:
                    nbs=[None]
                for nb in nbs:
                    if nb is None:
                        triples_new=triples+[None]
                    else:
                        triples_new=triples+[(tail,nb[1],nb[0])]
                    queue_tail.append(triples_new)
        entity2neighbors[entity]=neighbors

    return entity2neighbors
