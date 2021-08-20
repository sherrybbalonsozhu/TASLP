#coding:utf-8
from collections import defaultdict
import numpy as np
from helpers.kb_data_reader import read_word2id,read_word2vec,read_kb_triples


class VocabHelper(object):
    '''Read Vocab datas'''
    def __init__(self,params):
        self.init_params(params)
        self.trainPath=params.train_dataPath
        self.read_datas()

    def init_params(self,params):
        #vocab path
        self.entityVocabPath=params.entityVocabPath
        self.relationVocabPath=params.relationVocabPath

        self.entity_vector_path=params.entity_vector_path
        self.relation_vector_path=params.relation_vector_path
    def get_entity2relations(self,triples,e2id,r2id):
        entity2relations_in = defaultdict(set)
        entity2relations_out = defaultdict(set)
        triple_num = 0
        for h,r,t in triples:
            h=e2id[h]
            t=e2id[t]
            r=r2id[r]
            triple_num += 1
            entity2relations_in[t].add(r)
            entity2relations_out[h].add(r)
        #matrix
        e2r_in=np.zeros(shape=[len(e2id),len(r2id)])
        e2r_out=np.zeros(shape=[len(e2id),len(r2id)])
        for e,rs in entity2relations_in.items():
            for r in rs:
                e2r_in[e][r]=1/len(rs)
        for e,rs in entity2relations_out.items():
            for r in rs:
                e2r_out[e][r]=1/len(rs)

        return e2r_in, e2r_out

    def get_relation2relations(self,triples,e2id,r2id):
        entity2relations_in = defaultdict(set)
        entity2relations_out = defaultdict(set)
        triple_num = 0
        for h,r,t in triples:
            h=e2id[h]
            t=e2id[t]
            r=r2id[r]
            triple_num += 1
            entity2relations_in[t].add(r)
            entity2relations_out[h].add(r)
        #matrix
        # r2r_out=np.ones(shape=[len(r2id),len(r2id)])
        # r2r_in=np.ones(shape=[len(r2id),len(r2id)])
        r2r_out=np.zeros(shape=[len(r2id),len(r2id)])
        r2r_in=np.zeros(shape=[len(r2id),len(r2id)])
        for e,rs in entity2relations_out.items():
            for r1 in rs:
                for r2 in rs:
                    if r1==r2:
                        continue
                    r2r_out[r1][r2]+=1
        for e,rs in entity2relations_in.items():
            for r1 in rs:
                for r2 in rs:
                    if r1==r2:
                        continue
                    r2r_in[r1][r2]+=1
        r2r_in = r2r_in / (np.sum(r2r_in, axis=1, keepdims=True)+1e-8)
        r2r_out = r2r_out / (np.sum(r2r_out, axis=1, keepdims=True)+1e-8)
        #next relation
        next_relation_out=np.zeros(shape=[len(r2id),len(r2id)])+1e-8
        next_relation_in=np.zeros(shape=[len(r2id),len(r2id)])+1e-8
        for e,rs_in in entity2relations_in.items():
            rs_out=entity2relations_out.get(e,[])
            for r1 in rs_in:
                for r2 in rs_out:
                    next_relation_out[r1][r2]+=1
                    next_relation_in[r2][r1]+=1
        next_relation_out=next_relation_out/(np.sum(next_relation_out,axis=1,keepdims=True)+1e-8)
        next_relation_in=next_relation_in/(np.sum(next_relation_in,axis=1,keepdims=True)+1e-8)
        return r2r_out,r2r_in,next_relation_out,next_relation_in

    def get_relation_types(self,triples):
        head2tails=defaultdict(lambda :defaultdict(list))
        tail2heads=defaultdict(lambda :defaultdict(list))

        for h,r,t in triples:
            head2tails[r][h].append(t)
            tail2heads[r][t].append(h)
        relation2types={}
        for r in head2tails.keys():
            h2t=head2tails[r]
            t2h=tail2heads[r]
            ts_num=0
            h_num=0
            for h,ts in h2t.items():
                ts_num+=len(ts)
                h_num+=1
            avg_tail=ts_num/h_num
            hs_num=0
            t_num=0
            for t,hs in t2h.items():
                hs_num+=len(hs)
                t_num+=1
            avg_head=hs_num/t_num
            if avg_tail<=1.5 and avg_head<=1.5:
                relation2types[r]="1-1"
            elif avg_tail<=1.5 and avg_head>1.5:
                relation2types[r]="M-1"
            elif avg_tail>1.5 and avg_head>1.5:
                relation2types[r]="M-M"
            elif avg_tail>1.5 and avg_head<=1.5:
                relation2types[r]="1-M"
        
        return relation2types
    def get2hop_triples(self,triples,head2tails):
        nhop_triples=set()
        for h,r,t in triples:
            rel_tails=head2tails.get(t,{})
            for t,r2 in rel_tails.items():
                nhop=(h,r,r2,t)
                nhop_triples.add(nhop)
        return nhop_triples

    def get_relation_to_nhop(self,triples, nhop_triples,r2ids):
        entity2relations = defaultdict(set)
        for h, r, t in triples:
            entity = "%s_%s" % (h, t)
            entity2relations[entity].add(r)
        entity2nhop = defaultdict(set)
        for h, r1, r2, t in nhop_triples:
            entity = "%s_%s" % (h, t)
            entity2nhop[entity].add((r1, r2))
        relation2relations = np.zeros([len(r2ids),len(r2ids)])+1e-8
        for e, rs in entity2relations.items():
            nhops = entity2nhop.get(e, [])
            for r in rs:
                r=r2ids[r]
                for nhop in nhops:
                    # for r2 in nhop:
                    for r2 in nhop[:1]:
                        r2=r2ids[r2]
                        relation2relations[r][r2]+=1
        relation2relations=relation2relations/(np.max(relation2relations,axis=1,keepdims=True)+1e-8)
        return relation2relations
    def read_datas(self):
        self.entity2id,self.id2entity=read_word2id(self.entityVocabPath)
        self.relation2id,self.id2relation=read_word2id(self.relationVocabPath)
        self.entity_embeddings = np.array(read_word2vec(self.entity_vector_path)).astype("float32")
        self.relation_embeddings=np.array(read_word2vec(self.relation_vector_path)).astype("float32")
        self.train_triples, self.train_head2tails, self.train_tail2heads = read_kb_triples(self.trainPath)
        #relation2types
        self.relation2types=self.get_relation_types(self.train_triples)
        self.r2r_out,self.r2r_in,self.next_relation_out,self.next_relation_in=self.get_relation2relations(self.train_triples,self.entity2id,self.relation2id)

        nhop_triples=self.get2hop_triples(self.train_triples,self.train_head2tails)
        self.r2r_out=self.r2r_in=self.get_relation_to_nhop(self.train_triples,nhop_triples,self.relation2id)


        self.train_head2tails={self.entity2id[k]:v for k,v in self.train_head2tails.items()}
        self.train_tail2heads={self.entity2id[k]:v for k,v in self.train_tail2heads.items()}
        self.train_unique_entities=set(list(self.train_head2tails.keys())+list(self.train_tail2heads.keys()))
        print("entity embedding shape:",self.entity_embeddings.shape)
        print("relation embedding shape:",self.relation_embeddings.shape)
    def convert_entities_to_ids(self,entities):
        ids=[]
        for entity in entities:
            i=self.entity2id.get(entity)
            ids.append(i)
        return ids
    def convert_relations_to_ids(self,relations):
        ids=[]
        for relation in relations:
            i=self.relation2id.get(relation)
            ids.append(i)
        return ids
    def convert_triples_to_ids(self,triples):
        triple_ids=[]
        for h,r,t in triples:
            hid=self.entity2id.get(h)
            tid=self.entity2id.get(t)
            rid=self.relation2id.get(r)
            triple_ids.append([hid,rid,tid])
        return triple_ids
    def convert_ids_to_triple(self,triple_ids):
        hid,rid,tid=triple_ids
        h=self.id2entity[hid]
        r=self.id2relation[rid]
        t=self.id2entity[tid]
        return (h,r,t)
    def convert_nhop_triples_to_ids(self,triples):
        triple_ids=[]
        for triple in triples:
            hid=self.entity2id.get(triple[0])
            tid=self.entity2id.get(triple[-1])
            rids=[self.relation2id.get(r) for r in triple[1:-1]]
            triple_ids.append([hid]+rids+[tid])
        return triple_ids
