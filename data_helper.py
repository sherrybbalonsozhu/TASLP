#coding:utf-8
import random
from codecs import open
from collections import defaultdict
import numpy as np
from helpers.kb_data_reader import read_kb_triples,get_nhop_triples
import pickle
import os
import time
from helpers.data_helper_vocab import VocabHelper

class NeighborDataHelper(object):
    def __init__(self,dataPath,vocabHelper,params,max_nhop_rule_num=2):
        self.dataPath=dataPath
        triples,head2tails,tail2heads=read_kb_triples(dataPath)
        self.kb_triples=triples
        self.head2tails=head2tails
        self.entity2ids=vocabHelper.entity2id
        #2hop neighbors
        self.nhopPath=self.dataPath+"_nhop.pkl"
        self.entity2nhop,self.nhop_triples=self.get_2hop_neighbors()
        # relation2nhop,nhop2relations=count_relation_to_nhop(triples,self.nhop_triples,max_nhop_rule_num=max_nhop_rule_num)
        # self.possible_triples=self.get_possible_triples(self.nhop_triples,nhop2relations)
    def get_possible_triples(self,nhop_triples,nhop2relations):
        possible_triples=set()
        for triple in nhop_triples:
            h=triple[0]
            t=triple[-1]
            r=tuple(triple[1:-1])
            relations=nhop2relations.get(r,[])
            for rel in relations:
                possible_triple=(h,rel,t)
                possible_triples.add(possible_triple)
        return possible_triples



    def get_2hop_neighbors(self):
        if os.path.exists(self.nhopPath):
            entity2nhop=pickle.load(open(self.nhopPath,"rb"))
        else:
            entity2nhop = get_nhop_triples(self.head2tails, self.entity2ids.keys(), n=2)
            pickle.dump(entity2nhop,open(self.nhopPath,"wb"))
        nhop_triples=[]
        for entity,triples in entity2nhop.items():
            nhop_triples.extend(triples)
        print("nhop neighbors: %s"%len(nhop_triples))
        return entity2nhop,nhop_triples



class GraphDataHelper(object):
    def __init__(self,params,dataPath,vocabHelper,valid_triples=[],test_triples=[]):
        self.is_training=params.is_training
        self.init_params(params,dataPath)
        self.vocabHelper=vocabHelper
        self.entity2id=vocabHelper.entity2id
        self.id2entity=vocabHelper.id2entity
        self.all_entities=list(range(len(self.id2entity)))
        self.num_entities=len(self.all_entities)
        self.relation2id=vocabHelper.relation2id
        self.id2relation=vocabHelper.id2relation
        self.all_relations=[self.id2relation[i] for i in range(len(self.id2relation))]
        #build datas
        # self.neighborHelper = NeighborDataHelper(dataPath, vocabHelper, params,max_nhop_rule_num=self.max_nhop_rule_num)
        self.process_datas()
        self.headTailSelector=HeadTailSelector(list(self.triples)+list(valid_triples)+list(test_triples),self.id2entity.keys(),self.id2relation.keys())
        # self.headTailSelector=HeadTailSelector(list(self.triples),self.id2entity.keys(),self.id2relation.keys())
        self.test_triples_set=set([tuple(t) for t in test_triples])
        self.entitiy_num=len(self.id2entity)
        self.init_correct_entities()
    def init_correct_entities(self):
        # 读知识图谱中的三元组
        test_triples, _, _ = read_kb_triples(self.test_dataPath)
        train_triples, _, _ = read_kb_triples(self.train_dataPath)
        valid_triples, _, _ = read_kb_triples(self.valid_dataPath)
        # 构建正确实体集合
        correct_head_entities = defaultdict(set)
        correct_tail_entities = defaultdict(set)
        for triple in train_triples + valid_triples + test_triples:
            h, r, t = triple
            h=self.entity2id.get(h)
            r=self.relation2id.get(r)
            t=self.entity2id.get(t)
            correct_tail_entities[(h, r)].add(t)
            correct_head_entities[(r, t)].add(h)
        self.correct_head_entities=correct_head_entities
        self.correct_tail_entities=correct_tail_entities
    def get_candidate_triples(self,triple,replace_head=True):
        '''替换头、尾实体'''
        test_candidates = []
        test_labels = []
        masks=[]

        test_candidates.append(triple)
        test_labels.append(1)
        masks.append(0)
        h, r, t = triple
        correct_heads = self.correct_head_entities.get((r, t))
        correct_tails = self.correct_tail_entities.get((h, r))
        if replace_head:
            for entity in self.id2entity:
                if entity not in self.vocabHelper.train_unique_entities:
                    continue
                if entity not in correct_heads:
                    dummy_triple = (entity, r, t)
                    test_candidates.append(dummy_triple)
                    test_labels.append(-1)
                    masks.append(0)
        else:
            for entity in self.id2entity:
                if entity not in self.vocabHelper.train_unique_entities:
                    continue
                if entity not in correct_tails:
                    dummy_triple = (h, r, entity)
                    test_candidates.append(dummy_triple)
                    test_labels.append(-1)
                    masks.append(0)
        for i in range(len(test_candidates),self.entitiy_num):
            test_candidates.append(triple)
            test_labels.append(1)
            masks.append(1)
        return test_candidates,test_labels,masks
    def init_params(self,params,dataPath):
        #datasets
        self.data_dir=params.data_dir
        self.dataPath=dataPath
        self.train_dataPath=params.train_dataPath
        self.test_dataPath=params.test_dataPath
        self.valid_dataPath=params.valid_dataPath
        self.entityVocabPath=params.entityVocabPath
        self.relationVocabPath=params.relationVocabPath
        self.entityVectorPath=params.entityVectorPath
        self.relationVectorPath=params.relationVectorPath
        #skip possible
        self.skip_possible=params.skip_possible
        self.max_nhop_rule_num=params.max_nhop_rule_num or 2


        #kb data
        self.max_neighbor_num=params.max_neighbor_num
        self.num_neg=params.num_neg

    def read_datas(self):
        '''从文件中读取数据，返回dict'''
        triples,h2tails,tail2heads=read_kb_triples(self.dataPath)
        return triples
    def process_datas(self):
        triples=self.read_datas()
        triples=self.vocabHelper.convert_triples_to_ids(triples)
        self.triples=np.array(triples)
        self.data_num=len(triples)
        print("data num:",self.data_num)
        #possible triples
        # possible_triples=self.neighborHelper.possible_triples
        # possible_triples=self.vocabHelper.convert_triples_to_ids(possible_triples)
        # self.possible_triples=set([tuple(t) for t in possible_triples])

    def get_batch_datas(self,batch_triples):
        batch_triples=np.array(batch_triples)
        batch_datas={}
        batch_datas["head"]=batch_triples[:,0]
        batch_datas["relation"]=batch_triples[:,1]
        batch_datas["tail"]=batch_triples[:,2]
        # for hid,rid,tid in batch_triples:
        #     batch_datas["head"].append(hid)
        #     batch_datas["relation"].append(rid)
        #     batch_datas["tail"].append(tid)
        return batch_datas


    def train_batch_generator(self,batch_size,shuffle=False,cur_epoch=0):
        data_num=self.data_num
        ids=list(range(data_num))
        if shuffle:
            ids=random.sample(ids,data_num)
        batch_num=(data_num+batch_size-1)//batch_size
        false_negative_num = 0
        for i in range(batch_num):
            batch_start_time=time.time()
            bg=i*batch_size
            end=min((i+1)*batch_size,data_num)
            batch_ids=ids[bg:end]
            bsize=len(batch_ids)
            pos_batch_triples=self.triples[batch_ids]
            neg_batch_triples=[]
            neg_batch_labels=[]
            if i==batch_num-1:
                print("false negtive:",false_negative_num)
            # time_neg_end=time.time()
            # print("neg_sample time: %s"%(time_neg_end-time_neg_start))
            batch_datas={}
            batch_triples=list(pos_batch_triples)
            batch_labels=np.array([1 for d in pos_batch_triples])
            batch_datas["inputs"]=self.get_batch_datas(batch_triples)
            batch_datas["labels"]=batch_labels
            batch_end_time=time.time()
            # print("batch num: %s, data batch time: %s"%(batch_num,batch_end_time-batch_start_time))
            yield batch_datas


    def batch_generator(self, batch_size=1,start_index=0,test_num=100,mini_batch=4):
        if test_num is not None:
            triples=self.triples[start_index:start_index+test_num]
        else:
            triples=self.triples
        data_num = len(triples)
        batch_num = (data_num + batch_size - 1) // batch_size
        for i in range(batch_num):
            bg = i * batch_size
            end = min((i + 1) * batch_size, data_num)
            batch_datas = {}
            batch_triples = triples[bg:end]

            # 替换头和尾实体
            test_triples = []
            test_triples_labels = []
            masks=[]
            batch_triples_new=[]
            for triple in batch_triples:
                h,r,t=triple
                #跳过训练集中没有出度或入度的triple
                # if h not in self.vocabHelper.train_unique_entities or t not in self.vocabHelper.train_unique_entities:
                #     continue
                batch_triples_new.append(triple)
                head_candidates, head_labels,head_mask = self.get_candidate_triples(triple, replace_head=True)
                tail_candidates, tail_labels,tail_mask = self.get_candidate_triples(triple, replace_head=False)
                test_triples.extend(head_candidates + tail_candidates)
                test_triples_labels.extend(head_labels + tail_labels)
                masks.extend(head_mask+tail_mask)
            # print("batch triples:",len(test_triples),len(masks))
            mini_batch_size=(len(test_triples)+mini_batch-1)//mini_batch
            batch_datas = []
            if len(test_triples)==0:
                yield batch_datas
                continue

            for m in range(mini_batch):
                mini_batch_triples=test_triples[mini_batch_size*m:mini_batch_size*(m+1)]
                mini_batch_masks=masks[mini_batch_size*m:mini_batch_size*(m+1)]
                mini_batch_labels=test_triples_labels[mini_batch_size*m:mini_batch_size*(m+1)]
                batch_data ={}
                batch_data["inputs"]=self.get_batch_datas(mini_batch_triples)
                batch_data["masks"]=np.array(mini_batch_masks)
                batch_data["labels"]=mini_batch_labels
                batch_data["triples"]=batch_triples_new
                batch_datas.append(batch_data)
            yield batch_datas

class GraphDataHelperGAT(GraphDataHelper):
    def __init__(self,params,dataPath,vocabHelper,neighborHelper,valid_triples=[],test_triples=[]):
        self.is_training=params.is_training
        self.init_params(params,dataPath)
        self.vocabHelper=vocabHelper
        self.entity2id=vocabHelper.entity2id
        self.id2entity=vocabHelper.id2entity
        self.all_entities=list(range(len(self.id2entity)))
        self.num_entities=len(self.all_entities)
        self.relation2id=vocabHelper.relation2id
        self.id2relation=vocabHelper.id2relation
        self.all_relations=[self.id2relation[i] for i in range(len(self.id2relation))]
        #build datas
        self.neighborHelper=neighborHelper
        self.process_datas()
        self.entitiy_num=len(self.id2entity)
        self.init_correct_entities()

    def init_correct_entities(self):
        # 读知识图谱中的三元组
        test_triples, _, _ = read_kb_triples(self.test_dataPath)
        train_triples, _, _ = read_kb_triples(self.train_dataPath)
        valid_triples, _, _ = read_kb_triples(self.valid_dataPath)
        all_triples = train_triples + valid_triples + test_triples
        # 构建正确实体集合
        # correct_entities = defaultdict(lambda :np.zeros([len(self.id2entity)]))
        correct_tails = defaultdict(set)
        correct_heads = defaultdict(set)
        for triple in all_triples:
            h, r, t = triple
            hid = self.entity2id.get(h)
            rid = self.relation2id.get(r)
            tid = self.entity2id.get(t)
            # correct_entities[(hid, rid)][tid]=1
            # correct_entities[(tid,rid2)][hid]=1
            correct_tails[(hid, rid)].add(tid)
            correct_heads[(tid, rid)].add(hid)
        self.correct_tails=correct_tails
        self.correct_heads=correct_heads

        relation2tails=defaultdict(set)
        relation2heads=defaultdict(set)
        for h, r, t in all_triples:
            hid = self.entity2id.get(h)
            rid = self.relation2id.get(r)
            tid = self.entity2id.get(t)
            relation2tails[rid].add(tid)
            relation2heads[rid].add(hid)
        self.relation2heads=relation2heads
        self.relation2tails=relation2tails


    def init_params(self,params,dataPath):
        super(GraphDataHelperGAT,self).init_params(params,dataPath)
        #datasets
        self.dataPath=dataPath
        self.kb_path=params.kb_path
        self.entityVocabPath=params.entityVocabPath
        self.relationVocabPath=params.relationVocabPath
        self.entityVectorPath=params.entityVectorPath
        self.relationVectorPath=params.relationVectorPath
        #
        self.skip_unique_entities=params.skip_unique_entities
        self.type_constraint=params.type_constraint
        #kb data
        self.max_neighbor_num=params.max_neighbor_num
        self.num_neg=params.num_neg
        self.nhop_sample_num=params.nhop_sample_num or 2000
        # self.num_neg=params.num_neg_gat

    def read_datas(self):
        '''从文件中读取数据，返回dict'''
        triples,h2tails,tail2heads=read_kb_triples(self.dataPath)
        return triples
    def process_datas(self):
        triples=self.read_datas()
        triples=self.vocabHelper.convert_triples_to_ids(triples)
        self.triples=np.array(triples)
        self.data_num=len(triples)
        #kb triples
        kb_triples, _, _ = read_kb_triples(self.kb_path)
        kb_triples=self.vocabHelper.convert_triples_to_ids(kb_triples)
        self.kb_triples=np.array(kb_triples)

        nhop_triples=self.vocabHelper.convert_nhop_triples_to_ids(self.neighborHelper.nhop_triples)
        self.nhop_triples=np.array(nhop_triples)
        self.nhop_triple_num=len(nhop_triples)
        print("data num:",self.data_num)
    def get_batch_nhop_datas(self,nhop_triples):
        batch_datas = defaultdict(list)
        # nhop_triples=np.array(nhop_triples)
        batch_datas["head"]=nhop_triples[:,0]
        batch_datas["tail"]=nhop_triples[:,-1]
        batch_datas["relation"]=nhop_triples[:,1:-1]
        return batch_datas

    def train_batch_generator(self,batch_size,shuffle=False,cur_epoch=0):
        data_num=self.data_num
        ids=list(range(data_num))
        if shuffle:
            ids=random.sample(ids,data_num)
        batch_num=(data_num+batch_size-1)//batch_size
        adjacency_matrix=self.get_batch_datas(self.kb_triples)


        for i in range(batch_num):
            batch_start_time=time.time()
            bg=i*batch_size
            end=min((i+1)*batch_size,data_num)
            batch_ids=ids[bg:end]
            triple_masks=batch_ids
            pos_batch_triples=self.triples[batch_ids]
            pos_batch_labels=[1 for p in pos_batch_triples]
            #nhop
            nhop_ids=np.random.randint(0,self.nhop_triple_num,self.nhop_sample_num)
            # nhop_triples = random.sample(self.nhop_triples, self.nhop_sample_num)
            nhop_triples=self.nhop_triples[nhop_ids]
            nhops = self.get_batch_nhop_datas(nhop_triples)

            batch_triples=list(pos_batch_triples)
            batch_labels=pos_batch_labels
            #label_masks, label_masks_reverse = self.get_labels_mask(batch_triples)
            label_masks,label_masks_reverse=np.zeros(shape=[1,self.entitiy_num]),np.zeros(shape=[1,self.entitiy_num])

            batch_datas={}
            batch_datas["inputs"]=self.get_batch_datas(batch_triples)
            batch_datas["adjacency"]=adjacency_matrix
            batch_datas["nhop"]=nhops
            batch_datas["labels"]=np.array(batch_labels)
            batch_datas["label_masks"]=label_masks
            batch_datas["label_masks_reverse"]=label_masks_reverse
            batch_datas["triple_mask"]=triple_masks
            # batch_datas["triple_mask"]=[]
            # batch_end_time=time.time()
            # print("batch num: %s, batch time: %s"%(batch_num,batch_end_time-batch_start_time))
            yield batch_datas

    def get_labels_mask(self,batch_triples):
        batch_masks=[]
        batch_masks_reverse=[]
        for h,r,t in batch_triples:
            ids=list(self.correct_tails[(h,r)])
            # ids=list(self.query2labels[(h,r)])
            mask=np.zeros(shape=[self.num_entities])
            mask[ids]=1
            mask[t]=0
            batch_masks.append(mask)
            #reverse
            ids = list(self.correct_heads[(t, r)])
            mask=np.zeros(shape=[self.num_entities])
            mask[ids]=1
            mask[h]=0
            batch_masks_reverse.append(mask)

        batch_masks=np.array(batch_masks)
        batch_masks_reverse=np.array(batch_masks_reverse)
        return batch_masks,batch_masks_reverse
    def batch_generator(self, batch_size=1,start_index=0,test_num=None,mini_batch=4):
        if test_num is not None:
            triples=self.triples[start_index:start_index+test_num]
        else:
            triples=self.triples
        adjacency_matrix = self.get_batch_datas(self.kb_triples)
        nhop_triples = self.nhop_triples
        nhops = self.get_batch_nhop_datas(nhop_triples)

        data_num = len(triples)
        batch_num = (data_num + batch_size - 1) // batch_size
        for i in range(batch_num):
            bg = i * batch_size
            end = min((i + 1) * batch_size, data_num)

            batch_triples = triples[bg:end]

            # 替换头和尾实体
            test_triples = []
            test_triples_labels = []
            masks=[]
            masks_reverse=[]
            batch_triples_new=[]
            batch_targets=[]
            for triple in batch_triples:
                h,r,t=triple
                if self.skip_unique_entities:
                    #跳过训练集中没有出度或入度的triple
                    if h not in self.vocabHelper.train_unique_entities or t not in self.vocabHelper.train_unique_entities:
                        continue
                batch_triples_new.append(triple)
                batch_targets.append([t])
                corrects=list(self.correct_tails.get((h,r),[]))
                mask=np.zeros([self.num_entities])
                mask[corrects]=1
                mask[h]=1
                #type constraint
                if self.type_constraint:
                    type_tails=list(self.relation2tails.get(r))
                    type_mask=np.ones([self.num_entities])
                    type_mask[type_tails]=0
                    mask=mask+type_mask
                masks.append(mask)
                #mask reverse
                corrects = list(self.correct_heads.get((t, r), []))
                mask=np.zeros([self.num_entities])
                mask[corrects]=1
                mask[h]=1
                if self.type_constraint:
                    type_heads=list(self.relation2heads.get(r))
                    type_mask=np.ones([self.num_entities])
                    type_mask[type_heads]=0
                    mask=mask+type_mask
                masks_reverse.append(mask)

            batch_triples_new=np.array(batch_triples_new)
            batch_labels=[1 for triple in batch_triples_new]
            # label_masks,label_masks_reverse=self.get_labels_mask(batch_triples_new)
            label_masks,label_masks_reverse=np.zeros(shape=[1,self.entitiy_num]),np.zeros(shape=[1,self.entitiy_num])
            # neighbors=np.array(neighbors)
            if len(batch_triples_new)<=0:
                yield None
                continue
            batch_datas = {}
            batch_datas["inputs"]=self.get_batch_datas(batch_triples_new)
            batch_datas["masks"]=masks
            batch_datas["masks_reverse"]=masks_reverse
            # batch_datas["masks"]=masks+(1-relation_aware_nbrs_tails) #type constraints
            batch_datas["triples"]=batch_triples_new
            batch_datas["labels"]=batch_labels
            batch_datas["label_masks"]=label_masks
            batch_datas["label_masks_reverse"]=label_masks_reverse
            batch_datas["adjacency"]=adjacency_matrix
            batch_datas["nhop"]=nhops

            yield batch_datas



if __name__=="__main__":
    vocabHelper=VocabHelper(params=params)

    dataHelper=GraphDataHelper(dataPath=params.train_dataPath,params=params,vocabHelper=vocabHelper)

    gen=dataHelper.train_batch_generator(batch_size=32)
    for batch_data in gen:
        print(batch_data)
