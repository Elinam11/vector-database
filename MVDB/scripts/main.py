import time
from MVDB import MVDB, Mode
from load_dataset import load_mitstates, create_mitstates_query
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer

def recall(ground_truth, results, k):
    r = len(set(ground_truth) & set(results)) / k
    return r

if __name__ == '__main__':

    # MITSTATES
    N = 19500
    a = time.time()
    dataset_dir = "/home/elina/datasets" #Add path to directory with dataset
    images, captions, query_dict = load_mitstates(dataset_dir, N)
    b = time.time()
    print('data loaded:', b-a)    

    # Create Database
    path = "/home/elina/projects/data_indices" #Add path to folder with save files
    image_mode = Mode(model=SentenceTransformer("clip-ViT-B-32", device='cpu'), 
                      dim=512,
                      metric='L2',
                      datatype='IMAGE',
                      data=images[:N],
                      ifile=path+"image_clipl2.index",
                      efile=path+"image_clip.npz",)
    text_mode = Mode(model=SentenceTransformer("multi-qa-mpnet-base-dot-v1", device='cpu'), 
                      dim=768,
                      metric='L2',
                      datatype='TEXT',
                      data=captions[:N],
                      ifile=path+"text_mpl2.index",
                      efile=path+"text_mp.npz",)

    db = MVDB([text_mode, image_mode])

    # Creating from scratch
    db.create_indexes(use_precomputed_embeddings=False)
    db.save_indexes()

    # Loading from file
    # db.load_indexes()

    c = time.time()
    print('database created:', c-b)

    k = 1
    
    agg_recalls = []
    agg_latencies = []
    qc_recalls = []
    qc_latencies = []

    num_tests = 1000
    test = 0
    for adj in query_dict['adjectives'].keys():
        for noun in query_dict['adjectives'][adj]:
            if noun not in query_dict['nouns'].keys():
                continue

            ground_truth = query_dict['relevant_ids'][adj+' '+noun]
            if test >= num_tests:
                break
            image, text = create_mitstates_query(query_dict, noun, adj, idx=0)
            query = db.get_query([text, image])

            d = time.time()
            indexes = db.naive_topk(query, k)
            e = time.time()
            t1 = e-d
            if t1 > 0 and t1 < 300:
                agg_latencies.append(t1)
            r1 = recall(ground_truth, indexes, k)

            d2 = time.time()
            indexes2 = db.qc_topk(query, k, k, epsilon=0.6)
            e2 = time.time()
            t2 = e2-d2
            if t2 > 0 and t2 < 300:
                qc_latencies.append(t2)
            r2 = recall(ground_truth, indexes2, k)

            agg_recalls.append(r1)
            qc_recalls.append(r2)

            if test%100 == 0:
                print(test)
            test += 1

    print(test)
    print(k)
    print('Average Recall (Naive):', np.average(agg_recalls))
    print('Average Recall (Quick-Combine):', np.average(qc_recalls))
    print('Average Latency (Naive):', np.average(agg_latencies))
    print('Average Latency (Quick-Combine):', np.average(qc_latencies))