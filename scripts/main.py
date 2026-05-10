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

    # Uncomment to creaste embeddings from scratch
    #db.create_indexes(use_precomputed_embeddings=False)
    #db.save_indexes()

    # Loading from file
    db.load_indexes()

    # Experiment settings
    k = 10
    eps = 1
    probe_list = [1, 2, 5, 10, 20, 25,28,30,50, 100]
    num_tests = 100

    results = {p: {"qc_r": [], "naive_r": [], "qc_t": [], "naive_t": []} for p in probe_list}

    test = 0
    for adj in query_dict['adjectives'].keys():
        for noun in query_dict['adjectives'][adj]:
            if noun not in query_dict['nouns']:
                continue
            if test >= num_tests:
                break

            image, text = create_mitstates_query(query_dict, noun, adj, idx=0)
            query = db.get_query([text, image])

            # True ground truth (exact brute-force aggregated top-k)
            gt = db.exact_topk(query, k)

            for nprobe in probe_list:
                # Naive ANN-controlled baseline
                t0 = time.time()
                naive_ids = db.naive_topk_ann(query, k, nprobe=nprobe)
                t1 = time.time()
                naive_time = t1 - t0
                results[nprobe]["naive_t"].append(naive_time)
                results[nprobe]["naive_r"].append(recall(gt, naive_ids, k))

                # Quick-Combine
                t2 = time.time()
                qc_ids = db.qc_topk(query, k, p=3, epsilon=eps, nprobe=nprobe)
                t3 = time.time()
                qc_time = t3 - t2
                results[nprobe]["qc_t"].append(qc_time)
                results[nprobe]["qc_r"].append(recall(gt, qc_ids, k))

            if test % 25 == 0:
                print("query", test)
            test += 1

    print("done:", test, "queries")
    for nprobe in probe_list:
        print("\n== nprobe", nprobe, "==")
        print("Naive recall:", float(np.mean(results[nprobe]["naive_r"])))
        print("QC recall:", float(np.mean(results[nprobe]["qc_r"])))
        print("Naive latency:", float(np.mean(results[nprobe]["naive_t"])))
        print("QC latency:", float(np.mean(results[nprobe]["qc_t"])))
    