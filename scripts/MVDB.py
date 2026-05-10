from sentence_transformers import SentenceTransformer
from PIL import Image
import numpy as np
import faiss
import heapq

class Mode:
    def __init__(self, model, dim, metric, datatype, data, ifile, efile):
        self.model = model
        self.dim = dim
        self.metric = metric
        self.datatype = datatype
        self.data = data
        self.embeddings = None
        self.efile = efile
        self.index = None
        self.ifile = ifile

    def embed_query(self, data):
        if self.datatype=="TEXT":
            query = np.array(self.model.encode(data)).reshape(1, -1)
        elif self.datatype=="IMAGE":
            query = np.array(self.model.encode(Image.open(data))).reshape(1, -1)
        if self.metric == "COSINE":
            faiss.normalize_L2(query)
        return query
    
    def searchsort(self, query, k):
        D, I = self.index.search(query, k=k)
        D = D.reshape(D.shape[1]).astype(np.float64)
        if self.metric == 'L2':
            D = 1 / (1+D)
        I = I.reshape(I.shape[1])
        similarities = D[np.argsort(I)]
        return similarities
    
    def search(self, query, k):
        D, I = self.index.search(query, k=k)
        D = D.reshape(D.shape[1]).astype(np.float64)
        I = I.reshape(I.shape[1]).tolist()
        if self.metric == 'L2':
            D = 1 / (1+D)
        D = D.tolist()
        return D, I

    def create_embeddings(self):
        if self.datatype=="TEXT":
            embeddings = self.model.encode(self.data, show_progress_bar=False)
        elif self.datatype=="IMAGE":
            bs = 1000
            embeddings = np.zeros((len(self.data), self.dim))
            for i in range(0, len(self.data), bs):
                batch = self.data[i:i+bs]
                batch_embeddings = self.model.encode([Image.open(path) for path in batch], show_progress_bar=False)
                embeddings[i:i+bs,:] = batch_embeddings
        self.embeddings = embeddings.astype(np.float32)

    def create_index(self, use_precomputed_embeddings=False):
        if(use_precomputed_embeddings):
            data = np.load(self.efile)
            self.embeddings = data['arr_0']
            data.close()
            print('loaded embeddings')
        else:
            self.create_embeddings()
            print('created embeddings')

        if self.metric == 'COSINE':
            faiss.normalize_L2(self.embeddings)

        nlist = 500
        m = 16

        if self.metric in ['COSINE', 'INNER_PRODUCT']:
            quantizer = faiss.IndexFlatIP(self.dim)
            self.index = faiss.IndexIVFPQ(quantizer, self.dim, nlist, m, 8, faiss.METRIC_INNER_PRODUCT)
        elif self.metric == 'L2':
            quantizer = faiss.IndexFlatL2(self.dim)
            self.index = faiss.IndexIVFPQ(quantizer, self.dim, nlist, m, 8)

        self.index.train(self.embeddings)
        self.index.add(self.embeddings)

    def save(self):
        np.savez_compressed(self.efile, arr_0=self.embeddings)
        faiss.write_index(self.index, self.ifile)

    def load(self):
        data = np.load(self.efile)
        self.embeddings = data['arr_0']
        data.close()
        self.index = faiss.read_index(self.ifile)

class MVDB:
    def __init__(self, modes):
        self.modes = modes

    def create_indexes(self, use_precomputed_embeddings=False):
        for mode in self.modes:
            mode.create_index(use_precomputed_embeddings)

    # Saving/Loading
    def save_indexes(self):
        for mode in self.modes:
            mode.save()

    def load_indexes(self):
        for mode in self.modes:
            mode.load()

    # Querying
    def get_query(self, data):
        query = []
        for i, mode in enumerate(self.modes):
            query.append(mode.embed_query(data[i]))
        return query

    def naive_topk_original(self, query, k,nprobe):
        aggregation_fn = MVDB.average
        similarities = []
        for i, mode in enumerate(self.modes):
            mode.index.nprobe = mode.index.nlist
            similarities.append(mode.searchsort(query[i], k=mode.index.ntotal))

        aggregated_similarities = aggregation_fn(similarities)

        # Get top k
        knn = np.argpartition(aggregated_similarities,-k)[-k:]
        indexes = knn[np.argsort(aggregated_similarities[knn])][::-1]

        return indexes
    
    def average(similarities):
        aggregated = similarities[0]
        for i in range(1, len(similarities)):
            aggregated += similarities[i]
        aggregated = aggregated / len(similarities)
        return aggregated

    def get_score(self, mode, query, id):
        embedding = np.array(self.modes[mode].embeddings[id])
        q = np.array(query.flatten())

        if self.modes[mode].metric in ['COSINE', 'INNER_PRODUCT']:
            result = np.dot(q, embedding)
        elif self.modes[mode].metric == 'L2':
            result = np.linalg.norm(q - embedding) ** 2
            result = 1 / (1 + result)

        return result
    
    def topkinsert(heap, k, item):
        """
        Maintain a fixed-size min-heap containing the top-k scoring items.
        """
        if len(heap) < k:
            heapq.heappush(heap, item)
        elif item[0] > heap[0][0]:
            heapq.heappop(heap)
            heapq.heappush(heap, item)
    
    def naive_topk_ann(self, query, k,nprobe,candidate_k=200):
        """NN search in each mode to get candidate pool , rescore candidates using 
        stored embeddings aggregate and return top-k"""
        for mode in self.modes:
            mode.index.nprobe = nprobe
            
        candidate_ids = set()
        for i , mode in enumerate(self.modes):
            kk = min(candidate_k, mode.index.ntotal)
            _,ids = mode.search(query[i],k=kk)
            candidate_ids.update(ids)
            
        candidate_ids = np.array(list(candidate_ids),dtype= np.int64)
        
        agg = np.zeros(len(candidate_ids),dtype=np.float64)
        
        for i , mode in enumerate(self.modes):
            q = query[i].reshape(-1).astype(np.float32)
            E = mode.embeddings[candidate_ids].astype(np.float32)
            
            if mode.metric in ["COSINE", "INNER_PRODUCT"]:
                s = E @ q
            elif mode.metric == "L2":
                diff = E - q
                d2 = np.einsum("ij,ij->i",diff , diff)
                s = 1.0/(1.0 + d2)
            else:
                raise ValueError(f"Unsupported metric: {mode.metric}")
                
            agg += s.astype(np.float64)
            
        agg /= len(self.modes)
        
        topk_local = np.argpartition(agg, -k)[-k:]
        topk_local = topk_local[np.argsort(agg[topk_local])][::-1]
        return candidate_ids[topk_local]
            
    
    def exact_topk(self, query, k):
        """
        baseline: brute-force over all embeddings (no FAISS).
        query: list of per-mode query vectors, each shape (1, dim)
        Returns: np.ndarray of top-k ids
        """
        sims = []
        for i, mode in enumerate(self.modes):
            q = query[i].reshape(-1).astype(np.float32)          # (dim,)
            E = mode.embeddings.astype(np.float32)              # (N, dim)

            if mode.metric in ["COSINE", "INNER_PRODUCT"]:
                # If cosine, embeddings should be normalized when created; if not, normalize here.
                # q dot E^T
                s = E @ q
            elif mode.metric == "L2":
                # squared L2 distances
                diff = E - q
                d2 = np.einsum("ij,ij->i", diff, diff)
                s = 1.0 / (1.0 + d2)  
            else:
                raise ValueError(f"Unsupported metric: {mode.metric}")

            sims.append(s.astype(np.float64))

        agg = sims[0]
        for j in range(1, len(sims)):
            agg = agg + sims[j]
        agg = agg / len(sims)

        topk = np.argpartition(agg, -k)[-k:]
        topk = topk[np.argsort(agg[topk])][::-1]
        return topk

        
    def qc_topk(self, query, k, p, maxN=10000, epsilon = 1.0,nprobe=20):
        N = 20
        for i, mode in enumerate(self.modes):
            mode.index.nprobe = nprobe
        s1, i1 = self.modes[0].search(query[0], k=N)
        s2, i2 = self.modes[1].search(query[1], k=N)

        # Initialize
        z1 = p
        z2 = p
        visited = set()
        topk = []
        for i in range(p+1):
            if i1[i] not in visited:
                F = 0.5*s1[i] + 0.5*self.get_score(1, query[1], i1[i])
                MVDB.topkinsert(topk, k, (F, i1[i]))
                visited.add(i1[i])
            if i2[i] not in visited:
                F = 0.5*s2[i] + 0.5*self.get_score(0, query[0], i2[i])
                MVDB.topkinsert(topk, k, (F, i2[i]))
                visited.add(i2[i])

        minF = 0.5*s1[z1] + 0.5*s2[z2]
        kth = topk[0][0]

        
        while(kth < minF):
            d1 = 0.5*(s1[z1-p] - s1[z1])
            d2 = 0.5*(s2[z2-p] - s2[z2])
            # Choose Stream
            if d1 > d2: 
                # use stream 1
                z1 += 1
                if i1[z1] not in visited:
                    F = 0.5*s1[z1] + 0.5*self.get_score(1, query[1], i1[z1])
                    MVDB.topkinsert(topk, k, (F, i1[z1]))
                    visited.add(i1[z1])
            else:
                # use stream 2
                z2 += 1
                if i2[z2] not in visited:
                    F = 0.5*s2[z2] + 0.5*self.get_score(0, query[0], i2[z2])
                    MVDB.topkinsert(topk, k, (F, i2[z2]))
                    visited.add(i2[z2])
                
            minF = 0.5*s1[z1] + 0.5*s2[z2]
            kth = topk[0][0]
            
            if (kth >= minF):
                
                return [item[1] for item in topk]

            if(max(z1+1, z2+1) >= N):
                
                #add heuristic to decide to increase
                num_items = sum(1 for item in topk if item[0] > minF)
                
                if num_items >= epsilon*k:
                    break
                N = 3*N
                if N > maxN:
                    break
                s1, i1 = self.modes[0].search(query[0], k=N)
                s2, i2 = self.modes[1].search(query[1], k=N)
        

        return [item[1] for item in topk]