import numpy as np
import os

# MIT-States
def load_mitstates(root_dir, N):
    dataset_dir = root_dir + "/mitstates/release_dataset/images"
    images = []
    captions = []
    query_dict = {'adjectives': {}, 'nouns': {}, 'relevant_ids': {}}
    idx = 0
    for root, dirs, files in os.walk(dataset_dir):
        dirs.sort()
        if idx == N:
            break
        # Caption
        caption = root.split('/')[-1]
        
        if 'images' in caption:
            continue

        if 'adj ' in caption:
            # use for queries
            noun = caption.split(' ')[-1]
            objects = []
            for image in files:
                image_path = os.path.join(root, image)
                objects.append(image_path)
            query_dict['nouns'][noun] = objects
        else:
            adjective = caption.split(' ')[0]
            noun = caption.split(' ')[-1]
            if adjective not in query_dict['adjectives'].keys():
                query_dict['adjectives'][adjective] = []
            else:
                query_dict['adjectives'][adjective].append(noun)
            # Images
            category_ids = []
            for i, image in enumerate(files):
                image_path = os.path.join(root, image)
                images.append(image_path)
                captions.append(adjective)
                category_ids.append(idx)
                idx += 1
                if idx == N:
                    break

            query_dict['relevant_ids'][caption] = category_ids
    print(idx)
    return images, captions, query_dict

def create_mitstates_query(query_dict, noun, adjective, idx=-1):
    if idx == -1:
        idx = np.random.randint(len(query_dict['nouns'][noun]))
    image = query_dict['nouns'][noun][idx]
    text = adjective
    return image, text

