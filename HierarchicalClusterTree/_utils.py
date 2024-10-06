import numpy as np

def calculate_centroid(members, embedding):
    return np.mean(embedding[list(members), :], axis=0)