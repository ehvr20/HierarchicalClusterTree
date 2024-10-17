import numpy as np

def calculate_centroid(members, embedding):
    return np.mean(embedding[members, :], axis=0)