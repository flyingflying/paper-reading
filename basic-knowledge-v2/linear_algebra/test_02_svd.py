# -*- coding:utf-8 -*-
# Author: lqxu

#%% 
import numpy as np 
from sklearn.decomposition import TruncatedSVD
from scipy.linalg import svd

#%%
n_docs, vocab_size, n_topics = 10000, 500, 15

doc_term_matrix = np.random.randint(
    low=0, high=5, size=(n_docs, vocab_size)
).astype(np.float64)

model = TruncatedSVD(n_components=n_topics, algorithm="arpack")

doc_vecs = model.fit_transform(doc_term_matrix) # (n_docs, n_topics)

topic_vecs = model.components_  # (n_topics, vocab_size)

# %%

U, sigma, VT = svd(doc_term_matrix)

# %%

def is_same_array(a1: np.ndarray, a2: np.ndarray, eps: float = 1e-8) -> bool:
    if a1.shape != a2.shape:
        raise ValueError
    return np.all(np.abs(a1 - a2) < eps) or np.all(np.abs(a1 + a2) < eps)

print(sigma[:n_topics])
print(model.singular_values_)

print(is_same_array(sigma[:n_topics], model.singular_values_))

# %%

print(is_same_array(
    np.abs(topic_vecs), 
    np.abs(VT[:n_topics, :])
))

#%%

print(is_same_array(
    np.abs(U[:, :n_topics] * sigma[:n_topics]),
    np.abs(doc_vecs), 
))

# %%

print(is_same_array(
    U.dot(U.T), np.identity(n_docs)
))

#%%

temp = (U[:, :vocab_size] * sigma).T

print(is_same_array(
    (np.abs(temp @ temp.T) > 1e-8), np.identity(vocab_size)
))

# %%
