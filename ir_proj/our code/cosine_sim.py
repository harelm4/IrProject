import numpy as np


def cosine_similarity(D, Q):
    """
    Calculate the cosine similarity for each candidate document in D and a given query (e.g., Q).
    Generate a dictionary of cosine similarity scores
    key: doc_id
    value: cosine similarity score

    Parameters:
    -----------
    D: DataFrame of tfidf scores.

    Q: vectorized query with tfidf scores

    Returns:
    -----------
    dictionary of cosine similarity score as follows:
                                                                key: document id (e.g., doc_id)
                                                                value: cosine similarty score.
    """

    # YOUR CODE HERE
    d = {}
    mat = D.to_numpy()

    for i in range(0, len(mat)):
        x1 = mat[i] / np.linalg.norm(mat[i])
        x2 = Q / np.linalg.norm(Q)
        val = np.matmul(x1, x2.T)
        d[i] = val

    return d
