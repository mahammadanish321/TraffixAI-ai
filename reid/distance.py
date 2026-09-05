import numpy as np
from typing import List, Dict, Tuple, Optional, Union

def compute_cosine_similarity(
    vec_a: Union[List[float], np.ndarray],
    vec_b: Union[List[float], np.ndarray]
) -> float:
    """
    Computes cosine similarity between two feature vectors:
    similarity = (A . B) / (||A|| * ||B||)
    
    Since our Re-ID embeddings are pre-normalized to unit L2 length (||v|| = 1.0),
    the cosine similarity simplifies directly to the dot product.
    Returns float in range [-1.0, 1.0].
    """
    a = np.asarray(vec_a, dtype=np.float32)
    b = np.asarray(vec_b, dtype=np.float32)

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    sim = float(np.dot(a, b) / (norm_a * norm_b))
    return max(-1.0, min(1.0, sim))

def compute_euclidean_distance(
    vec_a: Union[List[float], np.ndarray],
    vec_b: Union[List[float], np.ndarray]
) -> float:
    """
    Computes standard Euclidean (L2) distance between two feature vectors.
    """
    a = np.asarray(vec_a, dtype=np.float32)
    b = np.asarray(vec_b, dtype=np.float32)
    return float(np.linalg.norm(a - b))

def match_embeddings(
    query_vec: Union[List[float], np.ndarray],
    gallery: Dict[str, Union[List[float], np.ndarray]],
    threshold: float = 0.75
) -> Tuple[Optional[str], float]:
    """
    Matches a query vehicle embedding against a gallery of registered vehicle embeddings.
    
    Returns:
        (best_vehicle_id, best_similarity_score)
        If best_similarity_score < threshold, returns (None, best_similarity_score).
    """
    if not gallery:
        return (None, 0.0)

    best_id: Optional[str] = None
    best_score: float = -1.0

    for vehicle_id, candidate_vec in gallery.items():
        score = compute_cosine_similarity(query_vec, candidate_vec)
        if score > best_score:
            best_score = score
            best_id = vehicle_id

    if best_score >= threshold:
        return (best_id, best_score)

    return (None, best_score)
