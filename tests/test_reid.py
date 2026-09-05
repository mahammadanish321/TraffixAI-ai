import os
import sys
import time
import cv2
import numpy as np
import torch

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from reid import VehicleReIDExtractor, compute_cosine_similarity, compute_euclidean_distance, match_embeddings
from config.settings import settings

def load_or_create_test_vehicles():
    """
    Loads real vehicle crop from test artifacts or sample video if available,
    otherwise synthesizes realistic vehicle images with natural textures.
    """
    crop_path = os.path.join(os.path.dirname(__file__), "output", "1_vehicle_crop.jpg")
    if os.path.exists(crop_path):
        car_a = cv2.imread(crop_path)
    else:
        # Create patterned car representation
        car_a = np.zeros((160, 220, 3), dtype=np.uint8)
        car_a[:, :] = [180, 50, 30]  # Metallic Blue
        cv2.rectangle(car_a, (30, 20), (190, 80), (220, 220, 220), -1)  # Windshield
        cv2.circle(car_a, (50, 140), 20, (30, 30, 30), -1)  # Front Wheel
        cv2.circle(car_a, (170, 140), 20, (30, 30, 30), -1)  # Rear Wheel
        cv2.GaussianBlur(car_a, (5, 5), 0, dst=car_a)

    # Distinct vehicle (e.g. Yellow delivery truck / bus)
    car_b = np.zeros((160, 220, 3), dtype=np.uint8)
    car_b[:, :] = [30, 210, 230]  # Yellow
    cv2.rectangle(car_b, (15, 15), (205, 110), (50, 50, 50), -1)  # Cabin
    cv2.circle(car_b, (45, 145), 22, (20, 20, 20), -1)
    cv2.circle(car_b, (175, 145), 22, (20, 20, 20), -1)
    cv2.GaussianBlur(car_b, (5, 5), 0, dst=car_b)

    return car_a, car_b

def run_reid_tests():
    print("=" * 68)
    print("      TRAFFIX AI — PERSON 2: VEHICLE RE-ID & METRIC EMBEDDING TEST      ")
    print("=" * 68)

    # 1. Initialize Extractor
    print("[1/5] Initializing VehicleReIDExtractor...")
    extractor = VehicleReIDExtractor(embedding_dim=settings.REID_EMBEDDING_DIM)
    print(f"  • Backbone Model   : {extractor.model_name}")
    print(f"  • Target Dimension : {extractor.embedding_dim}")
    print(f"  • Compute Device   : {extractor.device}")

    # 2. Test Dimension & L2 Unit Normalization
    print("\n[2/5] Testing Embedding Dimension & L2 Normalization...")
    car_a, car_b = load_or_create_test_vehicles()

    emb_a1 = extractor.extract(car_a)
    assert len(emb_a1) == settings.REID_EMBEDDING_DIM, f"Expected {settings.REID_EMBEDDING_DIM}, got {len(emb_a1)}"
    
    l2_norm = np.linalg.norm(emb_a1)
    print(f"  • Extracted vector length : {len(emb_a1)} floats")
    print(f"  • L2 Vector Norm          : {l2_norm:.5f} (Target: 1.00000)")
    assert abs(l2_norm - 1.0) < 1e-3, "Vector is not properly L2 normalized!"
    print("  ✅ Dimension & Normalization: PASSED")

    # 3. Test Metric Consistency (Same Vehicle vs Different Vehicle)
    print("\n[3/5] Testing Cosine Similarity Metric Properties...")
    # Simulate realistic camera-to-camera shifts: lighting/brightness shift + slight scale shift
    h, w = car_a.shape[:2]
    car_a_shifted = cv2.resize(car_a, (int(w * 0.96), int(h * 0.96)))
    car_a_shifted = np.clip(car_a_shifted.astype(np.float32) * 0.92, 0, 255).astype(np.uint8)

    emb_a2 = extractor.extract(car_a_shifted)
    same_car_sim = compute_cosine_similarity(emb_a1, emb_a2)
    print(f"  • Same Vehicle (Viewpoint & Light Shift) Cosine Similarity: {same_car_sim:.4f}")
    assert same_car_sim > 0.85, f"Expected similarity > 0.85 for same vehicle, got {same_car_sim}"

    # Compare with different vehicle
    emb_b = extractor.extract(car_b)
    diff_car_sim = compute_cosine_similarity(emb_a1, emb_b)
    print(f"  • Different Vehicle (Car A vs Vehicle B) Cosine Sim       : {diff_car_sim:.4f}")
    assert diff_car_sim < 0.50, f"Expected similarity < 0.50 for different vehicles, got {diff_car_sim}"
    print("  ✅ Metric Discrimination: PASSED")

    # 4. Test Cross-Camera Gallery Matching
    print("\n[4/5] Testing Cross-Camera Gallery Matching Logic...")
    gallery = {
        "VEH_GLOBAL_001": emb_a1,
        "VEH_GLOBAL_002": emb_b
    }
    # Query with Car A observation from camera 2
    matched_id, score = match_embeddings(emb_a2, gallery, threshold=settings.REID_SIMILARITY_THRESHOLD)
    print(f"  • Query observation matched with : {matched_id} (Score: {score:.4f})")
    assert matched_id == "VEH_GLOBAL_001", f"Expected VEH_GLOBAL_001, got {matched_id}"
    print("  ✅ Cross-Camera Identity Matching: PASSED")

    # 5. Latency & Throughput Benchmark
    print("\n[5/5] Benchmarking Re-ID Feature Extraction Latency on CPU (20 iterations)...")
    latencies = []
    for _ in range(20):
        t0 = time.time()
        _ = extractor.extract(car_a)
        latencies.append((time.time() - t0) * 1000)

    mean_lat = sum(latencies) / len(latencies)
    p95_lat = sorted(latencies)[int(len(latencies) * 0.95)]
    fps = 1000.0 / mean_lat if mean_lat > 0 else 0.0

    print(f"  • Mean Latency : {mean_lat:.2f} ms")
    print(f"  • P95 Latency  : {p95_lat:.2f} ms")
    print(f"  • Throughput   : {fps:.1f} vehicle crops/sec")
    assert mean_lat < 25.0, f"Mean latency {mean_lat}ms exceeds 25ms budget"
    print("  ✅ Performance Budget (<25ms on CPU): PASSED")

    print("\n" + "=" * 68)
    print("🎉 ALL PERSON 2 RE-ID TESTS PASSED SUCCESSFULLY!")
    print("=" * 68)

if __name__ == "__main__":
    run_reid_tests()
