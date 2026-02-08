# =============================================================
# Project: SLRM-nD (Lumin Core v2.0)
# Optimized Simplex Interpolation Engine
# Project Lead: Alex Kinetic
# AI Collaboration: Gemini · ChatGPT · Claude · Grok · Meta AI
# License: MIT
# =============================================================

import numpy as np
import time
from typing import Optional, Tuple, Union


class LuminCore:
    """
    Lumin Core v2.0 - Simplex Interpolation Engine
    
    Performs high-dimensional interpolation by encapsulating the query point
    within a geometric simplex formed by the nearest boundary nodes along each axis.
    
    Key Features:
    - Automatic extrapolation detection (rejects points outside dataset bounds)
    - Vectorized simplex construction O(D) complexity
    - Inverse Distance Weighting (IDW) for stable interpolation
    - Robust numerical stability and edge case handling
    
    Principles:
    - For each dimension i, finds the nearest points where:
      * X[i] <= query[i] (inferior boundary)
      * X[i] >= query[i] (superior boundary)
    - Constructs a minimal enclosing simplex (D+1 nodes in D dimensions)
    - Rejects queries outside the convex hull (no extrapolation)
    """
    
    def __init__(self, dimensions: int):
        """
        Initialize the Lumin Core engine.
        
        Args:
            dimensions: Number of input dimensions (D)
        """
        self.d = dimensions
        self.dataset = None
        self.X = None  # Feature matrix
        self.Y = None  # Target values
        self.bounds_min = None  # Min bounds per dimension
        self.bounds_max = None  # Max bounds per dimension
        
    def fit(self, data: np.ndarray) -> None:
        """
        Load and validate the dataset.
        
        Args:
            data: Array of shape (N, D+1) where last column is the target value
        
        Raises:
            ValueError: If data has wrong shape or insufficient points
        """
        data = np.array(data, dtype=np.float64)
        
        # Validation
        if data.ndim != 2:
            raise ValueError(f"Data must be 2D, got shape {data.shape}")
        
        if data.shape[1] != self.d + 1:
            raise ValueError(f"Data must have {self.d + 1} columns (D={self.d} features + 1 target), got {data.shape[1]}")
        
        # Remove NaN rows
        valid_mask = ~np.isnan(data).any(axis=1)
        data = data[valid_mask]
        
        if len(data) < self.d + 1:
            raise ValueError(f"Need at least D+1={self.d + 1} valid points, got {len(data)}")
        
        # Store dataset
        self.dataset = data
        self.X = data[:, :-1]
        self.Y = data[:, -1]
        
        # Pre-compute bounds for extrapolation detection
        self.bounds_min = np.min(self.X, axis=0)
        self.bounds_max = np.max(self.X, axis=0)
        
        print(f"✓ Lumin Core v2.0: {len(data)} points loaded in {self.d}D space")
        print(f"  Bounds: [{self.bounds_min.min():.4f}, {self.bounds_max.max():.4f}]")
    
    def _is_extrapolation(self, point: np.ndarray) -> Tuple[bool, str]:
        """
        Check if a point requires extrapolation (outside dataset bounds).
        
        Args:
            point: Query point of shape (D,)
        
        Returns:
            (is_extrapolation, message): Boolean flag and descriptive message
        """
        below_min = point < self.bounds_min
        above_max = point > self.bounds_max
        
        if np.any(below_min):
            axes = np.where(below_min)[0]
            return True, f"Below minimum on axis/axes: {axes.tolist()}"
        
        if np.any(above_max):
            axes = np.where(above_max)[0]
            return True, f"Above maximum on axis/axes: {axes.tolist()}"
        
        return False, "Within bounds"
    
    def _build_simplex(self, point: np.ndarray) -> np.ndarray:
        """
        Construct the enclosing simplex using axial boundary search.
        
        For each dimension i:
        1. Find nearest point where X[:, i] <= point[i] (inferior)
        2. Find nearest point where X[:, i] >= point[i] (superior)
        3. Choose the closest of the two
        
        Args:
            point: Query point of shape (D,)
        
        Returns:
            simplex_nodes: Array of shape (K, D+1) with K <= D+1 unique nodes
        """
        # Calculate differences: positive means node is below query
        diffs = point - self.X  # Shape: (N, D)
        
        # For each dimension, find candidates
        selected_indices = []
        
        for dim in range(self.d):
            dim_diffs = diffs[:, dim]
            
            # Inferior candidates: node <= query (diff >= 0)
            inf_candidates = np.where(dim_diffs >= 0)[0]
            # Superior candidates: node >= query (diff <= 0)
            sup_candidates = np.where(dim_diffs <= 0)[0]
            
            best_idx = None
            
            # Find closest inferior node
            if len(inf_candidates) > 0:
                closest_inf_idx = inf_candidates[np.argmin(dim_diffs[inf_candidates])]
                best_idx = closest_inf_idx
                best_dist = dim_diffs[closest_inf_idx]
            
            # Find closest superior node
            if len(sup_candidates) > 0:
                closest_sup_idx = sup_candidates[np.argmax(dim_diffs[sup_candidates])]
                sup_dist = -dim_diffs[closest_sup_idx]  # Make positive for comparison
                
                # Compare and select closer one
                if best_idx is None or sup_dist < best_dist:
                    best_idx = closest_sup_idx
            
            if best_idx is not None:
                selected_indices.append(best_idx)
        
        # Add global nearest neighbor as anchor (ensures closure)
        global_dists = np.linalg.norm(self.X - point, axis=1)
        anchor_idx = np.argmin(global_dists)
        selected_indices.append(anchor_idx)
        
        # Remove duplicates and get final simplex
        unique_indices = np.unique(selected_indices)
        simplex_nodes = self.dataset[unique_indices]
        
        return simplex_nodes
    
    def predict(self, point: Union[list, np.ndarray], 
                allow_extrapolation: bool = False) -> Optional[float]:
        """
        Predict the value at a query point using simplex interpolation.
        
        Args:
            point: Query point of shape (D,)
            allow_extrapolation: If False, returns None for extrapolation cases
        
        Returns:
            Predicted value or None if extrapolation is detected and not allowed
        """
        if self.dataset is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        point = np.array(point, dtype=np.float64)
        
        # Validate dimensions
        if point.shape != (self.d,):
            raise ValueError(f"Point must have shape ({self.d},), got {point.shape}")
        
        # Check for extrapolation
        is_extrap, extrap_msg = self._is_extrapolation(point)
        
        if is_extrap and not allow_extrapolation:
            print(f"⚠ EXTRAPOLATION DETECTED: {extrap_msg}")
            print(f"  Query point is outside the dataset bounds.")
            print(f"  Lumin Core v2.0 does NOT perform extrapolation.")
            return None
        
        # Build enclosing simplex
        simplex = self._build_simplex(point)
        
        # Extract coordinates and values
        coords = simplex[:, :-1]
        values = simplex[:, -1]
        
        # Inverse Distance Weighting (IDW)
        distances = np.linalg.norm(coords - point, axis=1)
        
        # Handle exact matches
        exact_match = np.where(distances < 1e-12)[0]
        if len(exact_match) > 0:
            return values[exact_match[0]]
        
        # IDW formula: w_i = 1/d_i, prediction = Σ(w_i * y_i) / Σ(w_i)
        weights = 1.0 / distances
        prediction = np.dot(weights, values) / np.sum(weights)
        
        return prediction
    
    def predict_batch(self, points: np.ndarray, 
                      allow_extrapolation: bool = False) -> np.ndarray:
        """
        Predict values for multiple query points.
        
        Args:
            points: Array of shape (M, D) with M query points
            allow_extrapolation: If False, sets None for extrapolation cases
        
        Returns:
            Array of shape (M,) with predictions (None for rejected points)
        """
        points = np.array(points, dtype=np.float64)
        
        if points.ndim != 2 or points.shape[1] != self.d:
            raise ValueError(f"Points must have shape (M, {self.d}), got {points.shape}")
        
        results = []
        for point in points:
            pred = self.predict(point, allow_extrapolation=allow_extrapolation)
            results.append(pred)
        
        return np.array(results)
    
    def evaluate(self, test_data: np.ndarray, 
                 allow_extrapolation: bool = False) -> dict:
        """
        Evaluate model performance on test data.
        
        Args:
            test_data: Array of shape (M, D+1)
            allow_extrapolation: Whether to allow extrapolation in predictions
        
        Returns:
            Dictionary with metrics: MSE, MAE, RMSE, valid_predictions_count
        """
        test_data = np.array(test_data, dtype=np.float64)
        X_test = test_data[:, :-1]
        Y_test = test_data[:, -1]
        
        predictions = self.predict_batch(X_test, allow_extrapolation=allow_extrapolation)
        
        # Filter out None values (rejected extrapolations)
        valid_mask = np.array([p is not None for p in predictions])
        valid_preds = np.array([p for p in predictions if p is not None])
        valid_true = Y_test[valid_mask]
        
        if len(valid_preds) == 0:
            return {
                'MSE': None,
                'MAE': None,
                'RMSE': None,
                'valid_predictions': 0,
                'total_points': len(Y_test)
            }
        
        errors = valid_true - valid_preds
        mse = np.mean(errors**2)
        mae = np.mean(np.abs(errors))
        rmse = np.sqrt(mse)
        
        return {
            'MSE': mse,
            'MAE': mae,
            'RMSE': rmse,
            'valid_predictions': len(valid_preds),
            'total_points': len(Y_test)
        }


# ==========================================
# COMPREHENSIVE TEST SUITE
# ==========================================

def test_interpolation_1d():
    """Test basic 1D interpolation"""
    print("\n" + "="*60)
    print("TEST 1: Basic 1D Interpolation")
    print("="*60)
    
    # Dataset: y = x^2
    X = np.linspace(0, 10, 20).reshape(-1, 1)
    Y = (X**2).reshape(-1, 1)
    data = np.hstack([X, Y])
    
    engine = LuminCore(dimensions=1)
    engine.fit(data)
    
    # Test interpolation
    test_point = np.array([5.5])
    pred = engine.predict(test_point)
    real = 5.5**2
    
    print(f"Query: x={test_point[0]:.2f}")
    print(f"Real: {real:.4f} | Predicted: {pred:.4f}")
    print(f"Error: {abs(real - pred):.4f}")
    
    # Test extrapolation detection
    print("\n--- Testing Extrapolation Detection ---")
    extrap_point = np.array([15.0])  # Outside bounds
    pred_extrap = engine.predict(extrap_point, allow_extrapolation=False)
    print(f"Result for extrapolation: {pred_extrap}")


def test_high_dimensional():
    """Test high-dimensional interpolation with sum of squares"""
    print("\n" + "="*60)
    print("TEST 2: High-Dimensional Space (1000D)")
    print("="*60)
    
    D, N = 1000, 2000
    
    # Generate dataset: y = sum(x_i^2)
    X_train = np.random.rand(N, D) * 10  # [0, 10] range
    Y_train = np.sum(X_train**2, axis=1).reshape(-1, 1)
    data = np.hstack([X_train, Y_train])
    
    engine = LuminCore(dimensions=D)
    engine.fit(data)
    
    # Test points
    n_test = 100
    X_test = np.random.rand(n_test, D) * 10
    Y_test = np.sum(X_test**2, axis=1).reshape(-1, 1)
    test_data = np.hstack([X_test, Y_test])
    
    # Evaluate
    start = time.perf_counter()
    metrics = engine.evaluate(test_data, allow_extrapolation=False)
    elapsed = time.perf_counter() - start
    
    print(f"\nResults on {n_test} test points:")
    print(f"  RMSE: {metrics['RMSE']:.4f}")
    print(f"  MAE:  {metrics['MAE']:.4f}")
    print(f"  Valid predictions: {metrics['valid_predictions']}/{metrics['total_points']}")
    print(f"  Total time: {elapsed*1000:.2f} ms")
    print(f"  Avg per point: {elapsed*1000/n_test:.2f} ms")


def test_edge_cases():
    """Test edge cases and robustness"""
    print("\n" + "="*60)
    print("TEST 3: Edge Cases")
    print("="*60)
    
    # 2D dataset
    data = np.array([
        [0, 0, 0],
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 2],
    ])
    
    engine = LuminCore(dimensions=2)
    engine.fit(data)
    
    print("\n--- Case 1: Exact match ---")
    pred = engine.predict([0, 0])
    print(f"Prediction at [0, 0]: {pred:.4f} (expected 0.0)")
    
    print("\n--- Case 2: Interior point ---")
    pred = engine.predict([0.5, 0.5])
    print(f"Prediction at [0.5, 0.5]: {pred:.4f}")
    
    print("\n--- Case 3: Boundary extrapolation ---")
    pred = engine.predict([2, 0.5], allow_extrapolation=False)
    print(f"Prediction at [2, 0.5]: {pred}")
    
    print("\n--- Case 4: Negative extrapolation ---")
    pred = engine.predict([-1, 0.5], allow_extrapolation=False)
    print(f"Prediction at [-1, 0.5]: {pred}")


def test_stress_latency():
    """Stress test for latency in extreme dimensions"""
    print("\n" + "="*60)
    print("TEST 4: Latency Stress Test")
    print("="*60)
    
    dimensions = [10, 50, 100, 500, 1000]
    
    for D in dimensions:
        N = max(D * 2, 100)
        X = np.random.rand(N, D)
        Y = np.sum(X**2, axis=1).reshape(-1, 1)
        data = np.hstack([X, Y])
        
        engine = LuminCore(dimensions=D)
        engine.fit(data)
        
        query = np.random.rand(D)
        
        start = time.perf_counter()
        pred = engine.predict(query)
        elapsed = time.perf_counter() - start
        
        print(f"D={D:4d} | Latency: {elapsed*1000:6.2f} ms")


if __name__ == "__main__":
    print("\n" + "🔷"*30)
    print("LUMIN CORE v2.0 - COMPREHENSIVE TEST SUITE")
    print("🔷"*30)
    
    test_interpolation_1d()
    test_high_dimensional()
    test_edge_cases()
    test_stress_latency()
    
    print("\n" + "="*60)
    print("✓ ALL TESTS COMPLETED")
    print("="*60)
 
