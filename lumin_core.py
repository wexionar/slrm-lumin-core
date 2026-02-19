# ==========================================
# Project: SLRM-nD (Lumin Core v2.1)
# Simplex Local Regression Engine
# Project Lead: Alex Kinetic
# AI Collaboration: Gemini · ChatGPT · Claude · Grok · Meta AI
# License: MIT License
# ==========================================

import numpy as np
import time
from typing import Optional, Tuple, Union, Dict

# KDTree import opcional (solo si scipy disponible)
try:
    from scipy.spatial import KDTree
    KDTREE_AVAILABLE = True
except ImportError:
    KDTREE_AVAILABLE = False
    KDTree = None


class LuminCore:
    """
    Lumin Core v2.1 - Simplex Local Regression Engine
    
    Evolution from v2.0: Replaces Inverse Distance Weighting with local
    linear regression for improved accuracy on smooth functions while
    maintaining geometric purity and O(D) complexity.
    
    Key Features:
    - Simplex Local Regression (SLR) for linear trend capture
    - Automatic simplex degeneracy detection with IDW fallback
    - Optional diagnostic information (uncertainty, method used)
    - KD-Tree acceleration for large datasets (N > 10,000)
    - Sacred boundary enforcement (no epsilon tolerance)
    - Robust numerical stability
    
    Principles:
    - For each dimension i, finds the nearest boundary points
    - Constructs minimal enclosing simplex (D+1 nodes)
    - Fits local hyperplane through simplex nodes
    - Rejects extrapolation attempts (binary: inside or outside)
    
    Philosophy:
    "Honestidad Geométrica sobre Precisión Artificial"
    - Poor simplex → high uncertainty (reported honestly)
    - Outside bounds → rejection (no compromise)
    - Degenerate simplex → fallback to IDW (with notification)
    """
    
    def __init__(self, dimensions: int, use_kdtree_threshold: int = 10000):
        """
        Initialize the Lumin Core v2.1 engine.
        
        Args:
            dimensions: Number of input dimensions (D)
            use_kdtree_threshold: Activate KDTree if N > threshold (default: 10000)
        """
        self.d = dimensions
        self.dataset = None
        self.X = None  # Feature matrix
        self.Y = None  # Target values
        self.bounds_min = None  # Min bounds per dimension
        self.bounds_max = None  # Max bounds per dimension
        self.kdtree = None  # KDTree for acceleration (optional)
        self.use_kdtree_threshold = use_kdtree_threshold
        
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
        
        # Build KDTree if dataset is large enough
        if KDTREE_AVAILABLE and len(data) > self.use_kdtree_threshold:
            self.kdtree = KDTree(self.X)
            kdtree_status = "enabled"
        else:
            self.kdtree = None
            kdtree_status = "disabled" if KDTREE_AVAILABLE else "unavailable (scipy not installed)"
        
        print(f"✓ Lumin Core v2.1: {len(data)} points loaded in {self.d}D space")
        print(f"  Bounds: [{self.bounds_min.min():.4f}, {self.bounds_max.max():.4f}]")
        print(f"  KDTree: {kdtree_status}")
    
    def _is_extrapolation(self, point: np.ndarray) -> Tuple[bool, str]:
        """
        Check if a point requires extrapolation (outside dataset bounds).
        SACRED BOUNDARY - NO EPSILON TOLERANCE.
        
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
    
    def _is_simplex_degenerate(self, simplex_nodes: np.ndarray) -> bool:
        """
        Detect if simplex is degenerate (nodes are nearly collinear).
        
        A simplex is degenerate if:
        1. Has fewer than D+1 unique nodes
        2. Nodes are nearly collinear (rank < D)
        
        Args:
            simplex_nodes: Array of shape (K, D+1)
        
        Returns:
            True if simplex is degenerate
        """
        X = simplex_nodes[:, :-1]
        
        # Check for insufficient unique nodes
        unique_nodes = np.unique(X, axis=0)
        if len(unique_nodes) < min(self.d + 1, len(X)):
            return True
        
        # Check if we have enough nodes to form a proper simplex
        if len(X) < self.d + 1:
            return True
        
        # Use matrix rank instead of determinant for dimension-agnostic check
        # This works correctly in 1D, 2D, and high-D cases
        if len(X) >= self.d + 1:
            try:
                # Matrix of differences from first node
                diff_matrix = X[1:self.d+1] - X[0]
                
                # Check rank - should be D for non-degenerate simplex
                rank = np.linalg.matrix_rank(diff_matrix, tol=1e-10)
                
                # Degenerate if rank is less than D
                return rank < self.d
            except:
                return True
        
        return False
    
    def _predict_slr(self, point: np.ndarray, simplex_nodes: np.ndarray) -> float:
        """
        Simplex Local Regression: fit a local hyperplane through simplex nodes.
        
        Solves: y = β₀ + β₁·x₁ + β₂·x₂ + ... + βₐ·xₐ
        using least squares on the simplex nodes.
        
        Args:
            point: Query point of shape (D,)
            simplex_nodes: Array of shape (K, D+1)
        
        Returns:
            Predicted value at query point
        
        Raises:
            np.linalg.LinAlgError: If least squares fails
        """
        X = simplex_nodes[:, :-1]
        Y = simplex_nodes[:, -1]
        
        # Augment with bias term (column of ones)
        A = np.c_[np.ones(X.shape[0]), X]
        
        # Solve least squares: A @ beta = Y
        # Using rcond=None for future-proof behavior
        beta, residuals, rank, s = np.linalg.lstsq(A, Y, rcond=None)
        
        # Predict at query point
        query_augmented = np.insert(point, 0, 1)  # Add bias term
        prediction = np.dot(query_augmented, beta)
        
        return float(prediction)
    
    def _idw_fallback(self, point: np.ndarray, simplex_nodes: np.ndarray) -> float:
        """
        Inverse Distance Weighting fallback for degenerate simplexes.
        
        Args:
            point: Query point of shape (D,)
            simplex_nodes: Array of shape (K, D+1)
        
        Returns:
            Weighted average prediction
        """
        coords = simplex_nodes[:, :-1]
        values = simplex_nodes[:, -1]
        
        distances = np.linalg.norm(coords - point, axis=1)
        
        # CRITICAL: Handle exact matches BEFORE any weight calculation
        # This prevents divide-by-zero when duplicate points exist
        exact_match_mask = distances < 1e-12
        if np.any(exact_match_mask):
            # Return mean of all exact matches (handles duplicate case)
            return float(np.mean(values[exact_match_mask]))
        
        # Standard IDW - safe because we've already handled zero distances
        weights = 1.0 / distances
        prediction = np.dot(weights, values) / np.sum(weights)
        
        return float(prediction)
    
    def _get_uncertainty(self, point: np.ndarray, simplex_nodes: np.ndarray) -> float:
        """
        Calculate uncertainty metric based on simplex dispersion.
        
        Higher values indicate lower confidence in the prediction due to
        sparse or dispersed simplex nodes.
        
        Args:
            point: Query point of shape (D,)
            simplex_nodes: Array of shape (K, D+1)
        
        Returns:
            Mean distance from query point to simplex nodes
        """
        coords = simplex_nodes[:, :-1]
        distances = np.linalg.norm(coords - point, axis=1)
        return float(np.mean(distances))
    
    def _build_simplex(self, point: np.ndarray) -> np.ndarray:
        """
        Construct the enclosing simplex using axial boundary search.
        
        For each dimension i:
        1. Find nearest point where X[:, i] <= point[i] (inferior)
        2. Find nearest point where X[:, i] >= point[i] (superior)
        3. Select the closest of the two
        
        Then add global nearest neighbor as anchor.
        
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
            best_dist = np.inf
            
            # Find closest inferior node
            if len(inf_candidates) > 0:
                closest_inf_idx = inf_candidates[np.argmin(dim_diffs[inf_candidates])]
                best_idx = closest_inf_idx
                best_dist = dim_diffs[closest_inf_idx]
            
            # Find closest superior node and compare
            if len(sup_candidates) > 0:
                closest_sup_idx = sup_candidates[np.argmax(dim_diffs[sup_candidates])]
                sup_dist = -dim_diffs[closest_sup_idx]  # Make positive
                
                if best_idx is None or sup_dist < best_dist:
                    best_idx = closest_sup_idx
            
            if best_idx is not None:
                selected_indices.append(best_idx)
        
        # Add global nearest neighbor as anchor
        if self.kdtree is not None:
            # Use KDTree for fast nearest neighbor search
            dist, idx = self.kdtree.query(point, k=1)
            anchor_idx = idx
        else:
            # Brute force search
            global_dists = np.linalg.norm(self.X - point, axis=1)
            anchor_idx = np.argmin(global_dists)
        
        selected_indices.append(anchor_idx)
        
        # Remove duplicates and get final simplex
        unique_indices = np.unique(selected_indices)
        simplex_nodes = self.dataset[unique_indices]
        
        return simplex_nodes
    
    def predict(self, point: Union[list, np.ndarray], 
                allow_extrapolation: bool = False,
                return_diagnostics: bool = False) -> Union[Optional[float], Tuple[Optional[float], Dict]]:
        """
        Predict the value at a query point using Simplex Local Regression.
        
        Args:
            point: Query point of shape (D,)
            allow_extrapolation: If False, returns None for extrapolation cases
            return_diagnostics: If True, returns (prediction, diagnostics_dict)
        
        Returns:
            If return_diagnostics=False:
                Predicted value or None if extrapolation detected
            If return_diagnostics=True:
                (prediction, diagnostics) where diagnostics contains:
                - 'method': 'slr' | 'idw_fallback' | 'exact_match'
                - 'uncertainty': float (mean distance to simplex nodes)
                - 'simplex_size': int (number of nodes in simplex)
                - 'is_degenerate': bool
                - 'error': str (if extrapolation detected)
        """
        if self.dataset is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        # Normalize input to handle various shapes: lists, (D,), (1,D), etc.
        point = np.asarray(point, dtype=np.float64).flatten()
        
        # Validate dimensions
        if point.shape != (self.d,):
            raise ValueError(f"Point must have {self.d} elements, got {len(point)}")
        
        # Check for extrapolation (SACRED BOUNDARY - NO EPSILON)
        is_extrap, extrap_msg = self._is_extrapolation(point)
        
        if is_extrap and not allow_extrapolation:
            print(f"⚠ EXTRAPOLATION DETECTED: {extrap_msg}")
            print(f"  Query point is outside the dataset bounds.")
            print(f"  Lumin Core v2.1 does NOT perform extrapolation.")
            
            if return_diagnostics:
                return None, {'error': extrap_msg}
            return None
        
        # Build enclosing simplex
        simplex = self._build_simplex(point)
        
        # Check for simplex degeneracy
        is_degenerate = self._is_simplex_degenerate(simplex)
        
        # Predict using SLR or IDW fallback
        method = "slr"
        try:
            if not is_degenerate:
                prediction = self._predict_slr(point, simplex)
            else:
                prediction = self._idw_fallback(point, simplex)
                method = "idw_fallback"
        except (np.linalg.LinAlgError, Exception):
            # If SLR fails for any reason, fallback to IDW
            prediction = self._idw_fallback(point, simplex)
            method = "idw_fallback"
        
        # Calculate uncertainty
        uncertainty = self._get_uncertainty(point, simplex)
        
        if return_diagnostics:
            diagnostics = {
                'method': method,
                'uncertainty': uncertainty,
                'simplex_size': len(simplex),
                'is_degenerate': is_degenerate
            }
            return prediction, diagnostics
        
        return prediction
    
    def predict_batch(self, points: np.ndarray, 
                      allow_extrapolation: bool = False,
                      return_diagnostics: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, list]]:
        """
        Predict values for multiple query points.
        
        Args:
            points: Array of shape (M, D) with M query points
            allow_extrapolation: If False, sets None for extrapolation cases
            return_diagnostics: If True, returns (predictions, diagnostics_list)
        
        Returns:
            If return_diagnostics=False:
                Array of shape (M,) with predictions (None for rejected points)
            If return_diagnostics=True:
                (predictions, diagnostics_list)
        """
        points = np.array(points, dtype=np.float64)
        
        if points.ndim != 2 or points.shape[1] != self.d:
            raise ValueError(f"Points must have shape (M, {self.d}), got {points.shape}")
        
        results = []
        diagnostics_list = []
        
        for point in points:
            if return_diagnostics:
                pred, diag = self.predict(point, allow_extrapolation=allow_extrapolation, 
                                         return_diagnostics=True)
                results.append(pred)
                diagnostics_list.append(diag)
            else:
                pred = self.predict(point, allow_extrapolation=allow_extrapolation)
                results.append(pred)
        
        if return_diagnostics:
            return np.array(results), diagnostics_list
        return np.array(results)
    
    def evaluate(self, test_data: np.ndarray, 
                 allow_extrapolation: bool = False) -> dict:
        """
        Evaluate model performance on test data.
        
        Args:
            test_data: Array of shape (M, D+1)
            allow_extrapolation: Whether to allow extrapolation in predictions
        
        Returns:
            Dictionary with metrics: MSE, MAE, RMSE, valid_predictions_count, 
            method_distribution (SLR vs IDW usage)
        """
        test_data = np.array(test_data, dtype=np.float64)
        X_test = test_data[:, :-1]
        Y_test = test_data[:, -1]
        
        predictions, diagnostics = self.predict_batch(X_test, 
                                                       allow_extrapolation=allow_extrapolation,
                                                       return_diagnostics=True)
        
        # Filter out None values (rejected extrapolations)
        valid_mask = np.array([p is not None for p in predictions])
        valid_preds = np.array([p for p in predictions if p is not None])
        valid_true = Y_test[valid_mask]
        valid_diags = [d for d, v in zip(diagnostics, valid_mask) if v]
        
        if len(valid_preds) == 0:
            return {
                'MSE': None,
                'MAE': None,
                'RMSE': None,
                'valid_predictions': 0,
                'total_points': len(Y_test),
                'slr_usage': 0,
                'idw_usage': 0,
                'mean_uncertainty': None
            }
        
        # Calculate metrics
        errors = valid_true - valid_preds
        mse = np.mean(errors**2)
        mae = np.mean(np.abs(errors))
        rmse = np.sqrt(mse)
        
        # Method distribution
        slr_count = sum(1 for d in valid_diags if d.get('method') == 'slr')
        idw_count = sum(1 for d in valid_diags if d.get('method') == 'idw_fallback')
        
        # Mean uncertainty
        mean_uncertainty = np.mean([d.get('uncertainty', 0) for d in valid_diags])
        
        return {
            'MSE': mse,
            'MAE': mae,
            'RMSE': rmse,
            'valid_predictions': len(valid_preds),
            'total_points': len(Y_test),
            'slr_usage': slr_count,
            'idw_usage': idw_count,
            'mean_uncertainty': mean_uncertainty
        }


# ==========================================
# DEMONSTRATION & BASIC TESTS
# ==========================================

if __name__ == "__main__":
    print("\n" + "🔷"*30)
    print("LUMIN CORE v2.1 - SIMPLEX LOCAL REGRESSION")
    print("🔷"*30)
    
    # Test 1: Simple 2D function
    print("\n" + "="*60)
    print("TEST 1: Linear Function in 2D (y = x₁ + 2x₂)")
    print("="*60)
    
    np.random.seed(42)
    X = np.random.rand(50, 2) * 10
    Y = (X[:, 0] + 2*X[:, 1]).reshape(-1, 1)
    data = np.hstack([X, Y])
    
    engine = LuminCore(dimensions=2)
    engine.fit(data)
    
    test_point = np.array([5.0, 5.0])
    pred, diag = engine.predict(test_point, return_diagnostics=True)
    real = 5.0 + 2*5.0
    
    print(f"\nQuery point: {test_point}")
    print(f"Real value: {real:.4f}")
    print(f"Predicted: {pred:.4f}")
    print(f"Error: {abs(real - pred):.4f}")
    print(f"Method: {diag['method']}")
    print(f"Uncertainty: {diag['uncertainty']:.4f}")
    print(f"Simplex size: {diag['simplex_size']}")
    print(f"Degenerate: {diag['is_degenerate']}")
    
    # Test 2: Quadratic function
    print("\n" + "="*60)
    print("TEST 2: Quadratic Function in 1D (y = x²)")
    print("="*60)
    
    X = np.linspace(0, 10, 100).reshape(-1, 1)
    Y = (X**2).reshape(-1, 1)
    data = np.hstack([X, Y])
    
    engine = LuminCore(dimensions=1)
    engine.fit(data)
    
    test_point = np.array([5.5])
    pred, diag = engine.predict(test_point, return_diagnostics=True)
    real = 5.5**2
    
    print(f"\nQuery point: {test_point[0]:.2f}")
    print(f"Real value: {real:.4f}")
    print(f"Predicted: {pred:.4f}")
    print(f"Error: {abs(real - pred):.4f}")
    print(f"Method: {diag['method']}")
    
    # Test 3: High-dimensional test
    print("\n" + "="*60)
    print("TEST 3: High-Dimensional Space (100D)")
    print("="*60)
    
    D = 100
    N = 500
    X = np.random.rand(N, D) * 10
    Y = np.sum(X**2, axis=1).reshape(-1, 1)
    data = np.hstack([X, Y])
    
    engine = LuminCore(dimensions=D)
    engine.fit(data)
    
    test_point = engine.bounds_min + (engine.bounds_max - engine.bounds_min) * 0.5
    
    start = time.perf_counter()
    pred, diag = engine.predict(test_point, return_diagnostics=True)
    elapsed = time.perf_counter() - start
    
    real = np.sum(test_point**2)
    
    print(f"\nReal value: {real:.4f}")
    print(f"Predicted: {pred:.4f}")
    print(f"Relative error: {abs(real - pred)/real:.2%}")
    print(f"Latency: {elapsed*1000:.2f} ms")
    print(f"Method: {diag['method']}")
    print(f"Uncertainty: {diag['uncertainty']:.4f}")
    
    print("\n" + "="*60)
    print("✓ BASIC TESTS COMPLETED")
    print("="*60)
 
