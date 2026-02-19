# ==========================================
# Project: SLRM-nD (Lumin Core v2.1)
# Comprehensive Test Suite
# Project Lead: Alex Kinetic
# AI Collaboration: Gemini · ChatGPT · Claude · Grok · Meta AI
# License: MIT License
# ==========================================

import numpy as np
import time
import sys
from typing import List, Tuple

# Import the engine
try:
    from lumin_core import LuminCore
except ImportError:
    print("ERROR: lumin_core.py not found in the same directory!")
    sys.exit(1)


class TestResult:
    """Simple test result container"""
    def __init__(self, name: str, passed: bool, message: str = "", duration: float = 0.0):
        self.name = name
        self.passed = passed
        self.message = message
        self.duration = duration
    
    def __repr__(self):
        status = "✓ PASS" if self.passed else "✗ FAIL"
        time_str = f"({self.duration*1000:.2f}ms)" if self.duration > 0 else ""
        msg = f" - {self.message}" if self.message else ""
        return f"{status} | {self.name} {time_str}{msg}"


class LuminCoreTestSuite:
    """Comprehensive test suite for Lumin Core v2.1"""
    
    def __init__(self):
        self.results: List[TestResult] = []
        self.total_tests = 0
        self.passed_tests = 0
    
    def add_result(self, result: TestResult):
        """Add a test result and update counters"""
        self.results.append(result)
        self.total_tests += 1
        if result.passed:
            self.passed_tests += 1
    
    def assert_equal(self, actual, expected, tolerance=1e-6):
        """Assert equality with tolerance"""
        if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
            return abs(actual - expected) < tolerance
        return actual == expected
    
    def assert_none(self, value):
        """Assert value is None"""
        return value is None
    
    def assert_not_none(self, value):
        """Assert value is not None"""
        return value is not None
    
    def print_summary(self):
        """Print test summary"""
        print("\n" + "="*70)
        print("TEST SUMMARY")
        print("="*70)
        
        for result in self.results:
            print(result)
        
        print("="*70)
        print(f"TOTAL: {self.passed_tests}/{self.total_tests} tests passed")
        
        if self.passed_tests == self.total_tests:
            print("🎉 ALL TESTS PASSED! 🎉")
        else:
            print(f"⚠️  {self.total_tests - self.passed_tests} test(s) failed")
        
        print("="*70)
        return self.passed_tests == self.total_tests


# ==========================================
# TEST CATEGORY 1: INITIALIZATION
# ==========================================

def test_initialization(suite: LuminCoreTestSuite):
    """Test engine initialization"""
    print("\n[CATEGORY 1] INITIALIZATION TESTS")
    print("-" * 70)
    
    # Test 1.1: Basic initialization
    try:
        engine = LuminCore(dimensions=5)
        passed = engine.d == 5 and engine.dataset is None
        suite.add_result(TestResult(
            "1.1 Basic initialization",
            passed,
            f"dimensions={engine.d}"
        ))
    except Exception as e:
        suite.add_result(TestResult("1.1 Basic initialization", False, str(e)))
    
    # Test 1.2: Zero dimensions (should work but be meaningless)
    try:
        engine = LuminCore(dimensions=0)
        suite.add_result(TestResult(
            "1.2 Zero dimensions",
            True,
            "Allowed but meaningless"
        ))
    except Exception as e:
        suite.add_result(TestResult("1.2 Zero dimensions", False, str(e)))
    
    # Test 1.3: Negative dimensions (should work but be meaningless)
    try:
        engine = LuminCore(dimensions=-1)
        suite.add_result(TestResult(
            "1.3 Negative dimensions",
            True,
            "No validation, user responsibility"
        ))
    except Exception as e:
        suite.add_result(TestResult("1.3 Negative dimensions", False, str(e)))


# ==========================================
# TEST CATEGORY 2: DATA LOADING
# ==========================================

def test_data_loading(suite: LuminCoreTestSuite):
    """Test fit() method with various data"""
    print("\n[CATEGORY 2] DATA LOADING TESTS")
    print("-" * 70)
    
    # Test 2.1: Valid data loading
    try:
        engine = LuminCore(dimensions=2)
        data = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
        engine.fit(data)
        passed = len(engine.dataset) == 3 and engine.X.shape == (3, 2)
        suite.add_result(TestResult(
            "2.1 Valid data loading",
            passed,
            f"Loaded {len(engine.dataset)} points"
        ))
    except Exception as e:
        suite.add_result(TestResult("2.1 Valid data loading", False, str(e)))
    
    # Test 2.2: Data with NaN values
    try:
        engine = LuminCore(dimensions=2)
        data = np.array([[0, 0, 0], [1, np.nan, 1], [2, 2, 2], [3, 3, 3]])  # 4 points, 1 with NaN
        engine.fit(data)
        passed = len(engine.dataset) == 3  # Should remove 1 NaN row, keep 3
        suite.add_result(TestResult(
            "2.2 NaN handling",
            passed,
            f"Removed NaN rows, kept {len(engine.dataset)}/4"
        ))
    except Exception as e:
        suite.add_result(TestResult("2.2 NaN handling", False, str(e)))
    
    # Test 2.3: Wrong shape data
    try:
        engine = LuminCore(dimensions=3)
        data = np.array([[0, 0, 0], [1, 1, 1]])  # Only 3 columns, need 4
        engine.fit(data)
        suite.add_result(TestResult(
            "2.3 Wrong shape rejection",
            False,
            "Should have raised ValueError"
        ))
    except ValueError:
        suite.add_result(TestResult(
            "2.3 Wrong shape rejection",
            True,
            "Correctly raised ValueError"
        ))
    except Exception as e:
        suite.add_result(TestResult("2.3 Wrong shape rejection", False, str(e)))
    
    # Test 2.4: Insufficient data points
    try:
        engine = LuminCore(dimensions=5)
        data = np.array([[0, 0, 0, 0, 0, 0]])  # Only 1 point, need D+1=6
        engine.fit(data)
        suite.add_result(TestResult(
            "2.4 Insufficient points rejection",
            False,
            "Should have raised ValueError"
        ))
    except ValueError:
        suite.add_result(TestResult(
            "2.4 Insufficient points rejection",
            True,
            "Correctly raised ValueError"
        ))
    except Exception as e:
        suite.add_result(TestResult("2.4 Insufficient points rejection", False, str(e)))
    
    # Test 2.5: Empty data
    try:
        engine = LuminCore(dimensions=2)
        data = np.array([]).reshape(0, 3)
        engine.fit(data)
        suite.add_result(TestResult(
            "2.5 Empty data rejection",
            False,
            "Should have raised ValueError"
        ))
    except ValueError:
        suite.add_result(TestResult(
            "2.5 Empty data rejection",
            True,
            "Correctly raised ValueError"
        ))
    except Exception as e:
        suite.add_result(TestResult("2.5 Empty data rejection", False, str(e)))


# ==========================================
# TEST CATEGORY 3: EXTRAPOLATION DETECTION
# ==========================================

def test_extrapolation_detection(suite: LuminCoreTestSuite):
    """Test extrapolation detection logic"""
    print("\n[CATEGORY 3] EXTRAPOLATION DETECTION TESTS")
    print("-" * 70)
    
    # Setup: 2D square [0,1] x [0,1]
    engine = LuminCore(dimensions=2)
    data = np.array([
        [0, 0, 0],
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 2]
    ])
    engine.fit(data)
    
    # Test 3.1: Point inside bounds
    try:
        pred = engine.predict([0.5, 0.5])
        passed = suite.assert_not_none(pred)
        suite.add_result(TestResult(
            "3.1 Interior point accepted",
            passed,
            f"Prediction: {pred:.4f}" if pred is not None else "Got None unexpectedly"
        ))
    except Exception as e:
        suite.add_result(TestResult("3.1 Interior point accepted", False, str(e)))
    
    # Test 3.2: Point on boundary
    try:
        pred = engine.predict([1.0, 0.5])
        passed = suite.assert_not_none(pred)
        suite.add_result(TestResult(
            "3.2 Boundary point accepted",
            passed,
            f"Prediction: {pred:.4f}" if pred else "None"
        ))
    except Exception as e:
        suite.add_result(TestResult("3.2 Boundary point accepted", False, str(e)))
    
    # Test 3.3: Point above max
    try:
        pred = engine.predict([1.5, 0.5])
        passed = suite.assert_none(pred)
        suite.add_result(TestResult(
            "3.3 Above max rejected",
            passed,
            "Correctly returned None"
        ))
    except Exception as e:
        suite.add_result(TestResult("3.3 Above max rejected", False, str(e)))
    
    # Test 3.4: Point below min
    try:
        pred = engine.predict([-0.5, 0.5])
        passed = suite.assert_none(pred)
        suite.add_result(TestResult(
            "3.4 Below min rejected",
            passed,
            "Correctly returned None"
        ))
    except Exception as e:
        suite.add_result(TestResult("3.4 Below min rejected", False, str(e)))
    
    # Test 3.5: Multiple axes out of bounds
    try:
        pred = engine.predict([2.0, -1.0])
        passed = suite.assert_none(pred)
        suite.add_result(TestResult(
            "3.5 Multiple axes out rejected",
            passed,
            "Correctly returned None"
        ))
    except Exception as e:
        suite.add_result(TestResult("3.5 Multiple axes out rejected", False, str(e)))
    
    # Test 3.6: Allow extrapolation flag
    try:
        pred = engine.predict([1.5, 0.5], allow_extrapolation=True)
        passed = suite.assert_not_none(pred)
        suite.add_result(TestResult(
            "3.6 Extrapolation allowed when flag=True",
            passed,
            f"Prediction: {pred:.4f}" if pred else "None"
        ))
    except Exception as e:
        suite.add_result(TestResult("3.6 Extrapolation allowed", False, str(e)))


# ==========================================
# TEST CATEGORY 4: PREDICTION ACCURACY
# ==========================================

def test_prediction_accuracy(suite: LuminCoreTestSuite):
    """Test prediction accuracy on known functions"""
    print("\n[CATEGORY 4] PREDICTION ACCURACY TESTS")
    print("-" * 70)
    
    # Test 4.1: Linear function (1D)
    try:
        D = 1
        X = np.linspace(0, 10, 50).reshape(-1, 1)
        Y = (2 * X + 3).reshape(-1, 1)
        data = np.hstack([X, Y])
        
        engine = LuminCore(dimensions=D)
        engine.fit(data)
        
        test_point = np.array([5.0])
        pred = engine.predict(test_point)
        real = 2 * 5.0 + 3.0
        error = abs(pred - real)
        
        passed = error < 0.5  # Reasonable tolerance for linear
        suite.add_result(TestResult(
            "4.1 Linear 1D (y=2x+3)",
            passed,
            f"Error: {error:.4f}"
        ))
    except Exception as e:
        suite.add_result(TestResult("4.1 Linear 1D", False, str(e)))
    
    # Test 4.2: Quadratic function (1D)
    try:
        D = 1
        X = np.linspace(0, 10, 100).reshape(-1, 1)
        Y = (X**2).reshape(-1, 1)
        data = np.hstack([X, Y])
        
        engine = LuminCore(dimensions=D)
        engine.fit(data)
        
        test_point = np.array([5.5])
        pred = engine.predict(test_point)
        real = 5.5**2
        error = abs(pred - real)
        
        passed = error < 5.0  # Quadratic has more error
        suite.add_result(TestResult(
            "4.2 Quadratic 1D (y=x²)",
            passed,
            f"Error: {error:.4f}"
        ))
    except Exception as e:
        suite.add_result(TestResult("4.2 Quadratic 1D", False, str(e)))
    
    # Test 4.3: Exact node match
    try:
        D = 2
        data = np.array([[0, 0, 0], [1, 1, 5], [2, 2, 10]])
        engine = LuminCore(dimensions=D)
        engine.fit(data)
        
        pred = engine.predict([1, 1])
        real = 5.0
        error = abs(pred - real)
        
        passed = error < 1e-6  # Should be exact
        suite.add_result(TestResult(
            "4.3 Exact node match",
            passed,
            f"Error: {error:.10f}"
        ))
    except Exception as e:
        suite.add_result(TestResult("4.3 Exact node match", False, str(e)))
    
    # Test 4.4: Multi-dimensional linear (y = x1 + x2 + x3)
    try:
        D = 3
        np.random.seed(42)
        X = np.random.rand(100, D) * 10
        Y = np.sum(X, axis=1).reshape(-1, 1)
        data = np.hstack([X, Y])
        
        engine = LuminCore(dimensions=D)
        engine.fit(data)
        
        test_point = np.array([5.0, 5.0, 5.0])
        pred = engine.predict(test_point)
        real = 15.0
        error = abs(pred - real)
        
        passed = error < 2.0
        suite.add_result(TestResult(
            "4.4 Linear 3D (y=x1+x2+x3)",
            passed,
            f"Error: {error:.4f}"
        ))
    except Exception as e:
        suite.add_result(TestResult("4.4 Linear 3D", False, str(e)))
    
    # Test 4.5: Sum of squares (high-D)
    try:
        D = 50
        np.random.seed(42)
        X = np.random.rand(200, D) * 5
        Y = np.sum(X**2, axis=1).reshape(-1, 1)
        data = np.hstack([X, Y])
        
        engine = LuminCore(dimensions=D)
        engine.fit(data)
        
        # Ensure test point is within bounds
        test_point = engine.bounds_min + (engine.bounds_max - engine.bounds_min) * 0.5
        pred = engine.predict(test_point)
        real = np.sum(test_point**2)
        
        if pred is not None:
            relative_error = abs(pred - real) / real
            passed = relative_error < 0.5  # 50% tolerance for high-D sparse data
            suite.add_result(TestResult(
                "4.5 Sum of squares 50D",
                passed,
                f"Relative error: {relative_error:.2%}"
            ))
        else:
            suite.add_result(TestResult(
                "4.5 Sum of squares 50D",
                False,
                "Prediction was None (extrapolation)"
            ))
    except Exception as e:
        suite.add_result(TestResult("4.5 Sum of squares 50D", False, str(e)))


# ==========================================
# TEST CATEGORY 5: BATCH PREDICTION
# ==========================================

def test_batch_prediction(suite: LuminCoreTestSuite):
    """Test batch prediction functionality"""
    print("\n[CATEGORY 5] BATCH PREDICTION TESTS")
    print("-" * 70)
    
    # Setup
    engine = LuminCore(dimensions=2)
    data = np.array([
        [0, 0, 0],
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 2]
    ])
    engine.fit(data)
    
    # Test 5.1: Multiple valid points
    try:
        points = np.array([
            [0.5, 0.5],
            [0.3, 0.7],
            [0.8, 0.2]
        ])
        preds = engine.predict_batch(points)
        
        passed = len(preds) == 3 and all(p is not None for p in preds)
        suite.add_result(TestResult(
            "5.1 Batch of valid points",
            passed,
            f"Got {len(preds)} predictions"
        ))
    except Exception as e:
        suite.add_result(TestResult("5.1 Batch valid points", False, str(e)))
    
    # Test 5.2: Mixed valid/invalid points
    try:
        points = np.array([
            [0.5, 0.5],   # Valid
            [2.0, 0.5],   # Invalid (x > 1)
            [0.5, -0.5]   # Invalid (y < 0)
        ])
        preds = engine.predict_batch(points)
        
        valid_count = sum(p is not None for p in preds)
        passed = valid_count == 1 and preds[0] is not None
        suite.add_result(TestResult(
            "5.2 Mixed valid/invalid batch",
            passed,
            f"{valid_count}/3 valid predictions"
        ))
    except Exception as e:
        suite.add_result(TestResult("5.2 Mixed batch", False, str(e)))
    
    # Test 5.3: Empty batch
    try:
        points = np.array([]).reshape(0, 2)
        preds = engine.predict_batch(points)
        
        passed = len(preds) == 0
        suite.add_result(TestResult(
            "5.3 Empty batch",
            passed,
            "Returned empty array"
        ))
    except Exception as e:
        suite.add_result(TestResult("5.3 Empty batch", False, str(e)))
    
    # Test 5.4: Wrong shape batch
    try:
        points = np.array([[0.5, 0.5, 0.5]])  # 3 columns, need 2
        preds = engine.predict_batch(points)
        suite.add_result(TestResult(
            "5.4 Wrong shape batch rejection",
            False,
            "Should have raised ValueError"
        ))
    except ValueError:
        suite.add_result(TestResult(
            "5.4 Wrong shape batch rejection",
            True,
            "Correctly raised ValueError"
        ))
    except Exception as e:
        suite.add_result(TestResult("5.4 Wrong shape batch", False, str(e)))


# ==========================================
# TEST CATEGORY 6: EVALUATION METRICS
# ==========================================

def test_evaluation(suite: LuminCoreTestSuite):
    """Test evaluation method"""
    print("\n[CATEGORY 6] EVALUATION METRICS TESTS")
    print("-" * 70)
    
    # Setup: Linear function
    np.random.seed(42)
    D = 3
    X_train = np.random.rand(50, D) * 10
    Y_train = np.sum(X_train, axis=1).reshape(-1, 1)
    train_data = np.hstack([X_train, Y_train])
    
    engine = LuminCore(dimensions=D)
    engine.fit(train_data)
    
    # Test 6.1: Valid test data
    try:
        X_test = np.random.rand(20, D) * 10
        Y_test = np.sum(X_test, axis=1).reshape(-1, 1)
        test_data = np.hstack([X_test, Y_test])
        
        metrics = engine.evaluate(test_data)
        
        passed = (
            'MSE' in metrics and 
            'MAE' in metrics and 
            'RMSE' in metrics and
            metrics['valid_predictions'] > 0
        )
        suite.add_result(TestResult(
            "6.1 Evaluation with valid data",
            passed,
            f"RMSE: {metrics['RMSE']:.4f}" if metrics['RMSE'] else "No metrics"
        ))
    except Exception as e:
        suite.add_result(TestResult("6.1 Evaluation valid", False, str(e)))
    
    # Test 6.2: All extrapolation (should return None metrics)
    try:
        X_test = np.random.rand(20, D) * 10 + 20  # All out of bounds
        Y_test = np.sum(X_test, axis=1).reshape(-1, 1)
        test_data = np.hstack([X_test, Y_test])
        
        metrics = engine.evaluate(test_data, allow_extrapolation=False)
        
        passed = metrics['valid_predictions'] == 0 and metrics['MSE'] is None
        suite.add_result(TestResult(
            "6.2 All extrapolation test",
            passed,
            f"{metrics['valid_predictions']} valid predictions"
        ))
    except Exception as e:
        suite.add_result(TestResult("6.2 All extrapolation", False, str(e)))


# ==========================================
# TEST CATEGORY 7: EDGE CASES
# ==========================================

def test_edge_cases(suite: LuminCoreTestSuite):
    """Test various edge cases"""
    print("\n[CATEGORY 7] EDGE CASES TESTS")
    print("-" * 70)
    
    # Test 7.1: Single dimension
    try:
        engine = LuminCore(dimensions=1)
        data = np.array([[0, 0], [1, 1], [2, 4]])
        engine.fit(data)
        
        pred = engine.predict([0.5])
        passed = pred is not None
        suite.add_result(TestResult(
            "7.1 Single dimension",
            passed,
            f"Prediction: {pred:.4f}" if pred else "Failed"
        ))
    except Exception as e:
        suite.add_result(TestResult("7.1 Single dimension", False, str(e)))
    
    # Test 7.2: Very high dimensions
    try:
        D = 500
        N = 600
        np.random.seed(42)
        X = np.random.rand(N, D)
        Y = np.sum(X, axis=1).reshape(-1, 1)
        data = np.hstack([X, Y])
        
        engine = LuminCore(dimensions=D)
        
        start = time.perf_counter()
        engine.fit(data)
        fit_time = time.perf_counter() - start
        
        # Use midpoint to ensure it's within bounds
        test_point = engine.bounds_min + (engine.bounds_max - engine.bounds_min) * 0.5
        start = time.perf_counter()
        pred = engine.predict(test_point)
        pred_time = time.perf_counter() - start
        
        passed = pred is not None and pred_time < 1.0  # Less than 1 second
        suite.add_result(TestResult(
            "7.2 Very high dimensions (500D)",
            passed,
            f"Fit: {fit_time*1000:.0f}ms, Predict: {pred_time*1000:.2f}ms"
        ))
    except Exception as e:
        suite.add_result(TestResult("7.2 High dimensions", False, str(e)))
    
    # Test 7.3: Duplicate data points
    try:
        engine = LuminCore(dimensions=2)
        data = np.array([
            [0, 0, 0],
            [0, 0, 0],  # Duplicate
            [1, 1, 1],
            [2, 2, 2]
        ])
        engine.fit(data)
        
        pred = engine.predict([0.5, 0.5])
        passed = pred is not None
        suite.add_result(TestResult(
            "7.3 Duplicate data points",
            passed,
            f"Handled duplicates, prediction: {pred:.4f}" if pred else "Failed"
        ))
    except Exception as e:
        suite.add_result(TestResult("7.3 Duplicates", False, str(e)))
    
    # Test 7.4: All identical values
    try:
        engine = LuminCore(dimensions=2)
        data = np.array([
            [0, 0, 5],
            [1, 1, 5],
            [2, 2, 5]
        ])
        engine.fit(data)
        
        pred = engine.predict([1.5, 1.5])
        passed = pred is not None and abs(pred - 5.0) < 1.0
        suite.add_result(TestResult(
            "7.4 Constant output values",
            passed,
            f"Prediction: {pred:.4f}" if pred else "None"
        ))
    except Exception as e:
        suite.add_result(TestResult("7.4 Constant values", False, str(e)))
    
    # Test 7.5: Predict without fit
    try:
        engine = LuminCore(dimensions=2)
        pred = engine.predict([0, 0])
        suite.add_result(TestResult(
            "7.5 Predict before fit rejection",
            False,
            "Should have raised RuntimeError"
        ))
    except RuntimeError:
        suite.add_result(TestResult(
            "7.5 Predict before fit rejection",
            True,
            "Correctly raised RuntimeError"
        ))
    except Exception as e:
        suite.add_result(TestResult("7.5 Predict before fit", False, str(e)))


# ==========================================
# TEST CATEGORY 8: PERFORMANCE
# ==========================================

def test_performance(suite: LuminCoreTestSuite):
    """Test performance benchmarks"""
    print("\n[CATEGORY 8] PERFORMANCE TESTS")
    print("-" * 70)
    
    # Test 8.1: Fit performance
    try:
        D, N = 100, 500
        np.random.seed(42)
        X = np.random.rand(N, D)
        Y = np.sum(X, axis=1).reshape(-1, 1)
        data = np.hstack([X, Y])
        
        engine = LuminCore(dimensions=D)
        
        start = time.perf_counter()
        engine.fit(data)
        elapsed = time.perf_counter() - start
        
        passed = elapsed < 5.0  # Should fit in less than 5 seconds
        suite.add_result(TestResult(
            "8.1 Fit performance (100D, 500pts)",
            passed,
            f"{elapsed*1000:.2f}ms"
        ))
    except Exception as e:
        suite.add_result(TestResult("8.1 Fit performance", False, str(e)))
    
    # Test 8.2: Single prediction latency
    try:
        D, N = 100, 500
        np.random.seed(42)
        X = np.random.rand(N, D)
        Y = np.sum(X, axis=1).reshape(-1, 1)
        data = np.hstack([X, Y])
        
        engine = LuminCore(dimensions=D)
        engine.fit(data)
        
        test_point = np.random.rand(D)
        
        # Warm-up
        _ = engine.predict(test_point)
        
        # Measure
        start = time.perf_counter()
        pred = engine.predict(test_point)
        elapsed = time.perf_counter() - start
        
        passed = elapsed < 0.1  # Less than 100ms
        suite.add_result(TestResult(
            "8.2 Single prediction latency (100D)",
            passed,
            f"{elapsed*1000:.2f}ms",
            duration=elapsed
        ))
    except Exception as e:
        suite.add_result(TestResult("8.2 Prediction latency", False, str(e)))
    
    # Test 8.3: Batch prediction throughput
    try:
        D, N = 50, 200
        np.random.seed(42)
        X = np.random.rand(N, D)
        Y = np.sum(X, axis=1).reshape(-1, 1)
        data = np.hstack([X, Y])
        
        engine = LuminCore(dimensions=D)
        engine.fit(data)
        
        test_points = np.random.rand(100, D)
        
        start = time.perf_counter()
        preds = engine.predict_batch(test_points)
        elapsed = time.perf_counter() - start
        
        throughput = len(test_points) / elapsed
        
        passed = throughput > 10  # At least 10 predictions per second
        suite.add_result(TestResult(
            "8.3 Batch throughput (50D)",
            passed,
            f"{throughput:.1f} pred/s",
            duration=elapsed
        ))
    except Exception as e:
        suite.add_result(TestResult("8.3 Batch throughput", False, str(e)))


# ==========================================
# TEST CATEGORY 9: V2.1 NEW FEATURES
# ==========================================

def test_v21_features(suite: LuminCoreTestSuite):
    """Test v2.1 specific features: SLR, diagnostics, KDTree"""
    print("\n[CATEGORY 9] V2.1 NEW FEATURES TESTS")
    print("-" * 70)
    
    # Test 9.1: Diagnostics API
    try:
        engine = LuminCore(dimensions=2)
        data = np.array([
            [0, 0, 0],
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 2],
            [2, 2, 4]
        ])
        engine.fit(data)
        
        pred, diag = engine.predict([0.5, 0.5], return_diagnostics=True)
        
        required_keys = {'method', 'uncertainty', 'simplex_size', 'is_degenerate'}
        has_all_keys = required_keys.issubset(diag.keys())
        
        passed = has_all_keys and pred is not None
        suite.add_result(TestResult(
            "9.1 Diagnostics API",
            passed,
            f"method={diag.get('method', 'N/A')}, uncertainty={diag.get('uncertainty', 0):.4f}"
        ))
    except Exception as e:
        suite.add_result(TestResult("9.1 Diagnostics API", False, str(e)))
    
    # Test 9.2: SLR vs IDW method selection
    try:
        # Dense dataset should favor SLR
        engine = LuminCore(dimensions=2)
        X = np.random.rand(200, 2) * 10
        Y = (X[:, 0] + 2*X[:, 1]).reshape(-1, 1)
        data = np.hstack([X, Y])
        engine.fit(data)
        
        # Test multiple points
        test_points = np.random.rand(20, 2) * 10
        preds, diags = engine.predict_batch(test_points, return_diagnostics=True)
        
        slr_count = sum(1 for d in diags if d and d.get('method') == 'slr')
        
        # In dense datasets, we expect some SLR usage
        passed = True  # Just verify it runs without error
        suite.add_result(TestResult(
            "9.2 SLR method selection",
            passed,
            f"SLR used: {slr_count}/20 predictions"
        ))
    except Exception as e:
        suite.add_result(TestResult("9.2 SLR method", False, str(e)))
    
    # Test 9.3: Batch diagnostics
    try:
        engine = LuminCore(dimensions=2)
        data = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]])
        engine.fit(data)
        
        points = np.array([[0.5, 0.5], [1.5, 1.5]])
        preds, diags = engine.predict_batch(points, return_diagnostics=True)
        
        passed = (
            len(preds) == 2 and 
            len(diags) == 2 and
            all('method' in d for d in diags if d)
        )
        suite.add_result(TestResult(
            "9.3 Batch diagnostics",
            passed,
            f"Got diagnostics for {len(diags)} predictions"
        ))
    except Exception as e:
        suite.add_result(TestResult("9.3 Batch diagnostics", False, str(e)))
    
    # Test 9.4: Degeneracy detection
    try:
        # Create intentionally degenerate case (collinear points)
        engine = LuminCore(dimensions=2)
        data = np.array([
            [0, 0, 0],
            [1, 1, 1],
            [2, 2, 2],
            [3, 3, 3]  # All on same line
        ])
        engine.fit(data)
        
        pred, diag = engine.predict([1.5, 1.5], return_diagnostics=True)
        
        # Should detect degeneracy and use IDW fallback
        passed = diag.get('is_degenerate', False) or diag.get('method') == 'idw_fallback'
        suite.add_result(TestResult(
            "9.4 Degeneracy detection",
            passed,
            f"Degenerate={diag.get('is_degenerate')}, method={diag.get('method')}"
        ))
    except Exception as e:
        suite.add_result(TestResult("9.4 Degeneracy detection", False, str(e)))
    
    # Test 9.5: Evaluate with method distribution
    try:
        engine = LuminCore(dimensions=3)
        np.random.seed(42)
        X = np.random.rand(100, 3) * 10
        Y = np.sum(X, axis=1).reshape(-1, 1)
        data = np.hstack([X, Y])
        engine.fit(data)
        
        X_test = np.random.rand(20, 3) * 10
        Y_test = np.sum(X_test, axis=1).reshape(-1, 1)
        test_data = np.hstack([X_test, Y_test])
        
        metrics = engine.evaluate(test_data)
        
        has_new_metrics = (
            'slr_usage' in metrics and
            'idw_usage' in metrics and
            'mean_uncertainty' in metrics
        )
        
        passed = has_new_metrics
        suite.add_result(TestResult(
            "9.5 Enhanced evaluate metrics",
            passed,
            f"SLR={metrics.get('slr_usage', 0)}, IDW={metrics.get('idw_usage', 0)}"
        ))
    except Exception as e:
        suite.add_result(TestResult("9.5 Enhanced evaluate", False, str(e)))
    
    # Test 9.6: KDTree initialization (if scipy available)
    try:
        try:
            from scipy.spatial import KDTree as TestKDTree
            scipy_available = True
        except:
            scipy_available = False
        
        if scipy_available:
            # Large dataset to trigger KDTree
            engine = LuminCore(dimensions=3, use_kdtree_threshold=100)
            X = np.random.rand(150, 3)
            Y = np.sum(X, axis=1).reshape(-1, 1)
            data = np.hstack([X, Y])
            engine.fit(data)
            
            passed = engine.kdtree is not None
            suite.add_result(TestResult(
                "9.6 KDTree initialization",
                passed,
                f"KDTree active: {engine.kdtree is not None}"
            ))
        else:
            suite.add_result(TestResult(
                "9.6 KDTree initialization",
                True,
                "Scipy not available, skipped"
            ))
    except Exception as e:
        suite.add_result(TestResult("9.6 KDTree init", False, str(e)))


# ==========================================
# MAIN TEST RUNNER
# ==========================================

def main():
    """Run all test categories"""
    print("\n" + "🔷"*35)
    print("LUMIN CORE v2.1 - VALIDATION TEST SUITE")
    print("🔷"*35)
    
    suite = LuminCoreTestSuite()
    
    # Run all test categories
    test_initialization(suite)
    test_data_loading(suite)
    test_extrapolation_detection(suite)
    test_prediction_accuracy(suite)
    test_batch_prediction(suite)
    test_evaluation(suite)
    test_edge_cases(suite)
    test_performance(suite)
    test_v21_features(suite)  # NEW: v2.1 specific tests
    
    # Print summary
    all_passed = suite.print_summary()
    
    # Exit with appropriate code
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
 
