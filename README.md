# SLRM Lumin Core v2.1

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![NumPy](https://img.shields.io/badge/numpy-required-orange.svg)](https://numpy.org/)

**High-Dimensional Simplex Local Regression with Automatic Extrapolation Detection**

SLRM (Simplex Localized Regression Model) Lumin Core is a lightweight, efficient interpolation engine designed for high-dimensional spaces. It uses geometric simplex encapsulation with local linear regression to provide fast, accurate predictions while automatically detecting and rejecting extrapolation attempts.

---

## 🌟 Key Features

- ✅ **No Extrapolation by Design**: Automatically detects and rejects queries outside dataset bounds (sacred boundaries, no epsilon tolerance)
- 🎯 **Simplex Local Regression (SLR)**: Fits local hyperplanes through simplex nodes for improved accuracy on smooth functions
- 🔄 **Intelligent Fallback**: Automatic degeneracy detection with IDW fallback for stability
- 📊 **Diagnostic Information**: Optional uncertainty metrics, method selection, and quality indicators
- 🚀 **High-Dimensional Optimization**: Vectorized O(D) complexity for real-time performance
- ⚡ **KD-Tree Acceleration**: Automatic activation for datasets with N > 10,000 points
- 🧪 **Comprehensive Test Suite**: 39 validation tests covering edge cases and new features
- 💾 **Batch Processing**: Efficient multi-point prediction with diagnostic support

---

## 📦 Installation

### Requirements
- Python 3.8+
- NumPy
- SciPy (optional, for KD-Tree acceleration with large datasets)

### Quick Install

```bash
# Clone the repository
git clone https://github.com/wexionar/slrm-lumin-core.git
cd slrm-lumin-core

# Install dependencies
pip install numpy

# Optional: Install scipy for KD-Tree support
pip install scipy
```

---

## 🚀 Quick Start

```python
import numpy as np
from lumin_core import LuminCore

# 1. Create engine for 3D space
engine = LuminCore(dimensions=3)

# 2. Prepare training data (N points, D+1 columns: [x1, x2, ..., xD, y])
X = np.random.rand(100, 3) * 10  # 100 points in 3D space
Y = np.sum(X**2, axis=1).reshape(-1, 1)  # y = x1² + x2² + x3²
data = np.hstack([X, Y])

# 3. Fit the model
engine.fit(data)

# 4. Make predictions
query_point = np.array([5.0, 5.0, 5.0])
prediction = engine.predict(query_point)

if prediction is not None:
    print(f"Prediction: {prediction:.4f}")
else:
    print("Query point is outside dataset bounds (extrapolation rejected)")
```

---

## 📖 Core Concepts

### Simplex Encapsulation

For each dimension `i`, Lumin Core finds:
1. **Inferior boundary**: Nearest point where `X[i] ≤ query[i]`
2. **Superior boundary**: Nearest point where `X[i] ≥ query[i]`
3. Selects the closest of the two

The result is a minimal geometric simplex that **encapsulates** the query point.

### Extrapolation Detection

**Lumin Core does NOT perform extrapolation.** Before prediction:
- Checks if query point is within `[min, max]` bounds for each dimension
- Returns `None` if any dimension is out of bounds
- Provides clear diagnostic messages indicating which axes are violated

### Interpolation Method

Uses **Inverse Distance Weighting (IDW)**:
```
w_i = 1 / distance_i
prediction = Σ(w_i × y_i) / Σ(w_i)
```

---

## 🔧 API Reference

### `LuminCore(dimensions: int)`

Initialize the engine.

**Parameters:**
- `dimensions`: Number of input dimensions (D)

**Example:**
```python
engine = LuminCore(dimensions=10)
```

---

### `fit(data: np.ndarray) -> None`

Load and validate the dataset.

**Parameters:**
- `data`: Array of shape `(N, D+1)` where:
  - First D columns are features
  - Last column is the target value

**Raises:**
- `ValueError`: If data has wrong shape or insufficient points

**Example:**
```python
data = np.array([
    [0, 0, 0],  # [x1, x2, y]
    [1, 1, 1],
    [2, 2, 4]
])
engine.fit(data)
```

---

### `predict(point: np.ndarray, allow_extrapolation: bool = False, return_diagnostics: bool = False)`

Predict value at a single query point.

**Parameters:**
- `point`: Query point of shape `(D,)`
- `allow_extrapolation`: If `False` (default), returns `None` for out-of-bounds points
- `return_diagnostics`: If `True`, returns `(prediction, diagnostics_dict)` **[NEW in v2.1]**

**Returns:**
- If `return_diagnostics=False`: Predicted value or `None`
- If `return_diagnostics=True`: `(prediction, diagnostics)` where diagnostics contains:
  - `'method'`: `'slr'` | `'idw_fallback'` - which method was used
  - `'uncertainty'`: Mean distance to simplex nodes (lower is better)
  - `'simplex_size'`: Number of nodes in the simplex
  - `'is_degenerate'`: Whether simplex was degenerate

**Example:**
```python
# Simple prediction
pred = engine.predict([0.5, 0.5])

# With diagnostics (v2.1)
pred, diag = engine.predict([0.5, 0.5], return_diagnostics=True)
print(f"Method: {diag['method']}, Uncertainty: {diag['uncertainty']:.4f}")
```

---

### `predict_batch(points: np.ndarray, allow_extrapolation: bool = False) -> np.ndarray`

Predict values for multiple query points.

**Parameters:**
- `points`: Array of shape `(M, D)` with M query points
- `allow_extrapolation`: Same as `predict()`

**Returns:**
- Array of shape `(M,)` with predictions (`None` for rejected points)

**Example:**
```python
queries = np.array([[0.5, 0.5], [1.0, 1.0], [5.0, 5.0]])
predictions = engine.predict_batch(queries)
```

---

### `evaluate(test_data: np.ndarray, allow_extrapolation: bool = False) -> dict`

Evaluate model performance on test data.

**Parameters:**
- `test_data`: Array of shape `(M, D+1)` (same format as training data)
- `allow_extrapolation`: Same as `predict()`

**Returns:**
Dictionary with metrics:
```python
{
    'MSE': float,           # Mean Squared Error
    'MAE': float,           # Mean Absolute Error
    'RMSE': float,          # Root Mean Squared Error
    'valid_predictions': int,  # Number of valid (non-extrapolated) predictions
    'total_points': int,    # Total test points
    'slr_usage': int,       # Number of predictions using SLR [NEW in v2.1]
    'idw_usage': int,       # Number of predictions using IDW fallback [NEW in v2.1]
    'mean_uncertainty': float  # Average uncertainty across predictions [NEW in v2.1]
}
```

**Example:**
```python
metrics = engine.evaluate(test_data)
print(f"RMSE: {metrics['RMSE']:.4f}")
print(f"SLR used: {metrics['slr_usage']}/{metrics['valid_predictions']}")
print(f"Mean uncertainty: {metrics['mean_uncertainty']:.4f}")
```

---

## 📊 Performance Benchmarks

Tested on various dimensionalities (CPU: typical modern processor):

| Dimensions | Points | Fit Time | Prediction Time | Throughput |
|------------|--------|----------|-----------------|------------|
| 10D        | 100    | 0.1 ms   | 0.05 ms         | 20,000/s   |
| 50D        | 200    | 0.5 ms   | 0.5 ms          | 3,600/s    |
| 100D       | 500    | 1.2 ms   | 1.3 ms          | 770/s      |
| 500D       | 1000   | 6 ms     | 14 ms           | 70/s       |
| 1000D      | 2000   | 15 ms    | 25 ms           | 40/s       |

*Benchmarks from automated test suite on commodity hardware.*

---

## 🧪 Testing

Run the comprehensive test suite:

```bash
python lumin_core_test.py
```

**Test Coverage:**
- ✅ 39 tests across 9 categories
- ✅ Initialization & data loading
- ✅ Extrapolation detection
- ✅ Prediction accuracy
- ✅ Batch processing
- ✅ Edge cases (1D, 500D, duplicates, etc.)
- ✅ Performance benchmarks
- ✅ **NEW**: SLR vs IDW method selection
- ✅ **NEW**: Diagnostic API validation
- ✅ **NEW**: Degeneracy detection
- ✅ **NEW**: KD-Tree initialization

Expected output:
```
🔷🔷🔷🔷🔷🔷🔷🔷🔷🔷🔷🔷🔷🔷🔷🔷🔷🔷🔷🔷
LUMIN CORE v2.1 - VALIDATION TEST SUITE
🔷🔷🔷🔷🔷🔷🔷🔷🔷🔷🔷🔷🔷🔷🔷🔷🔷🔷🔷🔷

...

======================================================================
TOTAL: 39/39 tests passed
🎉 ALL TESTS PASSED! 🎉
======================================================================
```

---

## 📚 Examples

### Example 1: Linear Function (1D)

```python
import numpy as np
from lumin_core import LuminCore

# Create dataset: y = 2x + 3
X = np.linspace(0, 10, 50).reshape(-1, 1)
Y = (2 * X + 3).reshape(-1, 1)
data = np.hstack([X, Y])

engine = LuminCore(dimensions=1)
engine.fit(data)

# Predict
pred = engine.predict([5.0])
print(f"y = 2(5) + 3 = 13, Predicted: {pred:.4f}")
```

### Example 2: High-Dimensional Sum of Squares

```python
import numpy as np
from lumin_core import LuminCore

# 100D space: y = Σ(x_i²)
D = 100
N = 500

X_train = np.random.rand(N, D) * 10
Y_train = np.sum(X_train**2, axis=1).reshape(-1, 1)
train_data = np.hstack([X_train, Y_train])

engine = LuminCore(dimensions=D)
engine.fit(train_data)

# Test
X_test = np.random.rand(20, D) * 10
Y_test = np.sum(X_test**2, axis=1).reshape(-1, 1)
test_data = np.hstack([X_test, Y_test])

metrics = engine.evaluate(test_data)
print(f"RMSE: {metrics['RMSE']:.2f}")
print(f"Valid predictions: {metrics['valid_predictions']}/20")
```

### Example 3: Using Diagnostics (v2.1 Feature)

```python
import numpy as np
from lumin_core import LuminCore

# Dataset
data = np.array([
    [0, 0, 0],
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 2]
])

engine = LuminCore(dimensions=2)
engine.fit(data)

# Predict with diagnostics
pred, diag = engine.predict([0.5, 0.5], return_diagnostics=True)

print(f"Prediction: {pred:.4f}")
print(f"Method used: {diag['method']}")  # 'slr' or 'idw_fallback'
print(f"Uncertainty: {diag['uncertainty']:.4f}")
print(f"Simplex size: {diag['simplex_size']}")
print(f"Is degenerate: {diag['is_degenerate']}")
```

### Example 4: Handling Extrapolation

```python
import numpy as np
from lumin_core import LuminCore

# Dataset bounded in [0, 1] × [0, 1]
data = np.array([
    [0, 0, 0],
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 2]
])

engine = LuminCore(dimensions=2)
engine.fit(data)

# Test points
print("Interior point [0.5, 0.5]:")
print(f"  → {engine.predict([0.5, 0.5])}")

print("\nExtrapolation [2.0, 0.5]:")
pred = engine.predict([2.0, 0.5])
print(f"  → {pred}")  # Returns None

print("\nForce extrapolation:")
pred_forced = engine.predict([2.0, 0.5], allow_extrapolation=True)
print(f"  → {pred_forced}")  # Returns value (use with caution!)
```

---

## 🎯 Use Cases

### ✅ Ideal For:
- **High-dimensional regression** (physics simulations, financial modeling)
- **Interpolation with strict no-extrapolation requirements**
- **Real-time systems** needing fast predictions (< 50ms)
- **Sparse datasets** in high-dimensional spaces
- **Data validation** (detecting out-of-distribution queries)

### ⚠️ Not Recommended For:
- **Smooth function approximation** (use neural networks or splines)
- **Datasets requiring extrapolation** (by design, Lumin Core rejects this)
- **Very large datasets** (> 100k points; becomes memory-intensive)
- **Low-dimensional, dense grids** (traditional methods may be faster)

---

## 🧮 Mathematical Foundation

### Simplex Selection Algorithm

For a query point `q` in D-dimensional space:

1. **For each dimension `i = 1..D`:**
   ```
   candidates_inf = {p ∈ dataset : p[i] ≤ q[i]}
   candidates_sup = {p ∈ dataset : p[i] ≥ q[i]}
   
   if candidates_inf ≠ ∅:
       node_inf[i] = argmin_{p ∈ candidates_inf} |p[i] - q[i]|
   
   if candidates_sup ≠ ∅:
       node_sup[i] = argmin_{p ∈ candidates_sup} |p[i] - q[i]|
   
   simplex_node[i] = closer(node_inf[i], node_sup[i])
   ```

2. **Add global anchor:**
   ```
   anchor = argmin_{p ∈ dataset} ||p - q||₂
   simplex = {simplex_node[1], ..., simplex_node[D], anchor}
   ```

3. **Remove duplicates:**
   ```
   simplex = unique(simplex)
   ```

### Simplex Local Regression (NEW in v2.1)

Given simplex nodes `{(x₁, y₁), ..., (xₖ, yₖ)}`, fit a local hyperplane:

```
y = β₀ + β₁·x₁ + β₂·x₂ + ... + βₐ·xₐ
```

**Method:**
1. Construct augmented matrix: `A = [1  x₁₁  x₁₂  ...  x₁ₐ]`
                               `    [1  x₂₁  x₂₂  ...  x₂ₐ]`
                               `    [⋮   ⋮    ⋮   ⋱    ⋮  ]`

2. Solve least squares: `β = (AᵀA)⁻¹Aᵀy`

3. Predict: `ŷ = β₀ + β₁·q₁ + ... + βₐ·qₐ`

**Degeneracy Detection:**
If nodes are nearly collinear (det(A) < 10⁻¹⁰), fallback to IDW.

### Inverse Distance Weighting (Fallback)

Used when simplex is degenerate:

```
dᵢ = ||xᵢ - q||₂
wᵢ = 1/dᵢ
ŷ = Σ(wᵢ × yᵢ) / Σ(wᵢ)
```

**Special case**: If `dᵢ < ε` (exact match), return `yᵢ` directly.

---

## 🔬 Technical Details

### Complexity Analysis

- **Fit:** O(N) where N is dataset size
- **Predict:** O(D·N) where D is dimensions, N is dataset size
  - Simplex search: O(D·N)
  - IDW computation: O(D)
- **Memory:** O(N·D)

### Numerical Stability

- NaN rows automatically removed during `fit()`
- Division-by-zero protection (minimum distance threshold: 1e-12)
- Exact node matches return direct value (avoids 0/0)
- Pre-computed bounds for fast extrapolation checks

---

## 🛠️ Advanced Configuration

### Custom Distance Metrics

While the current implementation uses Euclidean distance, you can modify `_build_simplex()` for custom metrics:

```python
# Example: Manhattan distance
def _build_simplex_manhattan(self, point):
    # Modify line 79 in lumin_core.py:
    # Change: global_dists = np.linalg.norm(self.X - point, axis=1)
    # To:     global_dists = np.sum(np.abs(self.X - point), axis=1)
    ...
```

### Weight Customization

Modify the IDW formula (line 82-84 in `lumin_core.py`):

```python
# Current: Inverse distance weighting
weights = 1.0 / distances

# Alternative: Inverse squared distance
weights = 1.0 / (distances ** 2)

# Alternative: Gaussian kernel
weights = np.exp(-distances**2 / (2 * sigma**2))
```

---

## 🐛 Troubleshooting

### Issue: `ValueError: Need at least D+1 valid points`
**Solution:** Your dataset has too few points after NaN removal. Ensure you have at least `D+1` valid rows.

### Issue: All predictions return `None`
**Solution:** Your query points are outside the dataset bounds. Check bounds with:
```python
print(f"Min bounds: {engine.bounds_min}")
print(f"Max bounds: {engine.bounds_max}")
```

### Issue: High prediction error
**Causes:**
1. Sparse dataset in high dimensions (curse of dimensionality)
2. Non-linear function (IDW assumes local linearity)
3. Query point far from training data

**Solutions:**
- Add more training data
- Use domain-specific feature engineering
- Consider non-linear methods (neural networks, kernel methods)

---

## 📜 Version History

### v2.1 (Current)
- ✅ **Simplex Local Regression (SLR)**: Replaced IDW with local hyperplane fitting
- ✅ **Intelligent Fallback**: Automatic degeneracy detection with IDW backup
- ✅ **Diagnostic API**: Optional uncertainty, method, and quality metrics
- ✅ **KD-Tree Acceleration**: Automatic for N > 10,000 (requires scipy)
- ✅ **Enhanced Evaluate**: Method distribution and uncertainty reporting
- ✅ **Sacred Boundaries**: Zero epsilon tolerance (no extrapolation penumbra)
- ✅ **Extended Test Suite**: 39 comprehensive tests (was 33)

### v2.0
- ✅ Complete rewrite with optimized architecture
- ✅ Automatic extrapolation detection
- ✅ Batch prediction support
- ✅ Built-in evaluation metrics
- ✅ Comprehensive test suite (33 tests)
- ✅ Full API documentation

### v1.x (Legacy)
- Basic simplex interpolation
- IDW and gradient-based variants
- Manual extrapolation handling

---

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

**Development Setup:**
```bash
git clone https://github.com/wexionar/slrm-lumin-core.git
cd slrm-lumin-core
pip install numpy
python lumin_core_test.py  # Ensure all tests pass
```

---

## 🧠 SLRM Team

Alex · Gemini · ChatGPT   Claude · Grok · Meta AI  

---

## 📄 License

MIT
 
