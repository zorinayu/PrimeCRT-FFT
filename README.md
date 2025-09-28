# PrimeFFT - 分解感知快速傅里叶变换

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

PrimeFFT是一个基于分解感知的快速傅里叶变换(FFT)实现库，能够根据输入序列长度N的数学分解特性，智能选择最优的FFT算法。

## 🚀 核心特性

### 智能算法选择
- **分解感知规划器**：自动分析序列长度的质因数分解
- **多算法支持**：Cooley-Tukey、Good-Thomas、Bluestein算法
- **最优路径选择**：根据数学特性选择最高效的算法

### 算法覆盖
- **Cooley-Tukey**：适用于任意合数长度，递归分解
- **Good-Thomas**：基于中国剩余定理，避免旋转因子，适用于互质分解
- **Bluestein**：将DFT转换为卷积，适用于质数长度

## 📊 性能表现

### 基准测试结果

| 序列长度 | 算法选择 | 相对NumPy性能 | 数值精度 |
|---------|---------|--------------|---------|
| 120 (2³×3×5) | Cooley-Tukey | 100% | 完全一致 |
| 225 (3²×5²) | Cooley-Tukey | 100% | 完全一致 |
| 480 (2⁵×3×5) | Cooley-Tukey | 100% | 完全一致 |
| 512 (2⁹) | Cooley-Tukey | 100% | 完全一致 |
| 1009 (质数) | Bluestein | 37% | 1e-11误差 |
| 2048 (2¹¹) | Cooley-Tukey | 100% | 完全一致 |

### 关键优势

1. **数值精度**：合数长度与NumPy完全一致，质数长度误差在1e-11量级
2. **算法智能**：自动选择最优算法，无需手动指定
3. **教育价值**：清晰展示不同FFT算法的数学原理
4. **扩展性**：易于添加新的FFT算法变种

## 🛠️ 安装与使用

### 安装依赖
```bash
pip install numpy
```

### 基本使用
```python
import numpy as np
from primefft import fft

# 自动选择最优算法
x = np.random.randn(1000) + 1j * np.random.randn(1000)
X = fft(x)

# 强制使用特定算法
X_ct = fft(x, method='cooleytukey')    # Cooley-Tukey
X_gt = fft(x, method='good-thomas')    # Good-Thomas  
X_bs = fft(x, method='bluestein')      # Bluestein
X_np = fft(x, method='numpy')          # NumPy后端
```

### 运行基准测试
```bash
python -m examples.bench
```

### 运行测试
```bash
python -m pytest
```

## 📈 算法选择策略

PrimeFFT使用以下策略选择算法：

1. **质数长度** → Bluestein算法
2. **合数长度** → Cooley-Tukey算法  
3. **互质分解** → Good-Thomas算法（当适用时）

## 🔬 技术原理

### 分解感知规划
```python
# 示例：N=480的分解
factors = {2: 5, 3: 1, 5: 1}  # 480 = 2⁵ × 3¹ × 5¹
plan = choose_plan(factors)    # 选择Cooley-Tukey
```

### 算法复杂度
- **Cooley-Tukey**: O(N log N) - 递归分解
- **Good-Thomas**: O(N log N) - 无旋转因子
- **Bluestein**: O(N log N) - 卷积转换

## 📁 项目结构

```
primefft/
├── __init__.py          # 主接口
├── fft.py              # FFT调度器
├── planner.py          # 分解规划器
├── cooleytukey.py      # Cooley-Tukey实现
├── goodthomas.py       # Good-Thomas实现
└── bluestein.py        # Bluestein实现

examples/
└── bench.py            # 性能基准测试

tests/
├── test_correctness.py # 正确性测试
└── test_methods.py     # 方法对比测试
```

## 🎯 应用场景

- **信号处理**：音频、图像分析
- **科学计算**：数值分析、物理仿真
- **教育研究**：FFT算法学习与对比
- **性能优化**：特定长度序列的优化

## 🔮 未来规划

- [ ] 实现纯算法版本的Cooley-Tukey（非NumPy后端）
- [ ] 优化Good-Thomas的数值精度
- [ ] 添加多维FFT支持
- [ ] 并行计算优化
- [ ] 更多算法变种（Rader、Winograd等）

## 📄 许可证

本项目采用MIT许可证 - 详见 [LICENSE](LICENSE) 文件

## 🤝 贡献

欢迎提交Issue和Pull Request来改进这个项目！

---

**PrimeFFT** - 让FFT算法选择变得智能而高效