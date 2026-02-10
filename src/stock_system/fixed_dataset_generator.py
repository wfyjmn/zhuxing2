"""
固定数据集生成器
使用固定随机种子生成确定性数据集，用于回归测试
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import train_test_split
import os
import json
import warnings
warnings.filterwarnings('ignore')


class FixedDatasetGenerator:
    """
    固定数据集生成器
    
    使用固定随机种子生成确定性数据集，确保每次运行得到相同的数据
    用于回归测试和模型性能监控
    """
    
    def __init__(self, seed: int = 42):
        """
        初始化固定数据集生成器
        
        参数:
            seed: 随机种子
        """
        self.seed = seed
        np.random.seed(seed)
    
    def generate_regression_test_dataset(
        self,
        n_samples: int = 5000,
        n_features: int = 20,
        positive_ratio: float = 0.1,
        signal_strength: float = 0.7
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        生成回归测试数据集
        
        参数:
            n_samples: 样本数
            n_features: 特征数
            positive_ratio: 正样本比例
            signal_strength: 信号强度（0-1，越高越容易预测）
            
        返回:
            (特征DataFrame, 标签数组)
        """
        np.random.seed(self.seed)
        
        # 生成特征
        X = np.random.randn(n_samples, n_features)
        
        # 生成标签
        y = np.random.choice([0, 1], size=n_samples, p=[1 - positive_ratio, positive_ratio])
        
        # 为正样本添加信号
        # 前5个特征与标签相关
        n_signal_features = int(n_features * signal_strength)
        for i in range(n_signal_features):
            X[y == 1, i] += np.random.normal(2.0, 0.5, y.sum())
            X[y == 0, i] += np.random.normal(0.0, 0.5, n_samples - y.sum())
        
        # 创建DataFrame
        feature_names = [f'feature_{i}' for i in range(n_features)]
        df = pd.DataFrame(X, columns=feature_names)
        
        return df, y
    
    def generate_time_series_dataset(
        self,
        n_samples: int = 1000,
        n_features: int = 15,
        positive_ratio: float = 0.08,
        trend: float = 0.001,
        noise_level: float = 0.02
    ) -> Tuple[pd.DataFrame, np.ndarray, pd.DatetimeIndex]:
        """
        生成时间序列数据集（模拟股票数据）
        
        参数:
            n_samples: 样本数
            n_features: 特征数
            positive_ratio: 正样本比例
            trend: 趋势强度
            noise_level: 噪声水平
            
        返回:
            (特征DataFrame, 标签数组, 日期索引)
        """
        np.random.seed(self.seed)
        
        # 生成日期
        dates = pd.date_range(start='2020-01-01', periods=n_samples, freq='D')
        
        # 生成基础价格序列（带趋势和噪声）
        returns = np.random.normal(trend, noise_level, n_samples)
        prices = 100 * np.cumprod(1 + returns)
        
        # 生成特征
        X = np.random.randn(n_samples, n_features)
        
        # 添加技术特征
        X[:, 0] = np.diff(prices, prepend=prices[0]) / prices[:-1]  # 收益率
        X[:, 1] = pd.Series(prices).rolling(5).mean().fillna(prices[0]).values  # 5日均线
        X[:, 2] = pd.Series(prices).rolling(20).mean().fillna(prices[0]).values  # 20日均线
        X[:, 3] = X[:, 0] * X[:, 1]  # 动量
        X[:, 4] = np.random.randn(n_samples) * 0.1  # 市场情绪
        
        # 生成标签（未来5天涨幅>3%）
        future_returns = np.zeros(n_samples)
        for i in range(n_samples - 5):
            future_returns[i] = (prices[i + 5] - prices[i]) / prices[i]
        
        # 填充最后5个
        future_returns[-5:] = future_returns[-6]
        
        y = (future_returns > 0.03).astype(int)
        
        # 调整正样本比例
        current_ratio = y.mean()
        if current_ratio > positive_ratio:
            # 随机将部分正样本改为负样本
            positive_indices = np.where(y == 1)[0]
            n_to_flip = int((current_ratio - positive_ratio) * n_samples)
            flip_indices = np.random.choice(positive_indices, n_to_flip, replace=False)
            y[flip_indices] = 0
        elif current_ratio < positive_ratio:
            # 随机将部分负样本改为正样本
            negative_indices = np.where(y == 0)[0]
            n_to_flip = int((positive_ratio - current_ratio) * n_samples)
            flip_indices = np.random.choice(negative_indices, n_to_flip, replace=False)
            y[flip_indices] = 1
        
        # 创建DataFrame
        feature_names = [f'feature_{i}' for i in range(n_features)]
        df = pd.DataFrame(X, columns=feature_names)
        df.index = dates
        
        return df, y, dates
    
    def save_dataset(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        save_dir: str = 'assets/fixed_datasets',
        dataset_name: str = 'regression_test'
    ):
        """
        保存数据集
        
        参数:
            X: 特征DataFrame
            y: 标签数组
            save_dir: 保存目录
            dataset_name: 数据集名称
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存特征
        X_path = os.path.join(save_dir, f'{dataset_name}_features.csv')
        X.to_csv(X_path, index=True)
        
        # 保存标签
        y_path = os.path.join(save_dir, f'{dataset_name}_labels.npy')
        np.save(y_path, y)
        
        # 保存元数据
        metadata = {
            'n_samples': len(X),
            'n_features': X.shape[1],
            'n_positive': int(y.sum()),
            'positive_ratio': float(y.mean()),
            'seed': self.seed
        }
        
        metadata_path = os.path.join(save_dir, f'{dataset_name}_metadata.json')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✓ 数据集已保存到 {save_dir}")
        print(f"  - 特征: {X_path}")
        print(f"  - 标签: {y_path}")
        print(f"  - 元数据: {metadata_path}")
    
    def load_dataset(
        self,
        save_dir: str = 'assets/fixed_datasets',
        dataset_name: str = 'regression_test'
    ) -> Tuple[pd.DataFrame, np.ndarray, Dict]:
        """
        加载数据集
        
        参数:
            save_dir: 保存目录
            dataset_name: 数据集名称
            
        返回:
            (特征DataFrame, 标签数组, 元数据字典)
        """
        # 加载特征
        X_path = os.path.join(save_dir, f'{dataset_name}_features.csv')
        X = pd.read_csv(X_path, index_col=0)
        
        # 加载标签
        y_path = os.path.join(save_dir, f'{dataset_name}_labels.npy')
        y = np.load(y_path)
        
        # 加载元数据
        metadata_path = os.path.join(save_dir, f'{dataset_name}_metadata.json')
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        return X, y, metadata


def create_all_fixed_datasets():
    """创建所有固定数据集"""
    print("=" * 70)
    print("创建固定数据集")
    print("=" * 70)
    print()
    
    # 回归测试数据集
    print("【1】创建回归测试数据集")
    print("-" * 70)
    generator = FixedDatasetGenerator(seed=42)
    X, y = generator.generate_regression_test_dataset(
        n_samples=5000,
        n_features=20,
        positive_ratio=0.1,
        signal_strength=0.7
    )
    generator.save_dataset(X, y, dataset_name='regression_test')
    print()
    
    # 时间序列数据集
    print("【2】创建时间序列数据集")
    print("-" * 70)
    generator = FixedDatasetGenerator(seed=42)
    X_ts, y_ts, dates = generator.generate_time_series_dataset(
        n_samples=1000,
        n_features=15,
        positive_ratio=0.08
    )
    generator.save_dataset(X_ts, y_ts, dataset_name='timeseries_test')
    print()
    
    # 高难度数据集（用于压力测试）
    print("【3】创建高难度数据集")
    print("-" * 70)
    generator = FixedDatasetGenerator(seed=42)
    X_hard, y_hard = generator.generate_regression_test_dataset(
        n_samples=3000,
        n_features=30,
        positive_ratio=0.05,
        signal_strength=0.4
    )
    generator.save_dataset(X_hard, y_hard, dataset_name='hard_test')
    print()
    
    print("=" * 70)
    print("✓ 所有固定数据集创建完成")
    print("=" * 70)


if __name__ == "__main__":
    create_all_fixed_datasets()
