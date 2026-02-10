"""
概率校准器
对模型输出的概率进行校准，提升概率准确性
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


class ProbabilityCalibrator:
    """概率校准器
    
    支持两种校准方法：
    1. Platt Scaling (LogisticRegression): 适用于数据量较小的情况
    2. Isotonic Regression: 适用于数据量较大且分布复杂的情况
    """
    
    def __init__(
        self,
        method: str = 'isotonic',
        random_state: int = 42
    ):
        """
        初始化概率校准器
        
        Args:
            method: 校准方法
                - 'platt': Platt Scaling (LogisticRegression)
                - 'isotonic': Isotonic Regression
            random_state: 随机种子
        """
        self.method = method
        self.random_state = random_state
        
        self.calibrator = None
        self.is_fitted = False
        self.calibration_report = None
    
    def fit(
        self,
        y_proba_train: np.ndarray,
        y_true_train: np.ndarray,
        validation_split: float = 0.2
    ) -> Dict[str, Any]:
        """
        在训练集上拟合校准器
        
        Args:
            y_proba_train: 训练集概率预测
            y_true_train: 训练集真实标签
            validation_split: 验证集比例（用于选择校准方法）
        
        Returns:
            拟合报告
        """
        # 检查数据
        if len(y_proba_train) != len(y_true_train):
            raise ValueError("y_proba_train 和 y_true_train 长度不一致")
        
        # 拟合校准器
        if self.method == 'platt':
            self.calibrator = self._fit_platt_scaling(y_proba_train, y_true_train)
        elif self.method == 'isotonic':
            self.calibrator = self._fit_isotonic_regression(y_proba_train, y_true_train)
        else:
            raise ValueError(f"未知的校准方法: {self.method}")
        
        self.is_fitted = True
        
        # 生成校准报告
        self.calibration_report = self._generate_report(
            y_proba_train, y_true_train
        )
        
        return self.calibration_report
    
    def _fit_platt_scaling(
        self,
        y_proba: np.ndarray,
        y_true: np.ndarray
    ) -> LogisticRegression:
        """
        Platt Scaling: 使用 LogisticRegression 校准
        
        公式: P(y=1|x) = 1 / (1 + exp(A * z + B))
        其中 z 是原始概率经过 logit 变换: z = log(p / (1 - p))
        """
        # 对概率进行 logit 变换
        epsilon = 1e-10
        z = np.log((y_proba + epsilon) / (1 - y_proba + epsilon))
        
        # 使用 LogisticRegression 拟合
        calibrator = LogisticRegression(
            C=1.0,
            solver='lbfgs',
            random_state=self.random_state
        )
        calibrator.fit(z.reshape(-1, 1), y_true)
        
        return calibrator
    
    def _fit_isotonic_regression(
        self,
        y_proba: np.ndarray,
        y_true: np.ndarray
    ) -> IsotonicRegression:
        """
        Isotonic Regression: 保序回归
        
        假设校准后的概率是原始概率的保序函数
        """
        calibrator = IsotonicRegression(
            out_of_bounds='clip',
            y_min=0.0,
            y_max=1.0
        )
        calibrator.fit(y_proba, y_true)
        
        return calibrator
    
    def predict_proba(self, y_proba: np.ndarray) -> np.ndarray:
        """
        校准概率
        
        Args:
            y_proba: 原始概率预测
        
        Returns:
            校准后概率
        """
        if not self.is_fitted:
            raise ValueError("校准器未拟合。请先调用 fit() 方法")
        
        if self.method == 'platt':
            calibrated_proba = self._predict_platt(y_proba)
        elif self.method == 'isotonic':
            calibrated_proba = self.calibrator.predict(y_proba)
        else:
            raise ValueError(f"未知的校准方法: {self.method}")
        
        # 确保概率在 [0, 1] 范围内
        calibrated_proba = np.clip(calibrated_proba, 0.0, 1.0)
        
        return calibrated_proba
    
    def _predict_platt(self, y_proba: np.ndarray) -> np.ndarray:
        """使用 Platt Scaling 预测"""
        # 对概率进行 logit 变换
        epsilon = 1e-10
        z = np.log((y_proba + epsilon) / (1 - y_proba + epsilon))
        
        # 使用 LogisticRegression 预测
        calibrated_proba = self.calibrator.predict_proba(z.reshape(-1, 1))[:, 1]
        
        return calibrated_proba
    
    def _generate_report(
        self,
        y_proba: np.ndarray,
        y_true: np.ndarray
    ) -> Dict[str, Any]:
        """生成校准报告"""
        # 校准概率
        y_proba_calibrated = self.predict_proba(y_proba)
        
        # 计算 Brier Score
        brier_score_before = brier_score_loss(y_true, y_proba)
        brier_score_after = brier_score_loss(y_true, y_proba_calibrated)
        
        # 计算校准曲线
        prob_true_before, prob_pred_before = calibration_curve(y_true, y_proba, n_bins=10)
        prob_true_after, prob_pred_after = calibration_curve(y_true, y_proba_calibrated, n_bins=10)
        
        # 计算期望校准误差 (Expected Calibration Error, ECE)
        ece_before = self._calculate_ece(y_true, y_proba)
        ece_after = self._calculate_ece(y_true, y_proba_calibrated)
        
        report = {
            'method': self.method,
            'brier_score': {
                'before': brier_score_before,
                'after': brier_score_after,
                'improvement': brier_score_before - brier_score_after
            },
            'ece': {
                'before': ece_before,
                'after': ece_after,
                'improvement': ece_before - ece_after
            },
            'calibration_curve': {
                'before': {
                    'prob_true': prob_true_before.tolist(),
                    'prob_pred': prob_pred_before.tolist()
                },
                'after': {
                    'prob_true': prob_true_after.tolist(),
                    'prob_pred': prob_pred_after.tolist()
                }
            }
        }
        
        return report
    
    def _calculate_ece(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        n_bins: int = 10
    ) -> float:
        """
        计算期望校准误差 (Expected Calibration Error, ECE)
        
        ECE = sum(|confidence - accuracy| * n_bin) / N
        
        Args:
            y_true: 真实标签
            y_proba: 预测概率
            n_bins: 分箱数量
        
        Returns:
            ECE 值
        """
        prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=n_bins)
        
        # 计算每个分箱的样本数
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(y_proba, bin_edges, right=True) - 1
        
        # 处理边界情况：确保索引在 [0, n_bins-1] 范围内
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
        
        bin_counts = np.bincount(bin_indices, minlength=n_bins)
        
        # 计算 ECE
        ece = 0.0
        n_valid_bins = 0
        
        for i in range(min(len(prob_true), len(prob_pred), len(bin_counts))):
            if bin_counts[i] > 0:
                ece += np.abs(prob_pred[i] - prob_true[i]) * bin_counts[i]
                n_valid_bins += 1
        
        if len(y_true) > 0:
            ece /= len(y_true)
        
        return ece
    
    def evaluate(
        self,
        y_proba_test: np.ndarray,
        y_true_test: np.ndarray
    ) -> Dict[str, Any]:
        """
        评估校准效果（在测试集上）
        
        Args:
            y_proba_test: 测试集概率预测（未校准）
            y_true_test: 测试集真实标签
        
        Returns:
            评估报告
        """
        if not self.is_fitted:
            raise ValueError("校准器未拟合。请先调用 fit() 方法")
        
        # 校准测试集概率
        y_proba_calibrated = self.predict_proba(y_proba_test)
        
        # 计算指标
        brier_score_before = brier_score_loss(y_true_test, y_proba_test)
        brier_score_after = brier_score_loss(y_true_test, y_proba_calibrated)
        
        ece_before = self._calculate_ece(y_true_test, y_proba_test)
        ece_after = self._calculate_ece(y_true_test, y_proba_calibrated)
        
        # 计算校准曲线
        prob_true_before, prob_pred_before = calibration_curve(y_true_test, y_proba_test, n_bins=10)
        prob_true_after, prob_pred_after = calibration_curve(y_true_test, y_proba_calibrated, n_bins=10)
        
        evaluation_report = {
            'method': self.method,
            'brier_score': {
                'before': brier_score_before,
                'after': brier_score_after,
                'improvement': brier_score_before - brier_score_after
            },
            'ece': {
                'before': ece_before,
                'after': ece_after,
                'improvement': ece_before - ece_after
            },
            'calibration_curve': {
                'before': {
                    'prob_true': prob_true_before.tolist(),
                    'prob_pred': prob_pred_before.tolist()
                },
                'after': {
                    'prob_true': prob_true_after.tolist(),
                    'prob_pred': prob_pred_after.tolist()
                }
            }
        }
        
        return evaluation_report
    
    def plot_calibration_curve(
        self,
        y_proba_test: np.ndarray,
        y_true_test: np.ndarray,
        save_path: str = None
    ):
        """
        绘制校准曲线
        
        Args:
            y_proba_test: 测试集概率预测
            y_true_test: 测试集真实标签
            save_path: 保存路径（可选）
        """
        if not self.is_fitted:
            raise ValueError("校准器未拟合。请先调用 fit() 方法")
        
        # 校准测试集概率
        y_proba_calibrated = self.predict_proba(y_proba_test)
        
        # 计算校准曲线
        prob_true_before, prob_pred_before = calibration_curve(y_true_test, y_proba_test, n_bins=10)
        prob_true_after, prob_pred_after = calibration_curve(y_true_test, y_proba_calibrated, n_bins=10)
        
        # 创建图表
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 绘制完美校准线
        ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration', linewidth=2)
        
        # 绘制校准前曲线
        ax.plot(prob_pred_before, prob_true_before, marker='o', 
                linewidth=2, label=f'Before Calibration (ECE={self._calculate_ece(y_true_test, y_proba_test):.4f})')
        
        # 绘制校准后曲线
        ax.plot(prob_pred_after, prob_true_after, marker='s', 
                linewidth=2, label=f'After Calibration (ECE={self._calculate_ece(y_true_test, y_proba_calibrated):.4f})')
        
        # 设置图表
        ax.set_xlabel('Mean Predicted Probability', fontsize=12)
        ax.set_ylabel('Fraction of Positives', fontsize=12)
        ax.set_title(f'Calibration Curve ({self.method.capitalize()} Scaling)', fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        
        plt.tight_layout()
        
        # 保存或显示
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"校准曲线已保存: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def save(self, file_path: str):
        """保存校准器"""
        import pickle
        
        model_data = {
            'method': self.method,
            'calibrator': self.calibrator,
            'is_fitted': self.is_fitted,
            'calibration_report': self.calibration_report
        }
        
        with open(file_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"校准器已保存: {file_path}")
    
    @classmethod
    def load(cls, file_path: str):
        """加载校准器"""
        import pickle
        
        with open(file_path, 'rb') as f:
            model_data = pickle.load(f)
        
        # 创建实例
        calibrator = cls(method=model_data['method'])
        calibrator.calibrator = model_data['calibrator']
        calibrator.is_fitted = model_data['is_fitted']
        calibrator.calibration_report = model_data['calibration_report']
        
        return calibrator


class EnsembleProbabilityCalibrator:
    """集成概率校准器
    
    结合多种校准方法，选择最优的
    """
    
    def __init__(
        self,
        methods: list = None,
        validation_ratio: float = 0.2
    ):
        """
        Args:
            methods: 校准方法列表，默认 ['platt', 'isotonic']
            validation_ratio: 验证集比例，用于选择最优方法
        """
        self.methods = methods or ['platt', 'isotonic']
        self.validation_ratio = validation_ratio
        
        self.calibrators = {}
        self.best_method = None
        self.best_calibrator = None
    
    def fit(
        self,
        y_proba_train: np.ndarray,
        y_true_train: np.ndarray
    ) -> Dict[str, Any]:
        """
        拟合多个校准器，选择最优的
        
        Args:
            y_proba_train: 训练集概率预测
            y_true_train: 训练集真实标签
        
        Returns:
            拟合报告
        """
        # 划分训练集和验证集
        from sklearn.model_selection import train_test_split
        
        if len(y_proba_train) < 1000:
            # 数据量小，使用全部数据
            X_cal, y_cal = y_proba_train, y_true_train
            X_val, y_val = y_proba_train, y_true_train
        else:
            X_cal, X_val, y_cal, y_val = train_test_split(
                y_proba_train, y_true_train,
                test_size=self.validation_ratio,
                random_state=42,
                stratify=y_true_train
            )
        
        # 拟合所有校准器
        best_ece = float('inf')
        
        for method in self.methods:
            calibrator = ProbabilityCalibrator(method=method)
            calibrator.fit(X_cal, y_cal)
            
            # 在验证集上评估
            evaluation = calibrator.evaluate(X_val, y_val)
            ece = evaluation['ece']['after']
            
            self.calibrators[method] = {
                'calibrator': calibrator,
                'evaluation': evaluation,
                'ece': ece
            }
            
            # 选择最优方法
            if ece < best_ece:
                best_ece = ece
                self.best_method = method
                self.best_calibrator = calibrator
        
        # 生成报告
        report = {
            'best_method': self.best_method,
            'best_ece': best_ece,
            'all_methods': {
                method: {
                    'ece': data['ece'],
                    'brier_score': data['evaluation']['brier_score']
                }
                for method, data in self.calibrators.items()
            }
        }
        
        return report
    
    def predict_proba(self, y_proba: np.ndarray) -> np.ndarray:
        """使用最优校准器预测"""
        if self.best_calibrator is None:
            raise ValueError("未拟合校准器。请先调用 fit() 方法")
        
        return self.best_calibrator.predict_proba(y_proba)
    
    def get_best_calibrator(self) -> ProbabilityCalibrator:
        """获取最优校准器"""
        return self.best_calibrator
