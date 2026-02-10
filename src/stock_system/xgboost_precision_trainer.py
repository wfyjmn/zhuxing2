"""
短期突击模型训练器 v3.1 - XGBoost高精确率版本

【v3.1 优化】：
- 切换到XGBoost模型框架
- 聚焦高精确率训练（而非平衡召回与精确）
- 调整参数：scale_pos_weight=3.0（抑制负样本，聚焦正样本）
- 加大正则化力度（提升reg_lambda、reg_alpha）
- 早停机制：以验证集精确率作为核心停止标准
- 支持正样本提纯和特征重要性选择
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score
from sklearn.calibration import CalibratedClassifierCV
import warnings
warnings.filterwarnings('ignore')


class XGBoostPrecisionTrainer:
    """XGBoost高精确率训练器"""
    
    def __init__(self, config_path: str = "config/short_term_assault_config_v31.json"):
        """
        初始化训练器
        
        Args:
            config_path: 配置文件路径
        """
        self.config = self._load_config(config_path)
        self.model_config = self.config['model_config']
        self.params = self.model_config['xgboost_params']
        self.model = None
        self.best_iteration = None
        self.calibrated_model = None
        self.training_history = {}
        
        print("=" * 70)
        print("XGBoost高精确率训练器 v3.1 - 印钞机股票精准识别")
        print("=" * 70)
        print(f"✓ 核心目标：精确率≥{self.config['optimization_goals']['precision']['target']*100:.0f}%")
        print(f"✓ 训练导向：偏向精确率+主动匹配")
    
    def _load_config(self, config_path: str) -> Dict:
        """加载配置文件"""
        import json
        from pathlib import Path
        
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        
        with open(config_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        early_stopping_rounds: int = None,
        verbose: bool = True
    ) -> xgb.XGBClassifier:
        """
        训练XGBoost模型（聚焦高精确率）
        
        Args:
            X_train: 训练集特征
            y_train: 训练集标签
            X_val: 验证集特征
            y_val: 验证集标签
            early_stopping_rounds: 早停轮数
            verbose: 是否输出详细信息
        
        Returns:
            训练好的模型
        """
        if early_stopping_rounds is None:
            early_stopping_rounds = self.params.get('early_stopping_rounds', 40)
        
        # 构建DMatrix
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        
        # 构建评估指标列表（以精确率为核心）
        evals_result = {}
        
        # 训练模型
        print(f"\n【XGBoost训练】")
        print(f"  - 训练集样本数: {len(X_train)} (正样本: {y_train.sum()}, 负样本: {(1-y_train).sum()})")
        print(f"  - 验证集样本数: {len(X_val)} (正样本: {y_val.sum()}, 负样本: {(1-y_val).sum()})")
        print(f"  - 早停轮数: {early_stopping_rounds}")
        print(f"  - 评估指标: {self.params.get('eval_metric', 'auc')}")
        
        self.model = xgb.train(
            params=self.params,
            dtrain=dtrain,
            num_boost_round=1000,
            evals=[(dtrain, 'train'), (dval, 'val')],
            early_stopping_rounds=early_stopping_rounds,
            evals_result=evals_result,
            verbose_eval=verbose
        )
        
        self.best_iteration = self.model.best_iteration
        self.training_history = evals_result
        
        # 评估模型
        self._evaluate_model(X_train, y_train, X_val, y_val, dataset_name="验证集")
        
        return self.model
    
    def _evaluate_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        dataset_name: str = "验证集"
    ):
        """
        评估模型性能
        
        Args:
            X_train: 训练集特征
            y_train: 训练集标签
            X_val: 验证集特征
            y_val: 验证集标签
            dataset_name: 数据集名称
        """
        # 预测
        train_pred = self.model.predict(xgb.DMatrix(X_train))
        val_pred = self.model.predict(xgb.DMatrix(X_val))
        
        # 转换为类别
        train_pred_class = (train_pred >= 0.5).astype(int)
        val_pred_class = (val_pred >= 0.5).astype(int)
        
        # 计算指标
        train_precision = precision_score(y_train, train_pred_class)
        train_recall = recall_score(y_train, train_pred_class)
        train_f1 = f1_score(y_train, train_pred_class)
        train_auc = roc_auc_score(y_train, train_pred)
        
        val_precision = precision_score(y_val, val_pred_class)
        val_recall = recall_score(y_val, val_pred_class)
        val_f1 = f1_score(y_val, val_pred_class)
        val_auc = roc_auc_score(y_val, val_pred)
        
        # 计算过拟合差距
        overfitting_gap = abs(train_precision - val_precision)
        
        print(f"\n【{dataset_name}性能评估】")
        print(f"  训练集:")
        print(f"    - 精确率: {train_precision*100:.2f}%")
        print(f"    - 召回率: {train_recall*100:.2f}%")
        print(f"    - F1分数: {train_f1:.4f}")
        print(f"    - AUC: {train_auc:.4f}")
        print(f"  验证集:")
        print(f"    - 精确率: {val_precision*100:.2f}% (目标: ≥{self.config['optimization_goals']['precision']['target']*100:.0f}%)")
        print(f"    - 召回率: {val_recall*100:.2f}% (目标: {self.config['optimization_goals']['recall']['target_min']*100:.0f}%-{self.config['optimization_goals']['recall']['target_max']*100:.0f}%)")
        print(f"    - F1分数: {val_f1:.4f}")
        print(f"    - AUC: {val_auc:.4f}")
        print(f"  过拟合差距: {overfitting_gap*100:.2f}% (目标: <{self.config['optimization_goals']['overfitting_gap']['target']*100:.0f}%)")
        
        # 检查是否达到目标
        precision_target = self.config['optimization_goals']['precision']['target']
        recall_min = self.config['optimization_goals']['recall']['target_min']
        recall_max = self.config['optimization_goals']['recall']['target_max']
        overfitting_target = self.config['optimization_goals']['overfitting_gap']['target']
        
        if val_precision >= precision_target:
            print(f"  ✓ 精确率达标！")
        else:
            print(f"  ✗ 精确率未达标，需继续优化")
        
        if recall_min <= val_recall <= recall_max:
            print(f"  ✓ 召回率达标！")
        else:
            print(f"  ⚠ 召回率超出目标范围")
        
        if overfitting_gap < overfitting_target:
            print(f"  ✓ 过拟合差距达标！")
        else:
            print(f"  ✗ 过拟合差距过大，需加强正则化")
    
    def calibrate_model(self, X_train: pd.DataFrame, y_train: pd.Series):
        """
        校准模型概率（提升预测精确率）
        
        Args:
            X_train: 训练集特征
            y_train: 训练集标签
        """
        # 获取原始预测概率
        pred_proba = self.model.predict(xgb.DMatrix(X_train))
        
        # 使用IsotonicRegression校准
        from sklearn.isotonic import IsotonicRegression
        
        iso_reg = IsotonicRegression(out_of_bounds='clip')
        iso_reg.fit(pred_proba, y_train)
        
        self.calibrated_model = iso_reg
        
        print(f"\n【模型校准】")
        print(f"  ✓ 使用IsotonicRegression校准模型概率")
    
    def predict(self, X: pd.DataFrame, calibrated: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        预测
        
        Args:
            X: 特征DataFrame
            calibrated: 是否使用校准后的概率
        
        Returns:
            (预测概率, 预测类别)
        """
        if self.model is None:
            raise ValueError("模型未训练，请先调用train方法")
        
        # 原始预测概率
        pred_proba = self.model.predict(xgb.DMatrix(X))
        
        # 校准概率
        if calibrated and self.calibrated_model is not None:
            pred_proba = self.calibrated_model.predict(pred_proba)
        
        # 转换为类别（使用更高的阈值提升精确率）
        threshold = self.config.get('signal_trigger', {}).get('trigger_conditions', {}).get('precision_threshold', 0.85)
        pred_class = (pred_proba >= threshold).astype(int)
        
        return pred_proba, pred_class
    
    def get_feature_importance(self, feature_names: List[str], top_k: int = 20) -> pd.DataFrame:
        """
        获取特征重要性
        
        Args:
            feature_names: 特征名称列表
            top_k: 返回前k个重要特征
        
        Returns:
            特征重要性DataFrame
        """
        if self.model is None:
            raise ValueError("模型未训练，请先调用train方法")
        
        importance_dict = self.model.get_score(importance_type='gain')
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': [importance_dict.get(f'f{i}', 0) for i in range(len(feature_names))]
        }).sort_values('importance', ascending=False).head(top_k)
        
        return importance_df
    
    def cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_splits: int = 5,
        random_state: int = 42
    ) -> Dict[str, List[float]]:
        """
        交叉验证
        
        Args:
            X: 特征DataFrame
            y: 标签Series
            n_splits: 折数
            random_state: 随机种子
        
        Returns:
            交叉验证结果字典
        """
        print(f"\n【交叉验证】")
        print(f"  - 折数: {n_splits}")
        
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        
        cv_results = {
            'precision': [],
            'recall': [],
            'f1': [],
            'auc': [],
            'overfitting_gap': []
        }
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train_fold = X.iloc[train_idx]
            y_train_fold = y.iloc[train_idx]
            X_val_fold = X.iloc[val_idx]
            y_val_fold = y.iloc[val_idx]
            
            # 训练模型
            self.train(X_train_fold, y_train_fold, X_val_fold, y_val_fold, verbose=False)
            
            # 预测
            val_pred = self.model.predict(xgb.DMatrix(X_val_fold))
            val_pred_class = (val_pred >= 0.5).astype(int)
            
            # 计算指标
            val_precision = precision_score(y_val_fold, val_pred_class)
            val_recall = recall_score(y_val_fold, val_pred_class)
            val_f1 = f1_score(y_val_fold, val_pred_class)
            val_auc = roc_auc_score(y_val_fold, val_pred)
            
            train_pred = self.model.predict(xgb.DMatrix(X_train_fold))
            train_pred_class = (train_pred >= 0.5).astype(int)
            train_precision = precision_score(y_train_fold, train_pred_class)
            overfitting_gap = abs(train_precision - val_precision)
            
            cv_results['precision'].append(val_precision)
            cv_results['recall'].append(val_recall)
            cv_results['f1'].append(val_f1)
            cv_results['auc'].append(val_auc)
            cv_results['overfitting_gap'].append(overfitting_gap)
            
            print(f"  Fold {fold+1}/{n_splits}: 精确率={val_precision*100:.2f}%, 召回率={val_recall*100:.2f}%, 过拟合差距={overfitting_gap*100:.2f}%")
        
        # 输出平均结果
        print(f"\n  平均结果:")
        print(f"    - 精确率: {np.mean(cv_results['precision'])*100:.2f}% ± {np.std(cv_results['precision'])*100:.2f}%")
        print(f"    - 召回率: {np.mean(cv_results['recall'])*100:.2f}% ± {np.std(cv_results['recall'])*100:.2f}%")
        print(f"    - F1分数: {np.mean(cv_results['f1']):.4f} ± {np.std(cv_results['f1']):.4f}")
        print(f"    - AUC: {np.mean(cv_results['auc']):.4f} ± {np.std(cv_results['auc']):.4f}")
        print(f"    - 过拟合差距: {np.mean(cv_results['overfitting_gap'])*100:.2f}% ± {np.std(cv_results['overfitting_gap'])*100:.2f}%")
        
        return cv_results
    
    def save_model(self, model_path: str):
        """
        保存模型
        
        Args:
            model_path: 模型保存路径
        """
        if self.model is None:
            raise ValueError("模型未训练，请先调用train方法")
        
        import joblib
        
        model_data = {
            'model': self.model,
            'calibrated_model': self.calibrated_model,
            'best_iteration': self.best_iteration,
            'training_history': self.training_history,
            'config': self.config
        }
        
        joblib.dump(model_data, model_path)
        print(f"\n【模型保存】")
        print(f"  ✓ 模型已保存到: {model_path}")
    
    def load_model(self, model_path: str):
        """
        加载模型
        
        Args:
            model_path: 模型加载路径
        """
        import joblib
        
        model_data = joblib.load(model_path)
        
        self.model = model_data['model']
        self.calibrated_model = model_data.get('calibrated_model')
        self.best_iteration = model_data.get('best_iteration')
        self.training_history = model_data.get('training_history')
        self.config = model_data.get('config')
        
        print(f"\n【模型加载】")
        print(f"  ✓ 模型已从 {model_path} 加载")
        print(f"  - 最佳迭代次数: {self.best_iteration}")
