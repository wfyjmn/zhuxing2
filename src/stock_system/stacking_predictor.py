"""
Stacking 集成推理器
用于加载训练好的集成模型并进行预测
"""

import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')


class StackingEnsemblePredictor:
    """Stacking 集成推理器"""
    
    def __init__(self, model_path: str = None, threshold: float = 0.5):
        """
        初始化集成推理器
        
        Args:
            model_path: 集成模型路径
            threshold: 决策阈值（默认 0.5）
        """
        self.model_path = model_path
        self.base_models = {}
        self.meta_model = None
        self.feature_sets = {}
        self.loaded = False
        self.threshold = threshold
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path: str):
        """
        加载集成模型
        
        Args:
            model_path: 集成模型路径
        """
        model_file = Path(model_path)
        
        if not model_file.exists():
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.base_models = model_data.get('base_models', {})
            self.meta_model = model_data.get('meta_model')
            self.feature_sets = model_data.get('feature_sets', {})
            self.config = model_data.get('config', {})
            self.results = model_data.get('results', {})
            self.loaded = True
            
            print(f"✓ 集成模型加载成功")
            print(f"  基学习器: {list(self.base_models.keys())}")
            print(f"  元学习器: {type(self.meta_model).__name__}")
            
        except Exception as e:
            raise RuntimeError(f"加载模型失败: {e}")
    
    def is_loaded(self) -> bool:
        """检查模型是否已加载"""
        return self.loaded
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        预测上涨概率
        
        Args:
            df: 包含特征的DataFrame
        
        Returns:
            上涨概率数组 (0-1)
        """
        if not self.loaded:
            raise RuntimeError("模型未加载，请先调用 load_model()")
        
        # 生成元特征
        meta_features = self._generate_meta_features(df)
        
        # 使用元学习器预测
        X_meta = meta_features[self._get_meta_feature_names()]
        prob = self.meta_model.predict_proba(X_meta)[:, 1]
        
        return prob
    
    def _generate_meta_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        生成元特征（基学习器的预测结果）
        
        Args:
            df: 包含特征的DataFrame
        
        Returns:
            元特征DataFrame
        """
        meta_features = pd.DataFrame(index=df.index)
        
        # XGBoost 预测
        if 'xgboost' in self.base_models:
            xgb_features = [f for f in self.feature_sets['xgboost'] if f in df.columns]
            X_xgb = df[xgb_features]
            meta_features['xgboost_prob'] = self.base_models['xgboost'].predict_proba(X_xgb)[:, 1]
        
        # LightGBM 预测
        if 'lightgbm' in self.base_models:
            lgb_features = [f for f in self.feature_sets['lightgbm'] if f in df.columns]
            X_lgb = df[lgb_features]
            meta_features['lgbm_prob'] = self.base_models['lightgbm'].predict_proba(X_lgb)[:, 1]
        
        # RandomForest 预测
        if 'randomforest' in self.base_models:
            rf_features = [f for f in self.feature_sets['randomforest'] if f in df.columns]
            X_rf = df[rf_features]
            meta_features['rf_prob'] = self.base_models['randomforest'].predict_proba(X_rf)[:, 1]
        
        # 生成一致性特征
        if 'xgboost_prob' in meta_features.columns and 'lgbm_prob' in meta_features.columns:
            meta_features['xgboost_lgbm_diff'] = np.abs(
                meta_features['xgboost_prob'] - meta_features['lgbm_prob']
            )
        
        if 'xgboost_prob' in meta_features.columns and 'rf_prob' in meta_features.columns:
            meta_features['xgboost_rf_diff'] = np.abs(
                meta_features['xgboost_prob'] - meta_features['rf_prob']
            )
        
        if 'lgbm_prob' in meta_features.columns and 'rf_prob' in meta_features.columns:
            meta_features['lgbm_rf_diff'] = np.abs(
                meta_features['lgbm_prob'] - meta_features['rf_prob']
            )
        
        # 置信度特征
        if 'xgboost_prob' in meta_features.columns:
            meta_features['xgboost_confidence'] = (meta_features['xgboost_prob'] > 0.5).astype(int)
        
        if 'lgbm_prob' in meta_features.columns:
            meta_features['lgbm_confidence'] = (meta_features['lgbm_prob'] > 0.5).astype(int)
        
        if 'rf_prob' in meta_features.columns:
            meta_features['rf_confidence'] = (meta_features['rf_prob'] > 0.5).astype(int)
        
        # 一致性特征
        if all(col in meta_features.columns for col in ['xgboost_prob', 'lgbm_prob', 'rf_prob']):
            meta_features['unanimous_high'] = (
                (meta_features['xgboost_prob'] > 0.7) &
                (meta_features['lgbm_prob'] > 0.7) &
                (meta_features['rf_prob'] > 0.7)
            ).astype(int)
            
            meta_features['unanimous_low'] = (
                (meta_features['xgboost_prob'] < 0.3) &
                (meta_features['lgbm_prob'] < 0.3) &
                (meta_features['rf_prob'] < 0.3)
            ).astype(int)
            
            model_probs = meta_features[['xgboost_prob', 'lgbm_prob', 'rf_prob']]
            meta_features['split_decision'] = (model_probs.std(axis=1) > 0.3).astype(int)
        
        return meta_features
    
    def _get_meta_feature_names(self) -> List[str]:
        """获取元特征名称"""
        return [
            'xgboost_prob',
            'lgbm_prob',
            'rf_prob',
            'xgboost_lgbm_diff',
            'xgboost_rf_diff',
            'lgbm_rf_diff',
            'xgboost_confidence',
            'lgbm_confidence',
            'rf_confidence',
            'unanimous_high',
            'unanimous_low',
            'split_decision'
        ]
    
    def get_model_info(self) -> Dict:
        """获取模型信息"""
        return {
            'base_models': list(self.base_models.keys()),
            'meta_model': type(self.meta_model).__name__,
            'feature_sets': {k: len(v) for k, v in self.feature_sets.items()},
            'results': self.results.get('ensemble', {}),
            'meta_feature_importance': self.results.get('meta_feature_importance', {}),
            'threshold': self.threshold
        }
    
    def predict_signal(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        预测信号（包含概率和信号等级）
        
        Args:
            df: 包含特征的DataFrame
        
        Returns:
            包含预测结果的DataFrame（新增 prob 和 signal 列）
        """
        if not self.loaded:
            raise RuntimeError("模型未加载，请先调用 load_model()")
        
        # 预测概率
        prob = self.predict(df)
        
        # 生成信号等级
        signal = self._classify_signal(prob)
        
        # 添加结果到原DataFrame
        result_df = df.copy()
        result_df['prob'] = prob
        result_df['signal'] = signal
        
        return result_df
    
    def _classify_signal(self, prob: np.ndarray) -> pd.Series:
        """
        根据概率对信号进行分级
        
        Args:
            prob: 预测概率数组
        
        Returns:
            信号等级Series（0: 看空, 1: 中性, 2: 看涨, 3: 强看涨）
        """
        signal = pd.Series(0, index=range(len(prob)))
        
        # 定义信号等级（基于优化后的阈值）
        # 阈值 0.5: 中性
        # 阈值 0.61: 看涨（优化后阈值，精确率 45%）
        # 阈值 0.75: 强看涨
        
        signal[prob >= 0.75] = 3  # 强看涨
        signal[prob >= 0.61] = 2  # 看涨
        signal[prob >= 0.40] = 1  # 中性
        signal[prob < 0.40] = 0  # 看空
        
        return signal
    
    def get_signal_stats(self, df: pd.DataFrame) -> Dict:
        """
        获取信号统计信息
        
        Args:
            df: 包含特征的DataFrame
        
        Returns:
            信号统计信息
        """
        result_df = self.predict_signal(df)
        
        stats = {
            'total_samples': len(result_df),
            'signal_distribution': result_df['signal'].value_counts().to_dict(),
            'signal_names': {
                0: '看空',
                1: '中性',
                2: '看涨',
                3: '强看涨'
            },
            'prob_stats': {
                'mean': result_df['prob'].mean(),
                'std': result_df['prob'].std(),
                'min': result_df['prob'].min(),
                'max': result_df['prob'].max(),
                'median': result_df['prob'].median()
            }
        }
        
        return stats
    
    def select_stocks(self, df: pd.DataFrame, min_signal: int = 2, top_n: int = None) -> pd.DataFrame:
        """
        选股：筛选出信号等级 >= min_signal 的股票
        
        Args:
            df: 包含特征的DataFrame
            min_signal: 最小信号等级（2=看涨, 3=强看涨）
            top_n: 返回前N只股票（按概率排序）
        
        Returns:
            选股结果DataFrame
        """
        result_df = self.predict_signal(df)
        
        # 筛选信号
        selected = result_df[result_df['signal'] >= min_signal].copy()
        
        # 按概率排序
        selected = selected.sort_values('prob', ascending=False)
        
        # 返回前N只
        if top_n is not None and top_n > 0:
            selected = selected.head(top_n)
        
        return selected


# 使用示例
if __name__ == "__main__":
    # 加载模型
    predictor = StackingEnsemblePredictor("assets/models/stacking_ensemble_xxx.pkl")
    
    # 预测
    # prob = predictor.predict(df_with_features)
    
    # 查看模型信息
    # info = predictor.get_model_info()
    # print(info)
