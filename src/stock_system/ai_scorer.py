"""
AI 智能评分系统（V5.0 模型集成）
将训练好的机器学习模型作为"大脑"，为柱形突击选股提供决策支持
"""

import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class AITradingBrain:
    """AI 交易大脑（V5.0 模型）"""
    
    def __init__(self, model_path: str = "assets/models/xgboost_v5_auto_optuna.pkl",
                 feature_names_path: str = "assets/models/v5_feature_names.pkl"):
        """
        初始化 AI 交易大脑
        
        Args:
            model_path: V5.0 模型路径
            feature_names_path: 特征名称路径
        """
        self.model = None
        self.feature_names = None
        self.model_loaded = False
        
        # 尝试加载模型
        self._load_model(model_path, feature_names_path)
    
    def _load_model(self, model_path: str, feature_names_path: str):
        """加载模型和特征名称"""
        model_file = Path(model_path)
        feature_file = Path(feature_names_path)
        
        if not model_file.exists():
            print(f"⚠ 模型文件不存在: {model_path}")
            print("  请先运行训练脚本生成模型")
            return
        
        if not feature_file.exists():
            print(f"⚠ 特征名称文件不存在: {feature_names_path}")
            return
        
        try:
            # 加载模型
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            # 如果模型数据是字典，提取实际的模型
            if isinstance(model_data, dict):
                self.model = model_data.get('model')
                self.feature_names = model_data.get('feature_names', [])
            else:
                self.model = model_data
                # 加载特征名称
                with open(feature_names_path, 'rb') as f:
                    self.feature_names = pickle.load(f)
            
            self.model_loaded = True
            print(f"✓ AI 大脑已加载: {model_path}")
            print(f"  模型类型: {type(self.model).__name__}")
            print(f"  特征数量: {len(self.feature_names)}")
        except Exception as e:
            print(f"✗ 加载模型失败: {str(e)}")
    
    def is_loaded(self) -> bool:
        """检查模型是否已加载"""
        return self.model_loaded
    
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """
        预测上涨概率
        
        Args:
            df: 包含特征的DataFrame
        
        Returns:
            上涨概率数组 (shape: [n_samples, 2], 第0列是负类概率，第1列是正类概率)
        """
        if not self.model_loaded:
            raise RuntimeError("模型未加载，请先运行训练脚本生成模型")
        
        # 检查特征是否存在
        missing_features = [f for f in self.feature_names if f not in df.columns]
        if missing_features:
            raise ValueError(f"数据缺少必要特征: {missing_features[:10]}...")
        
        # 提取特征
        X = df[self.feature_names].values
        
        # 预测概率
        proba = self.model.predict_proba(X)
        
        return proba
    
    def get_up_probability(self, df: pd.DataFrame) -> pd.Series:
        """
        获取上涨概率（仅返回正类概率）
        
        Args:
            df: 包含特征的DataFrame
        
        Returns:
            上涨概率 Series
        """
        proba = self.predict_proba(df)
        return pd.Series(proba[:, 1], index=df.index)


class AISignalGrader:
    """AI 信号分级器"""
    
    def __init__(self, config_path: str = "config/short_term_assault_config.json"):
        """
        初始化信号分级器
        
        Args:
            config_path: 配置文件路径
        """
        import json
        
        self.config = self._load_config(config_path)
        self.grading_thresholds = {
            'A': 0.75,    # A级：概率 > 75%
            'B': 0.65,    # B级：概率 > 65%
            'C': 0.55,    # C级：概率 > 55%
            'D': 0.0      # D级：概率 <= 55%
        }
        
        # 从配置加载分级阈值（如果配置中存在）
        if 'ai_grading' in self.config:
            thresholds = self.config['ai_grading'].get('thresholds', {})
            if thresholds:
                self.grading_thresholds = thresholds
    
    def _load_config(self, config_path: str) -> Dict:
        """加载配置文件"""
        config_file = Path(config_path)
        if not config_file.exists():
            print(f"⚠ 配置文件不存在: {config_path}，使用默认配置")
            return {}
        
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def get_signal_grade(self, probability: float) -> str:
        """
        根据概率确定信号等级
        
        Args:
            probability: 上涨概率 (0-1)
        
        Returns:
            信号等级 ('A', 'B', 'C', 'D')
        """
        if probability >= self.grading_thresholds['A']:
            return 'A'
        elif probability >= self.grading_thresholds['B']:
            return 'B'
        elif probability >= self.grading_thresholds['C']:
            return 'C'
        else:
            return 'D'
    
    def get_position_ratio(self, signal_grade: str, base_ratio: float = 0.05) -> float:
        """
        根据信号等级确定仓位比例
        
        Args:
            signal_grade: 信号等级
            base_ratio: 基础仓位比例
        
        Returns:
            仓位比例
        """
        grade_config = {
            'A': 2.0,    # A级：2倍基础仓位（10%）
            'B': 1.5,    # B级：1.5倍基础仓位（7.5%）
            'C': 1.0,    # C级：1倍基础仓位（5%）
            'D': 0.0     # D级：0仓位
        }
        
        return base_ratio * grade_config.get(signal_grade, 0.0)
    
    def get_stop_loss(self, signal_grade: str) -> float:
        """
        根据信号等级确定止损比例
        
        Args:
            signal_grade: 信号等级
        
        Returns:
            止损比例（负数）
        """
        grade_config = {
            'A': -0.08,   # A级：8%止损
            'B': -0.10,   # B级：10%止损
            'C': -0.12,   # C级：12%止损
            'D': -0.20    # D级：20%止损（虽然D级不买入）
        }
        
        return grade_config.get(signal_grade, -0.10)
    
    def get_threshold_info(self) -> Dict[str, float]:
        """获取当前分级阈值"""
        return self.grading_thresholds.copy()


class AITradingScorer:
    """AI 交易评分系统（整合模块）"""
    
    def __init__(self, 
                 model_path: str = "assets/models/xgboost_v5_auto_optuna.pkl",
                 feature_names_path: str = "assets/models/v5_feature_names.pkl",
                 config_path: str = "config/short_term_assault_config.json"):
        """
        初始化 AI 交易评分系统
        
        Args:
            model_path: V5.0 模型路径
            feature_names_path: 特征名称路径
            config_path: 配置文件路径
        """
        self.brain = AITradingBrain(model_path, feature_names_path)
        self.grader = AISignalGrader(config_path)
        
        print("=" * 70)
        print("AI 智能评分系统初始化完成")
        print("=" * 70)
        print(f"模型加载状态: {'✓ 已加载' if self.brain.is_loaded() else '✗ 未加载'}")
        print(f"分级阈值:")
        for grade, threshold in self.grader.get_threshold_info().items():
            if grade != 'D':
                print(f"  {grade}级: 概率 > {threshold:.2%}")
        print("=" * 70)
    
    def is_ready(self) -> bool:
        """检查系统是否就绪"""
        return self.brain.is_loaded()
    
    def score_stocks(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        为股票评分
        
        Args:
            df: 包含特征的DataFrame
        
        Returns:
            添加了评分和信号等级的DataFrame
        """
        if not self.is_ready():
            raise RuntimeError("系统未就绪，模型未加载")
        
        # 获取上涨概率
        df_scored = df.copy()
        df_scored['up_probability'] = self.brain.get_up_probability(df)
        
        # 确定信号等级
        df_scored['signal_grade'] = df_scored['up_probability'].apply(
            lambda p: self.grader.get_signal_grade(p)
        )
        
        # 添加仓位建议
        base_ratio = 0.05
        df_scored['position_ratio'] = df_scored['signal_grade'].apply(
            lambda g: self.grader.get_position_ratio(g, base_ratio)
        )
        
        # 添加止损建议
        df_scored['stop_loss'] = df_scored['signal_grade'].apply(
            lambda g: self.grader.get_stop_loss(g)
        )
        
        return df_scored
    
    def get_top_stocks(self, df_scored: pd.DataFrame, 
                      top_n: int = 10,
                      min_grade: str = 'C') -> pd.DataFrame:
        """
        获取排名前 N 的股票
        
        Args:
            df_scored: 已评分的DataFrame
            top_n: 返回数量
            min_grade: 最低信号等级
        
        Returns:
            排序后的股票DataFrame
        """
        # 筛选等级
        grade_order = {'A': 4, 'B': 3, 'C': 2, 'D': 1}
        df_filtered = df_scored[
            df_scored['signal_grade'].apply(
                lambda g: grade_order.get(g, 0) >= grade_order.get(min_grade, 0)
            )
        ].copy()
        
        # 排序（先按等级，再按概率）
        df_filtered['grade_order'] = df_filtered['signal_grade'].map(grade_order)
        df_sorted = df_filtered.sort_values(
            ['grade_order', 'up_probability'],
            ascending=[False, False]
        )
        
        return df_sorted.head(top_n).drop(columns=['grade_order'])


def create_ai_scorer_demo():
    """演示 AI 评分系统的使用"""
    print("=" * 70)
    print("AI 智能评分系统演示")
    print("=" * 70)
    print()
    
    # 1. 初始化评分系统
    print("【步骤1】初始化 AI 评分系统")
    print("-" * 70)
    scorer = AITradingScorer()
    print()
    
    if not scorer.is_ready():
        print("⚠ 模型未加载，请先运行训练脚本")
        print("  运行命令: python scripts/train_auto_optuna.py")
        return
    
    # 2. 生成测试数据
    print("【步骤2】生成测试数据")
    print("-" * 70)
    np.random.seed(42)
    n_samples = 100
    
    # 生成模拟特征
    df_test = pd.DataFrame({
        'open': np.random.uniform(10, 100, n_samples),
        'high': np.random.uniform(10, 100, n_samples),
        'low': np.random.uniform(10, 100, n_samples),
        'close': np.random.uniform(10, 100, n_samples),
        'volume': np.random.uniform(100000, 10000000, n_samples)
    })
    
    # 添加 V5.0 模型所需的特征（简化版）
    # 注意：这里只是演示，实际使用时需要完整的特征工程
    feature_cols = scorer.brain.feature_names
    for col in feature_cols:
        if col not in df_test.columns:
            df_test[col] = np.random.randn(n_samples)
    
    print(f"测试数据: {len(df_test)}条")
    print(f"特征数量: {len(feature_cols)}")
    print()
    
    # 3. 评分
    print("【步骤3】AI 评分")
    print("-" * 70)
    try:
        df_scored = scorer.score_stocks(df_test)
        
        print(f"评分完成！")
        print(f"  平均上涨概率: {df_scored['up_probability'].mean():.2%}")
        print()
        
        # 4. 信号分级统计
        print("【步骤4】信号分级统计")
        print("-" * 70)
        grade_counts = df_scored['signal_grade'].value_counts()
        for grade in ['A', 'B', 'C', 'D']:
            count = grade_counts.get(grade, 0)
            percentage = count / len(df_scored) * 100
            print(f"  {grade}级: {count}次 ({percentage:.1f}%)")
        print()
        
        # 5. 获取顶级股票
        print("【步骤5】顶级股票（Top 10）")
        print("-" * 70)
        top_stocks = scorer.get_top_stocks(df_scored, top_n=10, min_grade='C')
        
        print(f"{'等级':<6} {'概率':<10} {'仓位':<10} {'止损':<10}")
        print("-" * 70)
        for idx, row in top_stocks.head(10).iterrows():
            print(f"{row['signal_grade']:<6} "
                  f"{row['up_probability']:<10.2%} "
                  f"{row['position_ratio']:<10.2%} "
                  f"{row['stop_loss']:<10.2%}")
        print()
        
        print("=" * 70)
        print("演示完成！")
        print("=" * 70)
        
    except Exception as e:
        print(f"✗ 评分失败: {str(e)}")
        print("  可能原因：特征不匹配或模型版本不一致")


if __name__ == "__main__":
    create_ai_scorer_demo()
