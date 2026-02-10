"""
对抗性验证和数据质量检测
检测数据泄露和标签稳定性
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score
from scipy.stats import chi2_contingency, entropy
import warnings
warnings.filterwarnings('ignore')


class AdversarialValidation:
    """
    对抗性验证器
    
    检测训练集和测试集之间的分布差异，判断是否存在数据泄露
    
    原理：
    1. 将训练集和测试集合并，创建新标签（训练集=1，测试集=0）
    2. 训练分类器区分训练集和测试集
    3. 如果AUC > 0.6，说明存在显著分布差异（可能的数据泄露）
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        auc_threshold: float = 0.6,
        random_state: int = 42
    ):
        """
        初始化对抗性验证器
        
        参数:
            n_splits: 交叉验证折数
            auc_threshold: AUC阈值（超过此值认为有泄露）
            random_state: 随机种子
        """
        self.n_splits = n_splits
        self.auc_threshold = auc_threshold
        self.random_state = random_state
        
        self.adversarial_model = None
        self.results = None
    
    def validate(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        feature_importance_threshold: float = 0.05
    ) -> Dict[str, Any]:
        """
        执行对抗性验证
        
        参数:
            X_train: 训练集特征
            X_test: 测试集特征
            feature_importance_threshold: 特征重要性阈值
            
        返回:
            验证结果字典
        """
        # 确保特征一致
        common_features = list(set(X_train.columns) & set(X_test.columns))
        X_train = X_train[common_features]
        X_test = X_test[common_features]
        
        # 创建对抗性标签
        y_adversarial = np.concatenate([
            np.ones(len(X_train)),
            np.zeros(len(X_test))
        ])
        
        # 合并数据
        X_adversarial = pd.concat([X_train, X_test], ignore_index=True)
        
        # 训练对抗性模型
        self.adversarial_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        # 交叉验证
        cv = StratifiedKFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.random_state
        )
        
        cv_scores = cross_val_score(
            self.adversarial_model,
            X_adversarial,
            y_adversarial,
            cv=cv,
            scoring='roc_auc',
            n_jobs=-1
        )
        
        # 训练完整模型以获取特征重要性
        self.adversarial_model.fit(X_adversarial, y_adversarial)
        
        # 获取特征重要性
        feature_importance = pd.DataFrame({
            'feature': common_features,
            'importance': self.adversarial_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # 找出高重要性特征（可能是泄露特征）
        leaky_features = feature_importance[
            feature_importance['importance'] >= feature_importance_threshold
        ]['feature'].tolist()
        
        # 判断结果
        mean_auc = cv_scores.mean()
        std_auc = cv_scores.std()
        
        if mean_auc > self.auc_threshold:
            status = 'leak_detected'
            message = f"检测到可能的数据泄露（AUC={mean_auc:.4f}）"
        elif mean_auc > self.auc_threshold - 0.1:
            status = 'suspicious'
            message = f"存在轻微的分布差异（AUC={mean_auc:.4f}）"
        else:
            status = 'no_leak'
            message = f"未检测到数据泄露（AUC={mean_auc:.4f}）"
        
        self.results = {
            'status': status,
            'message': message,
            'mean_auc': float(mean_auc),
            'std_auc': float(std_auc),
            'cv_scores': cv_scores.tolist(),
            'feature_importance': feature_importance.to_dict('records'),
            'leaky_features': leaky_features,
            'n_leaky_features': len(leaky_features),
            'auc_threshold': self.auc_threshold
        }
        
        return self.results
    
    def generate_report(self, save_path: Optional[str] = None) -> str:
        """
        生成交抗性验证报告
        
        参数:
            save_path: 保存路径（可选）
            
        返回:
            报告字符串
        """
        if self.results is None:
            raise ValueError("请先执行验证")
        
        results = self.results
        
        report_lines = []
        
        report_lines.append("# 对抗性验证报告")
        report_lines.append("\n## 一、验证结果")
        report_lines.append(f"- **状态**: {results['status']}")
        report_lines.append(f"- **消息**: {results['message']}")
        report_lines.append(f"- **AUC**: {results['mean_auc']:.4f} ± {results['std_auc']:.4f}")
        report_lines.append(f"- **阈值**: {results['auc_threshold']}")
        
        report_lines.append("\n## 二、特征重要性")
        
        feature_importance = pd.DataFrame(results['feature_importance'])
        
        for idx, row in feature_importance.head(10).iterrows():
            report_lines.append(
                f"{idx + 1}. {row['feature']}: {row['importance']:.4f}"
            )
        
        if results['n_leaky_features'] > 0:
            report_lines.append("\n## 三、潜在泄露特征")
            report_lines.append(f"发现 {results['n_leaky_features']} 个潜在泄露特征:")
            for feature in results['leaky_features']:
                report_lines.append(f"- {feature}")
            
            report_lines.append("\n## 四、建议")
            report_lines.append("1. 检查这些特征是否包含未来信息")
            report_lines.append("2. 考虑移除或重新设计这些特征")
            report_lines.append("3. 重新划分数据集")
        else:
            report_lines.append("\n## 三、结论")
            report_lines.append("✅ 未发现明显的数据泄露，可以继续进行模型训练。")
        
        report = "\n".join(report_lines)
        
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report)
        
        return report


class LabelStabilityTest:
    """
    标签稳定性测试器
    
    检测标签的稳定性和质量
    """
    
    def __init__(
        self,
        expected_positive_ratio: float = 0.1,
        ratio_tolerance: float = 0.02,
        min_positive_samples: int = 10
    ):
        """
        初始化标签稳定性测试器
        
        参数:
            expected_positive_ratio: 预期正样本比例
            ratio_tolerance: 比例容忍度
            min_positive_samples: 最小正样本数
        """
        self.expected_positive_ratio = expected_positive_ratio
        self.ratio_tolerance = ratio_tolerance
        self.min_positive_samples = min_positive_samples
        
        self.results = None
    
    def test(
        self,
        y_train: np.ndarray,
        y_test: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        执行标签稳定性测试
        
        参数:
            y_train: 训练集标签
            y_test: 测试集标签（可选）
            
        返回:
            测试结果字典
        """
        results = {}
        
        # 训练集测试
        results['train'] = self._test_single_dataset(
            y_train,
            'train'
        )
        
        # 测试集测试
        if y_test is not None:
            results['test'] = self._test_single_dataset(
                y_test,
                'test'
            )
            
            # 一致性测试
            results['consistency'] = self._test_consistency(
                y_train, y_test
            )
        
        # 整体判断
        all_passed = all(
            test_result.get('passed', False)
            for test_result in results.values()
            if isinstance(test_result, dict) and 'passed' in test_result
        )
        
        results['overall_passed'] = all_passed
        results['overall_status'] = 'passed' if all_passed else 'failed'
        
        self.results = results
        
        return results
    
    def _test_single_dataset(
        self,
        y: np.ndarray,
        dataset_name: str
    ) -> Dict[str, Any]:
        """
        测试单个数据集
        
        参数:
            y: 标签数组
            dataset_name: 数据集名称
            
        返回:
            测试结果字典
        """
        n_samples = len(y)
        n_positive = int(y.sum())
        positive_ratio = n_positive / n_samples if n_samples > 0 else 0
        
        # 测试1: 样本数检查
        samples_passed = n_samples >= self.min_positive_samples * 10
        
        # 测试2: 正样本比例检查
        ratio_diff = abs(positive_ratio - self.expected_positive_ratio)
        ratio_passed = ratio_diff <= self.ratio_tolerance
        
        # 测试3: 标签分布检查（使用卡方检验）
        unique, counts = np.unique(y, return_counts=True)
        chi2_passed = len(unique) == 2  # 必须有正负两类
        
        # 综合判断
        passed = samples_passed and ratio_passed and chi2_passed
        
        return {
            'dataset': dataset_name,
            'n_samples': int(n_samples),
            'n_positive': n_positive,
            'positive_ratio': float(positive_ratio),
            'expected_ratio': float(self.expected_positive_ratio),
            'ratio_diff': float(ratio_diff),
            'samples_passed': samples_passed,
            'ratio_passed': ratio_passed,
            'chi2_passed': chi2_passed,
            'passed': passed,
            'issues': self._identify_issues(
                samples_passed,
                ratio_passed,
                chi2_passed
            )
        }
    
    def _test_consistency(
        self,
        y_train: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, Any]:
        """
        测试训练集和测试集的一致性
        
        参数:
            y_train: 训练集标签
            y_test: 测试集标签
            
        返回:
            一致性测试结果
        """
        train_ratio = y_train.mean()
        test_ratio = y_test.mean()
        
        ratio_diff = abs(train_ratio - test_ratio)
        
        # 使用卡方检验检查分布一致性
        observed = np.array([
            [y_train.sum(), (1 - y_train).sum()],
            [y_test.sum(), (1 - y_test).sum()]
        ])
        
        chi2, p_value, dof, expected = chi2_contingency(observed)
        
        # p值>0.05表示分布无显著差异
        consistency_passed = p_value > 0.05
        
        return {
            'train_ratio': float(train_ratio),
            'test_ratio': float(test_ratio),
            'ratio_diff': float(ratio_diff),
            'chi2_statistic': float(chi2),
            'p_value': float(p_value),
            'consistency_passed': consistency_passed,
            'passed': consistency_passed
        }
    
    def _identify_issues(
        self,
        samples_passed: bool,
        ratio_passed: bool,
        chi2_passed: bool
    ) -> List[str]:
        """
        识别问题
        
        参数:
            samples_passed: 样本数是否通过
            ratio_passed: 比例是否通过
            chi2_passed: 卡方检验是否通过
            
        返回:
            问题列表
        """
        issues = []
        
        if not samples_passed:
            issues.append("样本数不足")
        
        if not ratio_passed:
            issues.append(f"正样本比例超出容忍度（预期{self.expected_positive_ratio:.2%}）")
        
        if not chi2_passed:
            issues.append("标签分布异常（缺少正类或负类）")
        
        return issues
    
    def generate_report(self, save_path: Optional[str] = None) -> str:
        """
        生成标签稳定性测试报告
        
        参数:
            save_path: 保存路径（可选）
            
        返回:
            报告字符串
        """
        if self.results is None:
            raise ValueError("请先执行测试")
        
        results = self.results
        
        report_lines = []
        
        report_lines.append("# 标签稳定性测试报告")
        report_lines.append("\n## 一、总体结果")
        report_lines.append(f"- **状态**: {'✅ 通过' if results['overall_passed'] else '❌ 失败'}")
        
        # 训练集结果
        report_lines.append("\n## 二、训练集测试")
        train_result = results['train']
        report_lines.append(f"- 样本数: {train_result['n_samples']}")
        report_lines.append(f"- 正样本数: {train_result['n_positive']}")
        report_lines.append(f"- 正样本比例: {train_result['positive_ratio']:.2%}")
        report_lines.append(f"- 状态: {'✅' if train_result['passed'] else '❌'}")
        
        if train_result['issues']:
            report_lines.append("\n**问题**:")
            for issue in train_result['issues']:
                report_lines.append(f"- {issue}")
        
        # 测试集结果
        if 'test' in results:
            report_lines.append("\n## 三、测试集测试")
            test_result = results['test']
            report_lines.append(f"- 样本数: {test_result['n_samples']}")
            report_lines.append(f"- 正样本数: {test_result['n_positive']}")
            report_lines.append(f"- 正样本比例: {test_result['positive_ratio']:.2%}")
            report_lines.append(f"- 状态: {'✅' if test_result['passed'] else '❌'}")
            
            if test_result['issues']:
                report_lines.append("\n**问题**:")
                for issue in test_result['issues']:
                    report_lines.append(f"- {issue}")
        
        # 一致性测试
        if 'consistency' in results:
            report_lines.append("\n## 四、一致性测试")
            consistency_result = results['consistency']
            report_lines.append(f"- 训练集比例: {consistency_result['train_ratio']:.2%}")
            report_lines.append(f"- 测试集比例: {consistency_result['test_ratio']:.2%}")
            report_lines.append(f"- 比例差异: {consistency_result['ratio_diff']:.2%}")
            report_lines.append(f"- 卡方检验p值: {consistency_result['p_value']:.4f}")
            report_lines.append(f"- 状态: {'✅' if consistency_result['passed'] else '❌'}")
        
        # 建议
        if not results['overall_passed']:
            report_lines.append("\n## 五、建议")
            report_lines.append("1. 检查标签生成逻辑")
            report_lines.append("2. 调整数据采样策略")
            report_lines.append("3. 确保训练集和测试集分布一致")
        
        report = "\n".join(report_lines)
        
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report)
        
        return report


class DataQualityChecker:
    """
    数据质量检查器
    
    综合检查数据质量，包括对抗性验证和标签稳定性测试
    """
    
    def __init__(
        self,
        auc_threshold: float = 0.6,
        expected_positive_ratio: float = 0.1,
        ratio_tolerance: float = 0.02
    ):
        """
        初始化数据质量检查器
        
        参数:
            auc_threshold: 对抗性验证AUC阈值
            expected_positive_ratio: 预期正样本比例
            ratio_tolerance: 比例容忍度
        """
        self.adversarial_validator = AdversarialValidation(
            auc_threshold=auc_threshold
        )
        self.label_stability_tester = LabelStabilityTest(
            expected_positive_ratio=expected_positive_ratio,
            ratio_tolerance=ratio_tolerance
        )
        
        self.results = None
    
    def check(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_test: pd.DataFrame,
        y_test: np.ndarray
    ) -> Dict[str, Any]:
        """
        执行综合数据质量检查
        
        参数:
            X_train: 训练集特征
            y_train: 训练集标签
            X_test: 测试集特征
            y_test: 测试集标签
            
        返回:
            检查结果字典
        """
        results = {}
        
        # 对抗性验证
        print("执行对抗性验证...")
        results['adversarial'] = self.adversarial_validator.validate(
            X_train, X_test
        )
        
        # 标签稳定性测试
        print("执行标签稳定性测试...")
        results['label_stability'] = self.label_stability_tester.test(
            y_train, y_test
        )
        
        # 综合判断
        adversarial_passed = results['adversarial']['status'] == 'no_leak'
        label_stability_passed = results['label_stability']['overall_passed']
        
        results['overall_passed'] = adversarial_passed and label_stability_passed
        results['overall_status'] = 'passed' if results['overall_passed'] else 'failed'
        
        self.results = results
        
        return results
    
    def generate_comprehensive_report(
        self,
        save_path: Optional[str] = None
    ) -> str:
        """
        生成综合报告
        
        参数:
            save_path: 保存路径（可选）
            
        返回:
            报告字符串
        """
        if self.results is None:
            raise ValueError("请先执行检查")
        
        results = self.results
        
        report_lines = []
        
        report_lines.append("# 数据质量检查综合报告")
        report_lines.append("\n## 一、总体结果")
        report_lines.append(f"- **状态**: {'✅ 通过' if results['overall_passed'] else '❌ 失败'}")
        
        # 对抗性验证结果
        report_lines.append("\n## 二、对抗性验证")
        adversarial_result = results['adversarial']
        report_lines.append(f"- **状态**: {adversarial_result['status']}")
        report_lines.append(f"- **消息**: {adversarial_result['message']}")
        report_lines.append(f"- **AUC**: {adversarial_result['mean_auc']:.4f}")
        
        # 标签稳定性测试结果
        report_lines.append("\n## 三、标签稳定性测试")
        label_result = results['label_stability']
        report_lines.append(f"- **状态**: {'✅ 通过' if label_result['overall_passed'] else '❌ 失败'}")
        
        # 建议
        if not results['overall_passed']:
            report_lines.append("\n## 四、建议")
            
            if adversarial_result['status'] != 'no_leak':
                report_lines.append("1. 检查对抗性验证报告，移除泄露特征")
            
            if not label_result['overall_passed']:
                report_lines.append("2. 检查标签稳定性测试报告，修复标签问题")
            
            report_lines.append("3. 重新划分数据集")
            report_lines.append("4. 确保没有数据泄露")
        
        report = "\n".join(report_lines)
        
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report)
        
        return report
