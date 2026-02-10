"""
HTML报告生成器
功能：生成完整的HTML训练报告
"""
import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class HTMLReportGenerator:
    """HTML报告生成器"""
    
    def __init__(self, output_dir: str = None):
        """
        初始化报告生成器
        
        Args:
            output_dir: 输出目录
        """
        if output_dir is None:
            workspace_path = os.getenv("COZE_WORKSPACE_PATH", "/workspace/projects")
            output_dir = os.path.join(workspace_path, "assets/reports")
        
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"HTML报告生成器初始化完成，输出目录: {output_dir}")
    
    def generate_report(self, metrics: Dict, parameters: Dict, 
                         overfitting_result: Dict, param_analysis: Dict,
                         image_paths: Dict[str, str],
                         save_path: str = None) -> str:
        """
        生成完整的HTML报告
        
        Args:
            metrics: 模型指标
            parameters: 模型参数
            overfitting_result: 过拟合检测结果
            param_analysis: 参数分析结果
            image_paths: 图片路径字典
            save_path: 保存路径
            
        Returns:
            报告文件路径
        """
        html_content = self._generate_html(
            metrics=metrics,
            parameters=parameters,
            overfitting_result=overfitting_result,
            param_analysis=param_analysis,
            image_paths=image_paths
        )
        
        if save_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            save_path = os.path.join(self.output_dir, f"training_report_{timestamp}.html")
        
        # 同时生成最新的报告
        latest_save_path = os.path.join(self.output_dir, "training_report_latest.html")
        
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        with open(latest_save_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"HTML报告已生成: {save_path}")
        logger.info(f"最新报告已生成: {latest_save_path}")
        
        return save_path
    
    def _generate_html(self, metrics: Dict, parameters: Dict, 
                        overfitting_result: Dict, param_analysis: Dict,
                        image_paths: Dict[str, str]) -> str:
        """生成HTML内容"""
        
        # 计算综合得分
        overall_score = (metrics.get('auc', 0) * 0.4 + 
                        metrics.get('accuracy', 0) * 0.2 +
                        metrics.get('f1', 0) * 0.4)
        
        # 评级
        if overall_score >= 0.8:
            grade = "A (优秀)"
            grade_color = "#28a745"
        elif overall_score >= 0.7:
            grade = "B (良好)"
            grade_color = "#007bff"
        elif overall_score >= 0.6:
            grade = "C (中等)"
            grade_color = "#ffc107"
        else:
            grade = "D (较差)"
            grade_color = "#dc3545"
        
        # 过拟合状态
        if overfitting_result['is_overfitting']:
            overfitting_status = "⚠️ 检测到过拟合"
            overfitting_class = "warning"
        else:
            overfitting_status = "✅ 无过拟合"
            overfitting_class = "success"
        
        html_template = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>A股模型训练报告</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            min-height: 100vh;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 10px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.3);
            padding: 30px;
        }}
        
        h1 {{
            text-align: center;
            color: #333;
            margin-bottom: 10px;
            font-size: 32px;
        }}
        
        .subtitle {{
            text-align: center;
            color: #666;
            margin-bottom: 30px;
            font-size: 14px;
        }}
        
        .summary-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 30px;
        }}
        
        .summary-score {{
            text-align: center;
            font-size: 48px;
            font-weight: bold;
            margin-bottom: 10px;
        }}
        
        .summary-grade {{
            text-align: center;
            font-size: 24px;
            padding: 5px 20px;
            background: white;
            color: {grade_color};
            border-radius: 20px;
            display: inline-block;
            margin: 0 auto 20px;
        }}
        
        .summary-stats {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 15px;
            margin-top: 20px;
        }}
        
        .stat-item {{
            background: rgba(255,255,255,0.2);
            padding: 10px;
            border-radius: 5px;
            text-align: center;
        }}
        
        .stat-value {{
            font-size: 24px;
            font-weight: bold;
        }}
        
        .stat-label {{
            font-size: 12px;
            opacity: 0.9;
        }}
        
        .section {{
            margin-bottom: 30px;
        }}
        
        .section-title {{
            font-size: 20px;
            font-weight: bold;
            color: #333;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 2px solid #667eea;
        }}
        
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 20px;
        }}
        
        .metric-card {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }}
        
        .metric-name {{
            font-size: 14px;
            color: #666;
            margin-bottom: 5px;
        }}
        
        .metric-value {{
            font-size: 28px;
            font-weight: bold;
            color: #333;
        }}
        
        .metric-status {{
            font-size: 12px;
            margin-top: 5px;
        }}
        
        .status-good {{
            color: #28a745;
        }}
        
        .status-warning {{
            color: #ffc107;
        }}
        
        .status-bad {{
            color: #dc3545;
        }}
        
        .param-table {{
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 20px;
        }}
        
        .param-table th, .param-table td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        
        .param-table th {{
            background: #667eea;
            color: white;
        }}
        
        .param-table tr:hover {{
            background: #f5f5f5;
        }}
        
        .alert {{
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 15px;
        }}
        
        .alert-success {{
            background: #d4edda;
            border: 1px solid #c3e6cb;
            color: #155724;
        }}
        
        .alert-warning {{
            background: #fff3cd;
            border: 1px solid #ffeaa7;
            color: #856404;
        }}
        
        .alert-danger {{
            background: #f8d7da;
            border: 1px solid #f5c6cb;
            color: #721c24;
        }}
        
        .image-container {{
            text-align: center;
            margin: 20px 0;
        }}
        
        .image-container img {{
            max-width: 100%;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }}
        
        .image-caption {{
            font-size: 14px;
            color: #666;
            margin-top: 10px;
        }}
        
        .footer {{
            text-align: center;
            padding: 20px;
            color: #666;
            font-size: 12px;
            border-top: 1px solid #ddd;
            margin-top: 30px;
        }}
        
        .badge {{
            display: inline-block;
            padding: 3px 10px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: bold;
        }}
        
        .badge-success {{
            background: #28a745;
            color: white;
        }}
        
        .badge-warning {{
            background: #ffc107;
            color: #333;
        }}
        
        .badge-danger {{
            background: #dc3545;
            color: white;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📈 A股模型训练报告</h1>
        <p class="subtitle">训练时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <!-- 综合评估 -->
        <div class="summary-card">
            <div class="summary-score">综合得分: {overall_score:.4f}</div>
            <div class="summary-grade">{grade}</div>
            <div class="summary-stats">
                <div class="stat-item">
                    <div class="stat-value">{metrics.get('auc', 0):.4f}</div>
                    <div class="stat-label">AUC</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{metrics.get('accuracy', 0):.4f}</div>
                    <div class="stat-label">准确率</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{metrics.get('precision', 0):.4f}</div>
                    <div class="stat-label">精确率</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{metrics.get('recall', 0):.4f}</div>
                    <div class="stat-label">召回率</div>
                </div>
            </div>
        </div>
        
        <!-- 模型性能指标 -->
        <div class="section">
            <h2 class="section-title">📊 模型性能指标</h2>
            <div class="metrics-grid">
                {self._generate_metric_card('AUC', metrics.get('auc', 0), 0.7)}
                {self._generate_metric_card('准确率', metrics.get('accuracy', 0), 0.65)}
                {self._generate_metric_card('精确率', metrics.get('precision', 0), 0.6)}
                {self._generate_metric_card('召回率', metrics.get('recall', 0), 0.6)}
                {self._generate_metric_card('F1分数', metrics.get('f1', 0), 0.65)}
                {self._generate_metric_card('夏普比率', metrics.get('sharpe_ratio', 0), 1.5)}
            </div>
        </div>
        
        <!-- 过拟合检测 -->
        <div class="section">
            <h2 class="section-title">🔍 过拟合检测</h2>
            <div class="alert alert-{overfitting_class}">
                <strong>状态:</strong> {overfitting_status}<br>
                <strong>严重程度:</strong> {overfitting_result['severity'].upper()}
            </div>
            {self._generate_warnings(overfitting_result['warnings'])}
        </div>
        
        <!-- 模型参数 -->
        <div class="section">
            <h2 class="section-title">⚙️ 模型参数配置</h2>
            <table class="param-table">
                <tr>
                    <th>参数名</th>
                    <th>当前值</th>
                    <th>参数分析</th>
                </tr>
                <tr>
                    <td>n_estimators (树的数量)</td>
                    <td>{parameters.get('n_estimators', 'N/A')}</td>
                    <td>{self._get_param_advice('n_estimators', parameters.get('n_estimators', 100))}</td>
                </tr>
                <tr>
                    <td>max_depth (树的深度)</td>
                    <td>{parameters.get('max_depth', 'N/A')}</td>
                    <td>{self._get_param_advice('max_depth', parameters.get('max_depth', 6))}</td>
                </tr>
                <tr>
                    <td>learning_rate (学习率)</td>
                    <td>{parameters.get('learning_rate', 'N/A')}</td>
                    <td>{self._get_param_advice('learning_rate', parameters.get('learning_rate', 0.1))}</td>
                </tr>
                <tr>
                    <td>subsample</td>
                    <td>{parameters.get('subsample', 'N/A')}</td>
                    <td>{self._get_param_advice('subsample', parameters.get('subsample', 0.8))}</td>
                </tr>
                <tr>
                    <td>colsample_bytree</td>
                    <td>{parameters.get('colsample_bytree', 'N/A')}</td>
                    <td>{self._get_param_advice('colsample_bytree', parameters.get('colsample_bytree', 0.8))}</td>
                </tr>
                <tr>
                    <td>reg_alpha (L1正则化)</td>
                    <td>{parameters.get('reg_alpha', 'N/A')}</td>
                    <td>{self._get_param_advice('reg_alpha', parameters.get('reg_alpha', 0))}</td>
                </tr>
                <tr>
                    <td>reg_lambda (L2正则化)</td>
                    <td>{parameters.get('reg_lambda', 'N/A')}</td>
                    <td>{self._get_param_advice('reg_lambda', parameters.get('reg_lambda', 1))}</td>
                </tr>
            </table>
            
            {self._generate_param_suggestions(param_analysis)}
        </div>
        
        <!-- 可视化图表 -->
        <div class="section">
            <h2 class="section-title">📈 可视化图表</h2>
            {self._generate_image_section(image_paths)}
        </div>
        
        <!-- 优化建议 -->
        <div class="section">
            <h2 class="section-title">💡 优化建议</h2>
            {self._generate_optimization_suggestions(metrics, overfitting_result, param_analysis)}
        </div>
        
        <div class="footer">
            <p>此报告由 A股模型实盘对比系统 自动生成</p>
            <p>生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
    </div>
</body>
</html>
"""
        return html_template
    
    def _generate_metric_card(self, name: str, value: float, threshold: float) -> str:
        """生成指标卡片"""
        if value >= threshold:
            status_class = "status-good"
            status_text = "✓ 优秀"
        elif value >= threshold * 0.9:
            status_class = "status-warning"
            status_text = "⚠ 一般"
        else:
            status_class = "status-bad"
            status_text = "✗ 较差"
        
        return f"""
        <div class="metric-card">
            <div class="metric-name">{name}</div>
            <div class="metric-value">{value:.4f}</div>
            <div class="metric-status {status_class}">{status_text}</div>
        </div>
        """
    
    def _generate_warnings(self, warnings: List[str]) -> str:
        """生成警告信息"""
        if not warnings:
            return '<div class="alert alert-success">✓ 暂无警告</div>'
        
        warnings_html = ""
        for warning in warnings:
            warnings_html += f'<div class="alert alert-warning">⚠️ {warning}</div>'
        
        return warnings_html
    
    def _get_param_advice(self, param_name: str, value) -> str:
        """获取参数建议"""
        param_ranges = {
            'n_estimators': {'min': 100, 'max': 300, 'optimal': '100-300'},
            'max_depth': {'min': 5, 'max': 8, 'optimal': '5-8'},
            'learning_rate': {'min': 0.05, 'max': 0.2, 'optimal': '0.05-0.2'},
            'subsample': {'min': 0.7, 'max': 0.9, 'optimal': '0.7-0.9'},
            'colsample_bytree': {'min': 0.7, 'max': 0.9, 'optimal': '0.7-0.9'},
            'reg_alpha': {'min': 0, 'max': 1, 'optimal': '0-1'},
            'reg_lambda': {'min': 1, 'max': 2, 'optimal': '1-2'}
        }
        
        if param_name in param_ranges:
            range_info = param_ranges[param_name]
            if param_name == 'reg_alpha' and value == 0:
                return "建议: 如出现过拟合可适当增加"
            elif param_name == 'reg_lambda' and value == 1:
                return "建议: 如出现过拟合可适当增加"
            elif value < range_info['min'] or value > range_info['max']:
                return f"⚠️ 建议范围: {range_info['optimal']}"
            else:
                return f"✓ 在合理范围内 ({range_info['optimal']})"
        
        return "-"
    
    def _generate_param_suggestions(self, analysis: Dict) -> str:
        """生成参数建议"""
        if not analysis.get('suggestions'):
            return ""
        
        suggestions_html = '<div class="alert alert-info"><strong>参数建议:</strong><ul>'
        for suggestion in analysis['suggestions']:
            suggestions_html += f'<li>{suggestion}</li>'
        suggestions_html += '</ul></div>'
        
        return suggestions_html
    
    def _generate_image_section(self, image_paths: Dict[str, str]) -> str:
        """生成图片部分"""
        image_names = {
            'roc': 'ROC曲线',
            'confusion_matrix': '混淆矩阵',
            'feature_importance': '特征重要性',
            'learning_curve': '学习曲线',
            'pr_curve': '精确率-召回率曲线',
            'prediction_distribution': '预测概率分布',
            'industry_sampling': '行业采样分布',
            'summary_dashboard': '总结仪表盘'
        }
        
        images_html = ""
        for key, path in image_paths.items():
            if key in image_names and os.path.exists(path):
                filename = os.path.basename(path)
                images_html += f"""
                <div class="image-container">
                    <img src="{filename}" alt="{image_names[key]}">
                    <div class="image-caption">{image_names[key]}</div>
                </div>
                """
        
        return images_html if images_html else "<p>暂无可视化图表</p>"
    
    def _generate_optimization_suggestions(self, metrics: Dict, 
                                             overfitting_result: Dict,
                                             param_analysis: Dict) -> str:
        """生成优化建议"""
        suggestions = []
        
        # 根据AUC给建议
        auc_value = metrics.get('auc', 0)
        if auc_value < 0.6:
            suggestions.append("模型欠拟合，建议：")
            suggestions.append("- 增加树的深度（max_depth: 5-8）")
            suggestions.append("- 增加树的数量（n_estimators: 100-300）")
            suggestions.append("- 适当提高学习率（learning_rate: 0.1-0.2）")
        elif overfitting_result['is_overfitting']:
            suggestions.append("模型过拟合，建议：")
            suggestions.append("- 降低树的深度（max_depth: 4-6）")
            suggestions.append("- 增加正则化（reg_alpha > 0 或 reg_lambda > 1）")
            suggestions.append("- 降低学习率，增加树的数量")
            suggestions.append("- 使用 subsample 和 colsample_bytree 进行随机采样")
        else:
            suggestions.append("✓ 模型表现良好，建议：")
            suggestions.append("- 持续监控模型在实盘中的表现")
            suggestions.append("- 定期重新训练模型以适应市场变化")
            suggestions.append("- 关注行业分布的均衡性")
        
        # 根据召回率给建议
        recall_value = metrics.get('recall', 0)
        if recall_value < 0.6:
            suggestions.append("- 召回率偏低，建议降低决策阈值")
        
        suggestions_html = '<ul>'
        for suggestion in suggestions:
            suggestions_html += f'<li>{suggestion}</li>'
        suggestions_html += '</ul>'
        
        return suggestions_html
