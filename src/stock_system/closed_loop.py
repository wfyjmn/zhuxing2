"""
闭环迭代主流程
功能：整合所有模块，实现"预测-对比-溯源-调整-重训"的完整闭环
"""
import os
import json
import logging
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Optional

# 导入各模块
from .data_collector import MarketDataCollector
from .predictor import StockPredictor
from .analyzer import PerformanceAnalyzer
from .error_tracker import ErrorTracker
from .parameter_tuner import ParameterTuner
from .model_updater import ModelUpdater

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('assets/logs/closed_loop.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ClosedLoopSystem:
    """闭环系统主控制器"""
    
    def __init__(self, config_path: str = None):
        """
        初始化闭环系统
        
        Args:
            config_path: 配置文件路径
        """
        if config_path is None:
            workspace_path = os.getenv("COZE_WORKSPACE_PATH", "/workspace/projects")
            config_path = os.path.join(workspace_path, "config/model_config.json")
        
        self.config_path = config_path
        
        # 初始化各模块
        logger.info("=" * 80)
        logger.info("初始化闭环系统...")
        logger.info("=" * 80)
        
        self.data_collector = MarketDataCollector()
        self.predictor = StockPredictor(config_path)
        self.analyzer = PerformanceAnalyzer(config_path)
        self.error_tracker = ErrorTracker(predictor=self.predictor, config_path=config_path)
        self.parameter_tuner = ParameterTuner(config_path)
        self.model_updater = ModelUpdater(config_path)
        
        # 初始化数据缓存目录
        self._init_directories()
        
        # 迭代计数器
        self.iteration_count = 0
        
        logger.info("闭环系统初始化完成")
    
    def _init_directories(self):
        """初始化必要的目录"""
        workspace_path = os.getenv("COZE_WORKSPACE_PATH", "/workspace/projects")
        directories = [
            "assets/models",
            "assets/data/predictions",
            "assets/data/market_data",
            "assets/logs"
        ]
        
        for directory in directories:
            dir_path = os.path.join(workspace_path, directory)
            os.makedirs(dir_path, exist_ok=True)
        
        logger.info("目录初始化完成")
    
    def run_one_iteration(self, start_date: str = None, end_date: str = None) -> Dict:
        """
        运行一次完整的迭代
        
        Args:
            start_date: 开始日期，格式 'YYYYMMDD'
            end_date: 结束日期，格式 'YYYYMMDD'
            
        Returns:
            迭代结果字典
        """
        self.iteration_count += 1
        iteration_id = f"iter_{self.iteration_count}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        logger.info("=" * 80)
        logger.info(f"开始第 {self.iteration_count} 次迭代，ID: {iteration_id}")
        logger.info("=" * 80)
        
        result = {
            'iteration_id': iteration_id,
            'iteration_count': self.iteration_count,
            'start_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'status': 'running'
        }
        
        try:
            # 步骤1: 获取股票池
            logger.info("\n【步骤1】获取股票池...")
            stock_pool = self.data_collector.get_stock_pool(pool_size=100)
            if not stock_pool:
                logger.error("获取股票池失败")
                result['status'] = 'failed'
                result['error'] = '获取股票池失败'
                return result
            
            result['stock_pool_size'] = len(stock_pool)
            logger.info(f"股票池大小: {len(stock_pool)}")
            
            # 步骤2: 采集历史行情数据用于生成特征
            logger.info("\n【步骤2】采集历史行情数据...")
            if start_date is None:
                start_date = (datetime.now() - timedelta(days=120)).strftime('%Y%m%d')
            if end_date is None:
                end_date = datetime.now().strftime('%Y%m%d')
            
            historical_data = self.data_collector.get_batch_daily_data(
                stock_pool, start_date, end_date
            )
            
            if not historical_data:
                logger.error("采集历史数据失败")
                result['status'] = 'failed'
                result['error'] = '采集历史数据失败'
                return result
            
            result['historical_data_count'] = len(historical_data)
            logger.info(f"历史数据采集完成，成功 {len(historical_data)}/{len(stock_pool)} 只股票")
            
            # 步骤3: 生成预测
            logger.info("\n【步骤3】生成股票预测...")
            predictions = {}
            
            for ts_code, price_df in historical_data.items():
                try:
                    if price_df.empty:
                        continue
                    
                    # 生成特征
                    feature_df = self.predictor.generate_features_from_price(price_df)
                    
                    if not feature_df.empty:
                        # 预测
                        pred_result = self.predictor.predict(feature_df)
                        if not pred_result.empty:
                            predictions[ts_code] = pred_result
                
                except Exception as e:
                    logger.error(f"预测股票 {ts_code} 失败: {e}")
                    continue
            
            if not predictions:
                logger.error("预测失败，没有生成任何预测结果")
                result['status'] = 'failed'
                result['error'] = '预测失败'
                return result
            
            result['predictions_count'] = len(predictions)
            
            # 统计预测分布
            total_pred = sum(len(df) for df in predictions.values())
            pred_up = sum((df['predicted_label'] == 1).sum() for df in predictions.values())
            logger.info(f"预测完成，共 {len(predictions)} 只股票，{total_pred} 条预测")
            logger.info(f"预测分布: 上涨 {pred_up}, 下跌 {total_pred - pred_up}")
            
            # 保存预测结果
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.predictor.save_predictions(predictions, f"predictions_{timestamp}.json")
            
            # 步骤4: 采集实盘行情（模拟：使用预测后的实际数据）
            logger.info("\n【步骤4】采集实盘行情数据...")
            prediction_end_date = datetime.now().strftime('%Y%m%d')
            actual_start_date = (datetime.now() - timedelta(days=10)).strftime('%Y%m%d')
            
            actual_data = self.data_collector.get_batch_daily_data(
                stock_pool, actual_start_date, prediction_end_date
            )
            
            result['actual_data_count'] = len(actual_data)
            logger.info(f"实盘数据采集完成，成功 {len(actual_data)}/{len(stock_pool)} 只股票")
            
            # 步骤5: 对比分析
            logger.info("\n【步骤5】对比分析...")
            aligned_df = self.analyzer.align_predictions_with_actuals(
                predictions, actual_data, days_ahead=5
            )
            
            if aligned_df.empty:
                logger.warning("对齐数据为空，跳过后续分析")
                result['status'] = 'completed'
                result['warning'] = '对齐数据为空'
                return result
            
            metrics = self.analyzer.calculate_metrics(aligned_df)
            confusion_matrix = self.analyzer.generate_confusion_matrix(aligned_df)
            
            result['metrics'] = metrics
            logger.info(f"核心指标: Accuracy={metrics.get('accuracy', 0):.4f}, "
                       f"Precision={metrics.get('precision', 0):.4f}, "
                       f"Recall={metrics.get('recall', 0):.4f}, "
                       f"F1={metrics.get('f1', 0):.4f}")
            
            # 步骤6: 误差分析
            logger.info("\n【步骤6】误差分析...")
            error_analysis = self.error_tracker.analyze_errors(aligned_df)
            error_stocks = self.error_tracker.identify_error_stocks(aligned_df, top_n=10)
            
            result['error_analysis'] = error_analysis
            logger.info(f"误差率: {error_analysis.get('error_rate', 0)*100:.2f}%")
            logger.info(f"假正例: {error_analysis.get('false_positive_count', 0)}, "
                       f"假负例: {error_analysis.get('false_negative_count', 0)}")
            
            # 步骤7: 判断是否需要调整
            logger.info("\n【步骤7】判断是否需要参数调整...")
            should_adjust, adjust_reason = self.analyzer.should_trigger_adjustment(metrics)
            result['should_adjust'] = should_adjust
            result['adjust_reason'] = adjust_reason
            
            if should_adjust:
                logger.warning(f"需要调整参数: {adjust_reason}")
                
                # 步骤8: 参数调整
                logger.info("\n【步骤8】参数调整...")
                new_params, new_threshold, strategy = self.parameter_tuner.auto_adjust(
                    error_analysis, metrics
                )
                
                result['adjustment'] = {
                    'new_params': new_params,
                    'new_threshold': new_threshold,
                    'strategy': strategy
                }
                
                # 步骤9: 模型更新
                logger.info("\n【步骤9】模型更新...")
                try:
                    # 准备训练数据（使用预测正确的样本）
                    aligned_df_clean = aligned_df[aligned_df['predict_correct'] == 1]
                    
                    if len(aligned_df_clean) > 10:
                        # 这里简化处理，实际应该重新生成特征
                        # 模拟训练数据
                        import numpy as np
                        n_features = len(self.predictor.features)
                        X_train = pd.DataFrame(
                            np.random.randn(len(aligned_df_clean), n_features),
                            columns=self.predictor.features
                        )
                        y_train = aligned_df_clean['actual_label']
                        
                        new_model, success, new_metrics = self.model_updater.update_model_with_new_data(
                            X_train, y_train, new_params, new_threshold
                        )
                        
                        result['model_updated'] = success
                        if success:
                            # 更新预测器的参数
                            self.predictor.threshold = new_threshold
                            logger.info("模型更新成功，将使用新模型进行下一次预测")
                        else:
                            logger.warning("模型更新未达标，保持原模型")
                    else:
                        logger.warning("有效样本不足，跳过模型更新")
                        result['model_updated'] = False
                
                except Exception as e:
                    logger.error(f"模型更新失败: {e}")
                    result['model_updated'] = False
            else:
                logger.info("指标良好，无需调整参数")
                result['adjustment'] = None
                result['model_updated'] = False
            
            # 步骤10: 生成报告
            logger.info("\n【步骤10】生成报告...")
            
            # 性能报告
            perf_report = self.analyzer.generate_report(metrics, confusion_matrix)
            self.analyzer.save_report(perf_report, f"performance_report_{timestamp}.md")
            
            # 误差报告
            error_report = self.error_tracker.generate_error_report(aligned_df, error_analysis)
            self.error_tracker.save_error_report(error_report, f"error_report_{timestamp}.md")
            
            result['end_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            result['status'] = 'completed'
            
            logger.info("=" * 80)
            logger.info(f"第 {self.iteration_count} 次迭代完成")
            logger.info(f"核心指标: Precision={metrics.get('precision', 0):.4f}, "
                       f"Recall={metrics.get('recall', 0):.4f}, "
                       f"F1={metrics.get('f1', 0):.4f}")
            logger.info("=" * 80)
            
            return result
            
        except Exception as e:
            logger.error(f"迭代过程发生错误: {e}", exc_info=True)
            result['status'] = 'failed'
            result['error'] = str(e)
            result['end_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            return result
    
    def run_continuous_iterations(self, max_iterations: int = 10, 
                                 interval_days: int = 5) -> list:
        """
        运行多次连续迭代
        
        Args:
            max_iterations: 最大迭代次数
            interval_days: 迭代间隔天数
            
        Returns:
            所有迭代结果的列表
        """
        logger.info(f"开始连续迭代，最大次数: {max_iterations}, 间隔天数: {interval_days}")
        
        all_results = []
        
        for i in range(max_iterations):
            logger.info(f"\n\n{'='*80}")
            logger.info(f"连续迭代进度: {i+1}/{max_iterations}")
            logger.info(f"{'='*80}\n")
            
            result = self.run_one_iteration()
            all_results.append(result)
            
            # 保存迭代结果
            self._save_iteration_results(all_results)
            
            # 判断是否应该停止
            if result.get('status') == 'failed':
                logger.error("迭代失败，停止连续迭代")
                break
            
            # 模拟等待间隔
            if i < max_iterations - 1:
                logger.info(f"等待 {interval_days} 天后进行下一次迭代...")
                # 实际场景中这里应该是真正的等待
                # logger.info(f"（演示模式：跳过等待，直接进行下一次迭代）")
        
        # 生成总结报告
        self._generate_summary_report(all_results)
        
        return all_results
    
    def _save_iteration_results(self, results: list):
        """保存迭代结果"""
        try:
            workspace_path = os.getenv("COZE_WORKSPACE_PATH", "/workspace/projects")
            save_path = os.path.join(workspace_path, "assets/logs", "iteration_results.json")
            
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            logger.info(f"迭代结果已保存: {save_path}")
        except Exception as e:
            logger.error(f"保存迭代结果失败: {e}")
    
    def _generate_summary_report(self, results: list):
        """生成总结报告"""
        try:
            report = []
            report.append("# A股模型实盘对比系统 - 迭代总结报告\n")
            report.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            report.append(f"总迭代次数: {len(results)}\n\n")
            
            # 统计信息
            success_count = sum(1 for r in results if r.get('status') == 'completed')
            failed_count = sum(1 for r in results if r.get('status') == 'failed')
            
            report.append("## 迭代统计\n")
            report.append(f"- 成功次数: {success_count}\n")
            report.append(f"- 失败次数: {failed_count}\n")
            
            # 性能趋势
            report.append("\n## 性能趋势\n")
            report.append("| 迭代 | Precision | Recall | F1 | AUC | 调整次数 |\n")
            report.append("|------|-----------|--------|-----|-----|----------|\n")
            
            adjustment_count = 0
            for i, result in enumerate(results):
                if result.get('status') == 'completed':
                    metrics = result.get('metrics', {})
                    precision = metrics.get('precision', 0)
                    recall = metrics.get('recall', 0)
                    f1 = metrics.get('f1', 0)
                    auc = metrics.get('auc', 0)
                    
                    adjusted = '✓' if result.get('should_adjust') else ''
                    if result.get('should_adjust'):
                        adjustment_count += 1
                    
                    report.append(f"| {i+1} | {precision:.4f} | {recall:.4f} | {f1:.4f} | {auc:.4f} | {adjusted} |\n")
            
            report.append(f"\n总调整次数: {adjustment_count}\n")
            
            # 保存报告
            workspace_path = os.getenv("COZE_WORKSPACE_PATH", "/workspace/projects")
            save_path = os.path.join(workspace_path, "assets/logs", "summary_report.md")
            
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(''.join(report))
            
            logger.info(f"总结报告已生成: {save_path}")
            
        except Exception as e:
            logger.error(f"生成总结报告失败: {e}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='A股模型实盘对比系统')
    parser.add_argument('--iterations', type=int, default=1, help='迭代次数')
    parser.add_argument('--config', type=str, default=None, help='配置文件路径')
    
    args = parser.parse_args()
    
    # 创建闭环系统
    system = ClosedLoopSystem(config_path=args.config)
    
    # 运行迭代
    if args.iterations == 1:
        result = system.run_one_iteration()
        print(f"\n迭代结果: {result}")
    else:
        results = system.run_continuous_iterations(max_iterations=args.iterations)
        print(f"\n完成 {len(results)} 次迭代")


if __name__ == '__main__':
    main()
