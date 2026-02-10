"""
模型更新模块
功能：增量训练、模型缓存管理
"""
import os
import json
import pickle
import logging
import shutil
import numpy as np
import pandas as pd
import xgboost as xgb
from datetime import datetime
from typing import Dict, Optional

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ModelUpdater:
    """模型更新器"""
    
    def __init__(self, config_path: str = None):
        """
        初始化模型更新器
        
        Args:
            config_path: 配置文件路径
        """
        if config_path is None:
            workspace_path = os.getenv("COZE_WORKSPACE_PATH", "/workspace/projects")
            config_path = os.path.join(workspace_path, "config/model_config.json")
        
        self.config = self._load_config(config_path)
        self.model_cache_dir = self._init_cache_dir()
        self.backup_count = self.config['cache']['backup_count']
        
    def _load_config(self, config_path: str) -> Dict:
        """加载配置"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            logger.info(f"加载配置成功")
            return config
        except Exception as e:
            logger.error(f"加载配置失败: {e}")
            return {}
    
    def _init_cache_dir(self) -> str:
        """初始化缓存目录"""
        workspace_path = os.getenv("COZE_WORKSPACE_PATH", "/workspace/projects")
        cache_dir = os.path.join(workspace_path, self.config['cache']['model_cache_dir'])
        os.makedirs(cache_dir, exist_ok=True)
        logger.info(f"模型缓存目录: {cache_dir}")
        return cache_dir
    
    def save_model(self, model, model_metadata: Dict, version: str = None) -> str:
        """
        保存模型
        
        Args:
            model: XGBoost模型
            model_metadata: 模型元数据
            version: 版本号
            
        Returns:
            保存路径
        """
        try:
            if version is None:
                version = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # 保存模型
            model_filename = f"model_{version}.pkl"
            model_path = os.path.join(self.model_cache_dir, model_filename)
            
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            # 保存元数据
            metadata_filename = f"model_metadata_{version}.json"
            metadata_path = os.path.join(self.model_cache_dir, metadata_filename)
            
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(model_metadata, f, ensure_ascii=False, indent=2)
            
            # 更新最新模型链接
            latest_model_path = os.path.join(self.model_cache_dir, "best_model.pkl")
            latest_metadata_path = os.path.join(self.model_cache_dir, "model_metadata.json")
            
            shutil.copy2(model_path, latest_model_path)
            shutil.copy2(metadata_path, latest_metadata_path)
            
            logger.info(f"模型保存成功: {model_path}")
            logger.info(f"元数据保存成功: {metadata_path}")
            
            # 清理旧备份
            self._cleanup_old_backups()
            
            return model_path
        except Exception as e:
            logger.error(f"保存模型失败: {e}")
            raise
    
    def load_model(self, version: str = None) -> tuple:
        """
        加载模型
        
        Args:
            version: 版本号，None表示加载最新模型
            
        Returns:
            (model, metadata)
        """
        try:
            if version is None:
                model_path = os.path.join(self.model_cache_dir, "best_model.pkl")
                metadata_path = os.path.join(self.model_cache_dir, "model_metadata.json")
            else:
                model_path = os.path.join(self.model_cache_dir, f"model_{version}.pkl")
                metadata_path = os.path.join(self.model_cache_dir, f"model_metadata_{version}.json")
            
            # 加载模型
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            # 加载元数据
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            logger.info(f"模型加载成功: {model_path}")
            return model, metadata
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            raise
    
    def incremental_train(self, old_model, new_data: pd.DataFrame, 
                         labels: pd.Series, params: Dict) -> tuple:
        """
        增量训练模型
        
        Args:
            old_model: 旧模型
            new_data: 新数据
            labels: 标签
            params: 训练参数
            
        Returns:
            (新模型, 训练信息)
        """
        try:
            logger.info("开始增量训练...")
            
            # 准备数据
            X = new_data.values
            y = labels.values
            
            dtrain_new = xgb.DMatrix(X, label=y)
            
            # 使用旧模型作为初始化
            new_model = xgb.train(
                params,
                dtrain_new,
                xgb_model=old_model,
                num_boost_round=50,  # 增量训练轮数较少
                evals=[(dtrain_new, 'train')],
                verbose_eval=False
            )
            
            train_info = {
                'training_samples': len(new_data),
                'positive_samples': int(y.sum()),
                'negative_samples': int(len(y) - y.sum()),
                'boost_rounds': 50,
                'params': params
            }
            
            logger.info(f"增量训练完成，训练样本数: {len(new_data)}")
            return new_model, train_info
            
        except Exception as e:
            logger.error(f"增量训练失败: {e}")
            raise
    
    def evaluate_model(self, model, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """
        评估模型
        
        Args:
            model: XGBoost模型
            X_test: 测试数据
            y_test: 测试标签
            
        Returns:
            评估指标
        """
        try:
            dtest = xgb.DMatrix(X_test.values)
            
            # 预测
            probs = model.predict(dtest)
            preds = (probs >= self.config['xgboost']['threshold']).astype(int)
            
            # 计算指标
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
            
            metrics = {
                'accuracy': float(accuracy_score(y_test, preds)),
                'precision': float(precision_score(y_test, preds, zero_division=0)),
                'recall': float(recall_score(y_test, preds, zero_division=0)),
                'f1': float(f1_score(y_test, preds, zero_division=0)),
                'auc': float(roc_auc_score(y_test, probs))
            }
            
            logger.info(f"模型评估完成: {metrics}")
            return metrics
            
        except Exception as e:
            logger.error(f"模型评估失败: {e}")
            return {}
    
    def update_model_with_new_data(self, new_data: pd.DataFrame, labels: pd.Series,
                                   new_params: Dict, new_threshold: float,
                                   old_model=None) -> tuple:
        """
        使用新数据更新模型
        
        Args:
            new_data: 新数据
            labels: 标签
            new_params: 新参数
            new_threshold: 新阈值
            old_model: 旧模型，None则重新训练
            
        Returns:
            (新模型, 是否成功, 评估指标)
        """
        try:
            # 加载旧模型
            if old_model is None:
                old_model, _ = self.load_model()
            
            # 增量训练
            new_model, train_info = self.incremental_train(
                old_model, new_data, labels, new_params
            )
            
            # 评估模型
            metrics = self.evaluate_model(new_model, new_data, labels)
            
            # 判断是否达标
            targets = self.config['performance']['targets']
            all_met = all(
                metrics.get(k, 0) >= v 
                for k, v in targets.items() 
                if k in metrics
            )
            
            if all_met:
                # 达标，保存新模型
                metadata = {
                    'create_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'best_score': metrics.get('f1', 0),
                    'params': new_params,
                    'threshold': new_threshold,
                    'features': self.config['data']['train_features'],
                    'metrics': metrics,
                    'train_info': train_info
                }
                
                self.save_model(new_model, metadata)
                logger.info("新模型达标，已保存")
                return new_model, True, metrics
            else:
                # 未达标，回滚
                logger.warning(f"新模型未达标，指标: {metrics}")
                return old_model, False, metrics
            
        except Exception as e:
            logger.error(f"更新模型失败: {e}")
            return old_model, False, {}
    
    def _cleanup_old_backups(self):
        """清理旧备份"""
        try:
            # 获取所有模型文件
            model_files = [f for f in os.listdir(self.model_cache_dir) 
                          if f.startswith('model_') and f.endswith('.pkl')]
            
            # 按时间排序，保留最新的N个
            model_files.sort(reverse=True)
            
            # 删除多余的备份
            for filename in model_files[self.backup_count:]:
                file_path = os.path.join(self.model_cache_dir, filename)
                os.remove(file_path)
                logger.info(f"删除旧备份: {filename}")
                
        except Exception as e:
            logger.error(f"清理旧备份失败: {e}")
    
    def get_model_history(self) -> list:
        """
        获取模型历史
        
        Returns:
            模型历史列表
        """
        try:
            history = []
            model_files = [f for f in os.listdir(self.model_cache_dir) 
                          if f.startswith('model_') and f.endswith('.json')]
            
            for filename in model_files:
                try:
                    metadata_path = os.path.join(self.model_cache_dir, filename)
                    with open(metadata_path, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                    history.append(metadata)
                except:
                    continue
            
            # 按时间排序
            history.sort(key=lambda x: x.get('create_time', ''), reverse=True)
            
            return history
        except Exception as e:
            logger.error(f"获取模型历史失败: {e}")
            return []
    
    def rollback_to_version(self, version: str) -> tuple:
        """
        回滚到指定版本
        
        Args:
            version: 版本号
            
        Returns:
            (模型, 元数据)
        """
        try:
            model, metadata = self.load_model(version)
            
            # 保存为最新版本
            self.save_model(model, metadata, version="rollback_" + datetime.now().strftime('%Y%m%d_%H%M%S'))
            
            logger.info(f"回滚到版本 {version} 成功")
            return model, metadata
        except Exception as e:
            logger.error(f"回滚失败: {e}")
            raise


def test_model_updater():
    """测试模型更新器"""
    updater = ModelUpdater()
    
    print("\n=== 测试模型保存和加载 ===")
    
    # 创建测试模型
    np.random.seed(42)
    n_samples = 100
    n_features = 10
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, 2, n_samples)
    
    dtrain = xgb.DMatrix(X, label=y)
    params = {
        'objective': 'binary:logistic',
        'max_depth': 3,
        'learning_rate': 0.1
    }
    
    model = xgb.train(params, dtrain, num_boost_round=10)
    
    # 保存模型
    metadata = {
        'create_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'best_score': 0.75,
        'params': params,
        'threshold': 0.4,
        'features': [f'feature_{i}' for i in range(n_features)]
    }
    
    saved_path = updater.save_model(model, metadata)
    print(f"模型保存路径: {saved_path}")
    
    # 加载模型
    loaded_model, loaded_metadata = updater.load_model()
    print(f"加载的元数据: {loaded_metadata}")
    
    # 测试增量训练
    print("\n=== 测试增量训练 ===")
    new_data = pd.DataFrame(np.random.randn(50, n_features), 
                           columns=[f'feature_{i}' for i in range(n_features)])
    new_labels = pd.Series(np.random.randint(0, 2, 50))
    
    new_model, train_info = updater.incremental_train(
        model, new_data, new_labels, params
    )
    print(f"增量训练信息: {train_info}")
    
    # 评估模型
    print("\n=== 测试模型评估 ===")
    metrics = updater.evaluate_model(new_model, new_data, new_labels)
    print(f"评估指标: {metrics}")
    
    # 测试模型历史
    print("\n=== 测试模型历史 ===")
    history = updater.get_model_history()
    print(f"模型历史数量: {len(history)}")


if __name__ == '__main__':
    test_model_updater()
