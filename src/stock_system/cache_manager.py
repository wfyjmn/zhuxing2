"""
数据缓存管理器
功能：统一管理各类数据的缓存
"""
import os
import json
import logging
import pickle
from datetime import datetime, timedelta
from typing import Dict, Optional, Any

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CacheManager:
    """缓存管理器"""
    
    def __init__(self, cache_root: str = None):
        """
        初始化缓存管理器
        
        Args:
            cache_root: 缓存根目录
        """
        if cache_root is None:
            workspace_path = os.getenv("COZE_WORKSPACE_PATH", "/workspace/projects")
            cache_root = os.path.join(workspace_path, "assets/cache")
        
        self.cache_root = cache_root
        self._init_cache_dirs()
        
    def _init_cache_dirs(self):
        """初始化缓存目录结构"""
        dirs = [
            'predictions',
            'market_data',
            'models',
            'metrics',
            'errors',
            'logs'
        ]
        
        for dir_name in dirs:
            dir_path = os.path.join(self.cache_root, dir_name)
            os.makedirs(dir_path, exist_ok=True)
        
        logger.info(f"缓存目录初始化完成: {self.cache_root}")
    
    def save(self, data: Any, cache_type: str, key: str, 
             format: str = 'json', ttl_days: int = 30) -> str:
        """
        保存数据到缓存
        
        Args:
            data: 要缓存的数据
            cache_type: 缓存类型（predictions/market_data/models/metrics/errors/logs）
            key: 缓存键
            format: 数据格式（json/pickle/csv）
            ttl_days: 保存天数
            
        Returns:
            缓存文件路径
        """
        try:
            # 确保缓存类型目录存在
            cache_dir = os.path.join(self.cache_root, cache_type)
            os.makedirs(cache_dir, exist_ok=True)
            
            # 生成文件名
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{key}_{timestamp}.{format}"
            file_path = os.path.join(cache_dir, filename)
            
            # 根据格式保存
            if format == 'json':
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
            
            elif format == 'pickle':
                with open(file_path, 'wb') as f:
                    pickle.dump(data, f)
            
            elif format == 'csv':
                import pandas as pd
                if isinstance(data, pd.DataFrame):
                    data.to_csv(file_path, index=False, encoding='utf-8')
                else:
                    raise ValueError("CSV格式只能用于DataFrame数据")
            
            else:
                raise ValueError(f"不支持的格式: {format}")
            
            # 记录元数据
            self._save_metadata(cache_type, key, filename, ttl_days)
            
            logger.debug(f"数据缓存成功: {file_path}")
            return file_path
        
        except Exception as e:
            logger.error(f"保存缓存失败: {e}")
            raise
    
    def load(self, cache_type: str, key: str, 
             format: str = 'json', latest: bool = True) -> Optional[Any]:
        """
        从缓存加载数据
        
        Args:
            cache_type: 缓存类型
            key: 缓存键
            format: 数据格式
            latest: 是否加载最新版本
            
        Returns:
            缓存的数据
        """
        try:
            cache_dir = os.path.join(self.cache_root, cache_type)
            
            # 查找缓存文件
            cache_files = self._find_cache_files(cache_dir, key, format)
            
            if not cache_files:
                logger.debug(f"未找到缓存: {cache_type}/{key}")
                return None
            
            # 选择文件
            if latest:
                file_path = cache_files[-1]  # 最新
            else:
                file_path = cache_files[0]  # 最旧
            
            # 检查是否过期
            if not self._is_cache_valid(file_path):
                logger.info(f"缓存已过期: {file_path}")
                os.remove(file_path)
                return None
            
            # 根据格式加载
            if format == 'json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            
            elif format == 'pickle':
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
            
            elif format == 'csv':
                import pandas as pd
                data = pd.read_csv(file_path)
            
            else:
                raise ValueError(f"不支持的格式: {format}")
            
            logger.debug(f"数据加载成功: {file_path}")
            return data
        
        except Exception as e:
            logger.error(f"加载缓存失败: {e}")
            return None
    
    def _find_cache_files(self, cache_dir: str, key: str, format: str) -> list:
        """查找缓存文件"""
        import os
        import re
        
        if not os.path.exists(cache_dir):
            return []
        
        pattern = re.compile(f"{key}_\\d{{8}}_\\d{{6}}\\.{format}$")
        cache_files = []
        
        for filename in os.listdir(cache_dir):
            if pattern.match(filename):
                file_path = os.path.join(cache_dir, filename)
                cache_files.append(file_path)
        
        # 按文件修改时间排序
        cache_files.sort(key=lambda x: os.path.getmtime(x))
        
        return cache_files
    
    def _is_cache_valid(self, file_path: str, ttl_days: int = 30) -> bool:
        """检查缓存是否有效"""
        if not os.path.exists(file_path):
            return False
        
        # 检查文件修改时间
        file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
        expire_time = file_time + timedelta(days=ttl_days)
        
        return datetime.now() < expire_time
    
    def _save_metadata(self, cache_type: str, key: str, filename: str, ttl_days: int):
        """保存缓存元数据"""
        try:
            metadata_file = os.path.join(self.cache_root, cache_type, '.metadata.json')
            
            metadata = {}
            if os.path.exists(metadata_file):
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
            
            metadata[key] = {
                'filename': filename,
                'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'ttl_days': ttl_days,
                'expires_at': (datetime.now() + timedelta(days=ttl_days)).strftime('%Y-%m-%d %H:%M:%S')
            }
            
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        except Exception as e:
            logger.error(f"保存元数据失败: {e}")
    
    def clear_expired_cache(self, cache_type: str = None):
        """清理过期缓存"""
        try:
            if cache_type:
                cache_dirs = [os.path.join(self.cache_root, cache_type)]
            else:
                cache_dirs = [
                    os.path.join(self.cache_root, d) 
                    for d in os.listdir(self.cache_root) 
                    if os.path.isdir(os.path.join(self.cache_root, d))
                ]
            
            total_cleaned = 0
            for cache_dir in cache_dirs:
                for filename in os.listdir(cache_dir):
                    if filename.startswith('.'):
                        continue
                    
                    file_path = os.path.join(cache_dir, filename)
                    if not self._is_cache_valid(file_path):
                        os.remove(file_path)
                        total_cleaned += 1
                        logger.info(f"删除过期缓存: {file_path}")
            
            logger.info(f"清理过期缓存完成，共删除 {total_cleaned} 个文件")
        
        except Exception as e:
            logger.error(f"清理过期缓存失败: {e}")
    
    def get_cache_stats(self) -> Dict:
        """获取缓存统计信息"""
        try:
            stats = {
                'total_files': 0,
                'total_size': 0,
                'by_type': {}
            }
            
            for cache_type in os.listdir(self.cache_root):
                cache_dir = os.path.join(self.cache_root, cache_type)
                if not os.path.isdir(cache_dir):
                    continue
                
                type_stats = {
                    'file_count': 0,
                    'total_size': 0
                }
                
                for filename in os.listdir(cache_dir):
                    if filename.startswith('.'):
                        continue
                    
                    file_path = os.path.join(cache_dir, filename)
                    if os.path.isfile(file_path):
                        file_size = os.path.getsize(file_path)
                        type_stats['file_count'] += 1
                        type_stats['total_size'] += file_size
                        stats['total_files'] += 1
                        stats['total_size'] += file_size
                
                stats['by_type'][cache_type] = type_stats
            
            # 转换为MB
            stats['total_size_mb'] = stats['total_size'] / (1024 * 1024)
            
            return stats
        
        except Exception as e:
            logger.error(f"获取缓存统计失败: {e}")
            return {}


def test_cache_manager():
    """测试缓存管理器"""
    cache = CacheManager()
    
    print("\n=== 测试缓存管理器 ===")
    
    # 测试保存
    test_data = {
        'test': 'data',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    cache_path = cache.save(test_data, 'logs', 'test_log', format='json')
    print(f"保存路径: {cache_path}")
    
    # 测试加载
    loaded_data = cache.load('logs', 'test_log', format='json')
    print(f"加载数据: {loaded_data}")
    
    # 测试统计
    stats = cache.get_cache_stats()
    print(f"缓存统计: {stats}")


if __name__ == '__main__':
    test_cache_manager()
