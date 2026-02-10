# 环境变量配置指南

## 概述

为了提升系统安全性，tushare token 现在通过环境变量配置，不再硬编码在配置文件中。

## 配置步骤

### 方法一：使用 .env 文件（推荐）

1. **复制环境变量模板**
   ```bash
   cp .env.example .env
   ```

2. **编辑 .env 文件**
   ```bash
   nano .env
   # 或使用其他编辑器
   ```

3. **填入实际配置值**
   ```env
   TUSHARE_TOKEN=8f5cd68a38bb5bd3fe035ff544bc8c71c6c97e70b081d9a58f8d0bd7
   ```

4. **加载环境变量**
   ```bash
   # 方法1：在启动脚本中加载
   source .env
   python run_system.py

   # 方法2：使用 python-dotenv（推荐）
   # 系统已自动集成，无需手动加载
   ```

### 方法二：系统环境变量

**Linux/Mac:**
```bash
# 临时设置（当前会话有效）
export TUSHARE_TOKEN=8f5cd68a38bb5bd3fe035ff544bc8c71c6c97e70b081d9a58f8d0bd7

# 永久设置（写入 ~/.bashrc 或 ~/.zshrc）
echo 'export TUSHARE_TOKEN=8f5cd68a38bb5bd3fe035ff544bc8c71c6c97e70b081d9a58f8d0bd7' >> ~/.bashrc
source ~/.bashrc
```

**Windows (PowerShell):**
```powershell
# 临时设置
$env:TUSHARE_TOKEN="8f5cd68a38bb5bd3fe035ff544bc8c71c6c97e70b081d9a58f8d0bd7"

# 永久设置（系统环境变量）
setx TUSHARE_TOKEN "8f5cd68a38bb5bd3fe035ff544bc8c71c6c97e70b081d9a58f8d0bd7"
```

### 方法三：IDE 配置

**VS Code:**
在 `.vscode/launch.json` 中配置：
```json
{
  "configurations": [
    {
      "name": "Python: 运行系统",
      "type": "python",
      "request": "launch",
      "program": "run_system.py",
      "env": {
        "TUSHARE_TOKEN": "8f5cd68a38bb5bd3fe035ff544bc8c71c6c97e70b081d9a58f8d0bd7"
      }
    }
  ]
}
```

**PyCharm:**
1. Run -> Edit Configurations
2. 选择你的运行配置
3. 在 Environment variables 中添加：
   ```
   TUSHARE_TOKEN=8f5cd68a38bb5bd3fe035ff544bc8c71c6c97e70b081d9a58f8d0bd7
   ```

## 环境变量说明

| 变量名 | 必需 | 说明 | 默认值 |
|--------|------|------|--------|
| TUSHARE_TOKEN | 是 | Tushare API Token | 无 |
| COZE_WORKSPACE_PATH | 否 | 工作空间路径 | /workspace/projects |
| LOG_LEVEL | 否 | 日志级别 | INFO |
| MAX_WORKERS | 否 | 最大线程数 | 5 |
| REQUEST_TIMEOUT | 否 | 请求超时时间（秒） | 30 |
| CACHE_EXPIRY_HOURS | 否 | 缓存过期时间（小时） | 24 |

## 安全建议

1. **不要提交 .env 文件到版本控制**
   在 `.gitignore` 中添加：
   ```
   .env
   ```

2. **定期更换 Token**
   定期在 tushare.pro 重新生成 token

3. **使用不同的环境配置**
   - 开发环境：使用测试 token
   - 生产环境：使用正式 token
   - 文件命名：`.env.development`, `.env.production`

4. **权限控制**
   ```bash
   # 设置 .env 文件权限，仅当前用户可读写
   chmod 600 .env
   ```

## 验证配置

运行以下命令验证配置是否成功：

```bash
python test_system.py
```

如果看到 "tushare连接初始化成功" 和 "获取股票列表成功"，说明配置正确。

## 常见问题

### Q1: 提示 "未配置tushare token"
**A:** 检查环境变量是否正确设置：
```bash
echo $TUSHARE_TOKEN  # Linux/Mac
echo %TUSHARE_TOKEN%  # Windows
```

### Q2: Token 配置了但仍然提示错误
**A:** 可能原因：
1. Token 格式错误，检查是否有多余空格
2. Token 已过期，重新生成
3. 环境变量未生效，重启终端或 IDE

### Q3: 如何在不同环境中使用不同配置
**A:** 使用多个 .env 文件：
```bash
# 开发环境
cp .env.example .env.development
# 生产环境
cp .env.example .env.production

# 使用时指定
export ENV=development  # 或 production
python run_system.py
```
