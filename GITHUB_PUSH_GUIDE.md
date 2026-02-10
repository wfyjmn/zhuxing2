# GitHub 远程仓库推送指南

## 📦 远程仓库信息

**仓库地址**: https://github.com/wfyjmn/zhuxing2
**当前状态**: ✅ 远程仓库已配置

---

## 🔑 推送需要认证

GitHub 推送需要使用 **Personal Access Token (PAT)** 进行认证。

### 获取 GitHub Token

#### 步骤1：生成 Personal Access Token

1. 登录 GitHub：https://github.com
2. 点击右上角头像 → **Settings**（设置）
3. 左侧菜单 → **Developer settings**（开发者设置）
4. 左侧菜单 → **Personal access tokens**（个人访问令牌）
5. 点击 **Generate new token (classic)**（生成新令牌（经典））

#### 步骤2：配置Token权限

在配置页面：

**Note（备注）**: 输入描述，例如：`Coze Coding - zhuxing2`

**Expiration（过期时间）**: 选择合适的过期时间（建议90天或更长）

**Select scopes（选择权限）**：
- ✅ `repo` - 完整的仓库访问权限（必需）
  - repo:status
  - repo_deployment
  - public_repo
  - repo:invite
  - security_events

点击页面底部的 **Generate token（生成令牌）**

#### 步骤3：复制Token

⚠️ **重要**：Token只显示一次，请立即复制！

---

## 🚀 推送代码到远程仓库

### 方法1：使用命令行（推荐）

```bash
cd /workspace/projects

# 设置远程URL（包含Token）
git remote set-url origin https://wfyjmn:YOUR_TOKEN@github.com/wfyjmn/zhuxing2.git

# 推送代码
git push -u origin main
```

**替换 `YOUR_TOKEN` 为你刚生成的Token。**

### 方法2：配置Git凭据

```bash
# 配置用户信息
git config user.name "wfyjmn"
git config user.email "your_email@example.com"

# 推送代码（会提示输入用户名和Token）
git push -u origin main

# 用户名: wfyjmn
# 密码: 粘贴你的GitHub Token
```

---

## 📊 推送内容

### 已提交的代码

```bash
提交ID: 80f83b8
提交信息: feat: 完整恢复DeepQuant选股系统，包含选股A/B/C和短线集合程序
```

**包含文件**:
- ✅ 核心选股脚本（4个）
- ✅ 配置文件（2个）
- ✅ 文档文件（2个）

---

## 🔄 推送备份文件

### 选项1：推送所有文件（包含备份）

```bash
# 添加所有文件（包括817MB备份）
git add .

# 提交
git commit -m "feat: 添加完整项目备份（817MB）"

# 推送
git push -u origin main
```

⚠️ **注意**: 817MB的备份文件较大，可能需要：

1. 使用 **Git LFS**（Large File Storage）管理大文件
2. 或将备份文件添加到 `.gitignore`

### 选项2：仅推送代码，不推送备份

```bash
# 将备份文件添加到.gitignore
echo "restored_files/" >> .gitignore
echo "projects/" >> .gitignore

# 提交.gitignore
git add .gitignore
git commit -m "chore: 添加.gitignore排除备份文件"

# 推送
git push -u origin main
```

---

## 🛠️ 推荐方案：使用Git LFS管理大文件

如果需要推送备份文件，建议使用 Git LFS：

### 安装Git LFS

```bash
# 检查是否已安装
git lfs version

# 如果未安装，运行：
git lfs install
```

### 配置Git LFS

```bash
# 添加大文件类型到LFS追踪
git lfs track "*.gz"
git lfs track "*.tar"
git lfs track "*.zip"

# 添加.gitattributes
git add .gitattributes
git commit -m "chore: 配置Git LFS"

# 推送代码
git push -u origin main
```

### 推送大文件

```bash
# 添加备份文件
git add restored_files/

# 提交（LFS会自动处理大文件）
git commit -m "feat: 添加项目备份文件"

# 推送
git push -u origin main
```

---

## 📝 推送后验证

### 查看远程仓库

```bash
# 查看远程仓库信息
git remote -v

# 查看远程分支
git branch -r

# 查看推送状态
git log --oneline -5
```

### 访问GitHub

推送成功后，访问：https://github.com/wfyjmn/zhuxing2

你应该能看到：
- ✅ 所有提交记录
- ✅ 源代码文件
- ✅ 文档文件

---

## ⚠️ 常见问题

### Q1: 提示 "Authentication failed"
**A**: Token可能过期或权限不足。请检查：
1. Token是否已过期
2. Token是否有 `repo` 权限
3. Token是否正确复制（没有多余空格）

### Q2: 提示 "Repository not found"
**A**: 检查：
1. 仓库地址是否正确
2. 你是否有仓库访问权限
3. 仓库是否已创建

### Q3: 推送大文件失败
**A**: 使用 Git LFS：
```bash
git lfs install
git lfs track "*.gz"
git add .gitattributes
git commit -m "配置LFS"
git push
```

### Q4: 提示 "refusing to merge unrelated histories"
**A**: 强制推送（谨慎使用）：
```bash
git push -u origin main --force
```

---

## 🎯 推荐操作步骤

### 快速推送（仅代码，不包含备份）

```bash
cd /workspace/projects

# 1. 配置远程URL（替换YOUR_TOKEN）
git remote set-url origin https://wfyjmn:YOUR_TOKEN@github.com/wfyjmn/zhuxing2.git

# 2. 添加.gitignore排除备份
echo "restored_files/" >> .gitignore
echo "projects/" >> .gitignore

# 3. 提交
git add .gitignore
git commit -m "chore: 添加.gitignore排除备份文件"

# 4. 推送
git push -u origin main
```

### 完整推送（包含备份，使用LFS）

```bash
cd /workspace/projects

# 1. 配置远程URL（替换YOUR_TOKEN）
git remote set-url origin https://wfyjmn:YOUR_TOKEN@github.com/wfyjmn/zhuxing2.git

# 2. 安装Git LFS
git lfs install

# 3. 配置LFS追踪大文件
git lfs track "*.gz"
git lfs track "*.tar"

# 4. 添加所有文件
git add .

# 5. 提交
git commit -m "feat: 完整项目，包含备份文件（817MB）"

# 6. 推送
git push -u origin main
```

---

## 📞 需要帮助？

如果推送过程中遇到问题，请提供：
1. 错误信息截图
2. 执行的命令
3. 操作步骤

我将帮助您解决问题。

---

**准备好GitHub Token后，选择一种方案执行即可！**
