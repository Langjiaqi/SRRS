# GitHub代码上传指南

## 前置准备

### 1. 在GitHub上创建仓库
1. 登录 [GitHub](https://github.com)
2. 点击右上角的 "+" 号，选择 "New repository"
3. 填写仓库名称和描述
4. 选择 Public 或 Private
5. **不要**勾选 "Initialize this repository with README"
6. 点击 "Create repository"
7. 复制仓库的 HTTPS URL（形如：`https://github.com/用户名/仓库名.git`）

### 2. 准备GitHub认证
- **推荐方式：Personal Access Token**
  1. 进入 GitHub Settings > Developer settings > Personal access tokens
  2. 生成新的 token，权限选择 `repo`
  3. 保存好这个 token（只显示一次）

## 使用自动上传脚本

### 方法一：使用提供的脚本（推荐）

```bash
# 运行上传脚本
./upload_to_github.sh https://github.com/您的用户名/您的仓库名.git
```

按照提示输入：
- GitHub用户名
- 邮箱地址
- 提交信息（可选，默认为"Initial commit"）
- 认证信息（用户名和Token）

### 方法二：手动执行命令

```bash
# 1. 配置Git用户信息
git config --global user.name "您的用户名"
git config --global user.email "您的邮箱"

# 2. 添加所有文件
git add .

# 3. 提交代码
git commit -m "Initial commit"

# 4. 设置主分支
git branch -M main

# 5. 添加远程仓库
git remote add origin https://github.com/您的用户名/您的仓库名.git

# 6. 推送到GitHub
git push -u origin main
```

## 后续更新代码

当您修改代码后，使用以下命令更新：

```bash
# 添加修改的文件
git add .

# 提交修改
git commit -m "描述您的修改内容"

# 推送到GitHub
git push
```

## 常见问题

### 1. 认证失败
- 确保使用正确的用户名和Personal Access Token
- 不要使用密码，GitHub已经停止支持密码认证

### 2. 推送被拒绝
```bash
# 如果远程有更新，先拉取
git pull origin main

# 然后再推送
git push
```

### 3. 文件太大
- 检查 `.gitignore` 文件是否正确配置
- 使用 `git rm --cached 文件名` 移除已跟踪的大文件

### 4. 查看当前状态
```bash
# 查看文件状态
git status

# 查看提交历史
git log --oneline

# 查看远程仓库信息
git remote -v
```

## 安全建议

1. 不要提交敏感信息（密码、API密钥等）
2. 使用 `.gitignore` 文件排除不必要的文件
3. 定期检查提交历史，确保没有意外提交敏感文件
4. 使用Personal Access Token而不是密码

## 项目文件说明

- `.gitignore` - 指定Git忽略的文件和目录
- `upload_to_github.sh` - 自动上传脚本
- `GitHub上传说明.md` - 本说明文档

现在您可以安全地将代码上传到GitHub了！ 