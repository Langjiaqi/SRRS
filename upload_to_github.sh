#!/bin/bash

echo "=== GitHub代码上传脚本 ==="
echo ""

# 检查是否提供了GitHub仓库URL
if [ $# -eq 0 ]; then
    echo "使用方法: ./upload_to_github.sh <GitHub仓库URL>"
    echo "例如: ./upload_to_github.sh https://github.com/yourusername/your-repo.git"
    exit 1
fi

REPO_URL=$1

echo "1. 设置Git用户信息（如果还没有设置的话）"
echo "请输入您的GitHub用户名:"
read -p "用户名: " USERNAME
echo "请输入您的邮箱:"
read -p "邮箱: " EMAIL

# 设置Git配置
git config --global user.name "$USERNAME"
git config --global user.email "$EMAIL"

echo ""
echo "2. 检查当前Git状态..."
git status

echo ""
echo "3. 添加所有文件到Git..."
git add .

echo ""
echo "4. 查看将要提交的文件..."
git status

echo ""
echo "5. 提交代码..."
read -p "请输入提交信息 (默认: Initial commit): " COMMIT_MSG
COMMIT_MSG=${COMMIT_MSG:-"Initial commit"}
git commit -m "$COMMIT_MSG"

echo ""
echo "6. 设置主分支为main..."
git branch -M main

echo ""
echo "7. 添加远程仓库..."
git remote add origin "$REPO_URL"

echo ""
echo "8. 推送代码到GitHub..."
echo "注意：首次推送可能需要输入GitHub用户名和密码/Token"
git push -u origin main

echo ""
echo "=== 上传完成! ==="
echo "您的代码已经成功上传到: $REPO_URL"
echo ""
echo "后续更新代码，可以使用以下命令："
echo "git add ."
echo "git commit -m \"更新说明\""
echo "git push" 