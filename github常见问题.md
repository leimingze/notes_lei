# origin
origin：远程仓库的“昵称”
当你第一次把本地仓库关联到 GitHub 时（比如用 git remote add origin https://github.com/你/notes_lei.git），Git 就把这个 URL 记作 origin。
以后你要推送（git push）或拉取（git pull）都默认对 origin 操作，就不用每次都敲一大串网址。
```bash
git remote -v

origin  https://github.com/你/notes_lei.git (fetch)
origin  https://github.com/你/notes_lei.git (push)
```
# branch
branch：分支
分支就像是一条独立的工作路线，允许你在不影响其他工作的情况下做改动。
main（以前叫 master）是 Git 给第一条、最主要的分支默认起的名字，就好像小说的“正篇”。
你在本地仓库里，默认情况下会在 main 这条线上写代码、提交历史。
```bash
git branch

* main
```
# push流程
```bash
git add .  #把“工作区”中的改动（包括新文件）放入暂存区，告诉 Git 你要把它们包括到下次提交里。
git commit -m "一句话描述这次改动" #把暂存区里的内容记录到 本地仓库，并附上这次改动的说明。
git push origin main #把本地仓库的最新改动推送到远端仓库。
```
# 仓库是私有的，如何clone
## 配置SSH公钥认证
1. 生成SSH公钥
在服务器上生成SSH公钥
```bash
ssh-keygen -t rsa -C "<EMAIL>" -f ~/.ssh/<dir_name>
```
2. 查看公钥内容
```bash
cat ~/.ssh/<dir_name>
```
3. 复制公钥内容交给github账户，添加到SSH Keys
4. 测试
```bash
ssh -T git@github.com
```   
