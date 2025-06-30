# origin
origin：远程仓库的“昵称”
当你第一次把本地仓库关联到 GitHub 时（比如用 git remote add origin https://github.com/你/notes_lei.git），Git 就把这个 URL 记作 origin。
以后你要推送（git push）或拉取（git pull）都默认对 origin 操作，就不用每次都敲一大串网址。
不同仓库都可以叫origin,因为他们对应本地的目录不一样。
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
git push 仓库别名 分支 #把本地仓库的最新改动推送到远端仓库。
```
# 仓库是私有的，如何clone
从github上复制仓库https地址，然后在本地用git clone命令克隆到本地。
```bash
git clone 项目https地址
```
输入用户名和token

# 查看当前在哪个分支
```bash
git branch
* main
```
# github的merge
## 什么是merge？
merge就是一个分支的代码合进另一个分支，使代码变成统一的新状态。通常把别的分支的改动合并到main
## 举例团队合作中如何使用merge
1. 首先保证自己本地和main是一致的
```bash
git checkout main #切换到main分支
git status #查看状态，如果和当前分支不一致会报错
# 如果不一致，可以使用git pull恢复到main状态
# 注意区分 git pull 和git restore .  前者是从远程仓库拉最新代码，并且和你当前分支合并。后者是恢复到当前分支的最近一次commit的内容，远程仓库不知道
```
2. 创建一个新分支
```bash
git checkout -b 分支名 #创建一个新分支，并切换到新分支
```
3. 在新分支上修改代码
4. 提交代码到新分支
5. 从github上把新分支的代码合并到main
   在 GitHub 上点 New Pull Request，选择 base:main，compare:dev，提交 PR
   审核通过后点击 Merge，GitHub 自动合并

