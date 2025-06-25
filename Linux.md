# 云服务器给别人开账号，大致流程：
## 方法一：用户/密码登录
### 原理：通过给新建用户设置密码，别人用用户名+密码登录
### 步骤：centos
1. 登录服务器，新建用户
```bash
useradd -m 新用户名
passwd 新用户名
#-m会为用户创建家目录
```
2. 测试登录
新开一个终端，运行：
```bash
ssh 新用户名@服务器ip
#它会提示你输入刚才设的密码，输入正确就能登录。
```
3. 给sudo权限
```bash
usermod -aG wheel 新用户名
#usermod 修改已有用户属性的命令
#-aG 将用户添加到组，-G 将用户添加到组，-aG 与 -G 的区别是，-aG 是追加，-G 是替换
#wheel 是管理员组，将新用户添加到管理员组，就相当于给新用户sudo权限了
```
## 方法二：SSH公钥登录（推荐，安全）
### 原理：通过给新建用户设置公钥，别人用公钥登录
### 步骤：centos
1. 对方打开自己的终端，运行
```bash
ssh-keygen -t rsa -b 4096 -C "your_email@example.com"
#-t 指定加密算法，-b 指定密钥长度，-C 添加注释
#一路回车，最后会在~/.ssh/目录下生成两个文件：id_rsa和id_rsa.pub
```
id_rsa是私钥，id_rsa.pub是公钥，把公钥发给hoster
2. hoster打开自己的终端，运行
```bash
useradd -m bob
passwd bob   # 然后输入一个临时密码
#-m会为用户创建家目录
```
3. hoster切换到该用户家目录，并创建.ssh文件夹
```bash
su - bob         # 切换到bob用户
mkdir -p ~/.ssh   # 创建.ssh文件夹-p 参数表示如果上级目录不存在就一并创建；不会报错
chmod 700 ~/.ssh   # 修改.ssh文件夹权限7 对应 rwx（读/写/执行），即只有目录拥有者（bob）能访问
```
4. hoster在authorized_keys文件中添加bob的公钥
```bash
echo "ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAACAQD…bob@laptop" \
  >> ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys
```


