# 元组
## 元组不可变
元组不可变，原因是其可以作为字典的key和集合的元素，hash(key)决定了数据存储的唯一位置
进一步说明。list不能作为key，但是tuple可以
元组也不可以删除某一个元素。但是可能整个删除， del tup
```py
>>> a = (1, 2, 3)
>>> b = (4, 5, 6)
>>> a += b
>>> a
(1, 2, 3, 4, 5, 6)
# 这里的元组a变成了一个新的元组
```
## 将list转换成元组
```py
>>> list1= ['Google', 'Taobao', 'Runoob', 'Baidu']
>>> tuple1=tuple(list1)
>>> tuple1
('Google', 'Taobao', 'Runoob', 'Baidu')
```
# 字典
## 字典的key必须是不可变对象,可以是数字，字符串或元组
```py
#!/usr/bin/python3
tinydict = {['Name']: 'Runoob', 'Age': 7} 
print ("tinydict['Name']: ", tinydict['Name'])

Traceback (most recent call last):
  File "test.py", line 3, in <module>
    tinydict = {['Name']: 'Runoob', 'Age': 7}
TypeError: unhashable type: 'list'
```
## 字典的浅拷贝
```py
#!/usr/bin/python3
dict1 = {'Name': 'Runoob', 'Age': 7, 'Class': 'First'}
dict2 = dict1.copy()
print ("新复制的字典为 : ",dict2)
```
直接赋值与copy的区别是，直接赋值是引用，copy是深拷贝
## 字典fromkeys()
```py
dict.fromkeys(seq[, value])
seq = ('name', 'age', 'sex')
 
tinydict = dict.fromkeys(seq)
print ("新的字典为 : %s" %  str(tinydict))
 
tinydict = dict.fromkeys(seq, 10)
print ("新的字典为 : %s" %  str(tinydict))
```
## 字典get()
```py
dict.get(key, default=None)
#返回指定键的值，如果键不在字典中返回 default 设置的默认值
```
## 字典 update()
```py
#!/usr/bin/python3
 
tinydict = {'Name': 'Runoob', 'Age': 7}
tinydict2 = {'Sex': 'female' }
tinydict.update(tinydict2)
print ("更新字典 tinydict : ", tinydict)
```

#集合
## 集合的创建
两种方式
```py
set1 = {1, 2, 3, 4}            # 直接使用大括号创建集合
set2 = set([4, 5, 6, 7])      # 使用 set() 函数从列表创建集合
```
注意：创建空集合用set()，而不是{}，因为{}创建的是一个空字典
## 集合运算
```py
>>> a = set('abracadabra')
>>> b = set('alacazam')
>>> a                                  
{'a', 'r', 'b', 'c', 'd'}
>>> a - b                              # 集合a中包含而集合b中不包含的元素
{'r', 'd', 'b'}
>>> a | b                              # 集合a或b中包含的所有元素
{'a', 'c', 'r', 'd', 'b', 'm', 'z', 'l'}
>>> a & b                              # 集合a和b中都包含了的元素
{'a', 'c'}
>>> a ^ b                              # 不同时包含于a和b的元素
{'r', 'd', 'b', 'm', 'z', 'l'}
```
## 集合的添加
```py
1. s.add()# 将元素加入到集合s中，如果元素存在，则不进行任何操作
2. s.update(x)# 参数可以是列表，元组，字典等，用逗号隔开
>>> thisset = set(("Google", "Runoob", "Taobao"))
>>> thisset.update({1,3})
>>> print(thisset)
{1, 3, 'Google', 'Taobao', 'Runoob'}
>>> thisset.update([1,4],[5,6])  
>>> print(thisset)
{1, 3, 4, 5, 6, 'Google', 'Taobao', 'Runoob'}
>>>
```
## 集合删除
```py
#两种方式
s.remove(x)# 若x不存在则会报错
s.discard(x)# 若x不存在，不会报错
s.pop()#随机删除
s.clear()#清空集合
```
# 循环
## range()函数
```py
for i in range(5)#0 0 1 2 3 4
for i in range(3,5)#3 4
for i in range(3,10,2)#3 5 7 9
```

# 推导式
## 列表推导式
```py
[表达式 for 变量 in 列表] 
[表达式 for 变量 in 列表 if 条件]
>>> names = ['Bob','Tom','alice','Jerry','Wendy','Smith']
>>> new_names = [name.upper()for name in names if len(name)>3]
>>> print(new_names)
['ALICE', 'JERRY', 'WENDY', 'SMITH']


结果值1 if 判断条件 else 结果2  for 变量名 in 原列表
list1 = ['python', 'test1', 'test2']
list2 = [word.title() if word.startswith('p') else word.upper() for word in list1]
print(list2)
#['Python', 'TEST1', 'TEST2']
```
## 字典推导式
```py
{ key_expr: value_expr for value in collection if condition }
listdemo = ['Google','Runoob', 'Taobao']
# 将列表中各字符串值为键，各字符串的长度为值，组成键值对
>>> newdict = {key:len(key) for key in listdemo}
>>> newdict
{'Google': 6, 'Runoob': 6, 'Taobao': 6}
```
## 集合推导式
```py
{ expression for item in Sequence if conditional }
```
## 元组推导式
```py
(expression for item in Sequence )
或
(expression for item in Sequence if conditional )
# 需要注意的是返回的是生成器对象
a=(x for x in range(10))
tuple(a)将生成器对象转换成元组
```
# 迭代器
迭代器是一个可以记住遍历位置的对象
```py
list=[1,2,3,4]
it=iter(list)#创建迭代器对象
print(next(it))
#1 
#为什么输出1？

#对于一个对象
class MyNumbers:
  def __iter__(self):
    self.a = 1
    return self
 
  def __next__(self):
    x = self.a
    self.a += 1
    return x

print(next(it))
#2
it = iter(list)    # 创建迭代器对象
for x in it:
    print (x, end=" ")
```
# 生成器
yield
什么时候要用yield？
```py
def fab(max): 
   n, a, b = 0, 0, 1 
   L = [] 
   while n < max: 
       L.append(b) 
       a, b = b, a + b 
       n = n + 1 
   return L
f = iter(fab(1000))
while True:
    try:
        print (next(f), end=" ")
    except StopIteration:
        sys.exit()

对于上边这个例子，我们能看到其实是生成了一个1000个元素的list:f，然后再去使用f，非常占内存。再看看下面这个方法
def fab(max): 
    n, a, b = 0, 0, 1 
    while n < max: 
        yield b 
        # print b 
        a, b = b, a + b 
        n = n + 1 
for x in fab(1000):
    print(x)  
这样的话实际上每次的调用都是再yield处中断并返回一个结果。
```
# with（需要补充）
优点：自动释放资源，代码简洁
```py
file = open('example.txt', 'r')
try:
    content = file.read()
    # 处理文件内容
finally:
    file.close()

with open('example.txt', 'r') as file:
    content = file.read()
    print(content)
# 文件已自动关闭   
```
# 可变对象和不可变对象
strings, tuples, 和 numbers 是不可更改的对象，而 list,dict 等则是可以修改的对象。
不可变类型：变量赋值 a=5 后再赋值 a=10，这里实际是新生成一个 int 值对象 10，再让 a 指向它，而 5 被丢弃，不是改变 a 的值，相当于新生成了 a。
可变类型：变量赋值 la=[1,2,3,4] 后再赋值 la[2]=5 则是将 list la 的第三个元素值更改，本身la没有动，只是其内部的一部分值被修改了。
python 函数的参数传递：
不可变类型：类似 C++ 的值传递，如整数、字符串、元组。如 fun(a)，传递的只是 a 的值，没有影响 a 对象本身。如果在 fun(a) 内部修改 a 的值，则是新生成一个 a 的对象。
可变类型：类似 C++ 的引用传递，如 列表，字典。如 fun(la)，则是将 la 真正的传过去，修改后 fun 外部的 la 也会受影响
# 装饰器
## 为什么需要装饰器
装饰器是一种设计模式，本质上是在不修改函数代码的前提下，为其添加额外功能。
典型场景：日志记录，权限校验，性能统计，缓存等
## 函数是“第一类对象”
python中函数可以像变量一样被使用
```py
#1 赋值给变量
def foo():
    print('i am foo')
foo()
# i am foo
foo2 = foo
foo2()
# i am foo

# 2 作为参数传递
def greet():
    print('hello')
def call_func(func):
    func()
call_func(greet)
# hello

#3 嵌套定义
def outer():
    def inner():
        print('Inside')
    inner()
outer()
# Inside
```
## 基本装饰器：无参，无返回值
```py
def my_decorator(func):
    def wrapper():
        print('【前置处理】开始执行')
        func()
        print('【后置处理】执行完毕')
    return wrapper
#使用语法糖@
@my_decorator
def say_hello():
    print('hello')

#等价于
def say_hello():
    print('hello')
say_hello = my_decorat(say_hello)

say_hello()
#【前置处理】开始执行
#hello
#【后置处理】执行完毕
```
读取@my_decorator，会把下面定义的say_hello当作参数传给my_decorator,然后执行my_decorator函数，返回值赋给say_hello，say_hello就变成了my_decorator的返回值wrapper，say_hello()就相当于执行wrapper()函数。
## 带参数的装饰器
```py
import functools

def decorator(func):
    @functools.wraps(func)      # 保留原函数的 __name__、__doc__
    def wrapper(*args, **kwargs):
        print(">>> 调用前")
        result = func(*args, **kwargs)
        print("<<< 调用后")
        return result
    return wrapper

@decorator
def add(x, y):
    """计算 x+y"""
    return x + y

print(add(3, 5))   # 8
print(add.__name__)  # 仍然是 "add"
```
## 装饰器工厂
```py
import functools

# 1. 装饰器工厂：接收参数 times
def repeat(times):
    def real_decorator(func):                     # 2. 真正的装饰器
        @functools.wraps(func)
        def wrapper(*args, **kwargs):             # 3. 包装器：接收并透传参数
            for i in range(times):                
                print(f"第 {i+1} 次调用 {func.__name__}")
                func(*args, **kwargs)             # 调用原函数
        return wrapper
    return real_decorator

# 4. 使用时：@repeat(times=3)
@repeat(times=3)
def greet(name):
    print(f"Hello, {name}!")

# 5. 调用
#greet("Alice")
# 第 1 次调用 greet
# Hello, Alice!
# 第 2 次调用 greet
# Hello, Alice!
# 第 3 次调用 greet
# Hello, Alice!
```
总结一句话就是给外边再套一层工厂，传参数用的

## 类装饰器(需要补充)
## 内置装饰器
@staticmethod 和java的静态方法一样
@classmethod 



