# 第二阶段
## 1.static修饰成员变量，类变量应用场景
![alt text](image.png)
类变量：也叫静态成员变量。static修饰，属于类，与类一起加载，在计算机里只有一份，会被类的所有对象共享。
实例对象：也叫对象变量。无static修饰，属于每个对象的。
类名.类变量（推荐），对象名.类变量（不推荐）相当于要从栈->堆->方法区->栈。跑冤枉路了
## 2.static修饰成员方法
![alt text](image-1.png)
## 3.static修饰类方法的应用场景-工具类
![alt text](image-2.png)
![alt text](image-3.png)
私有化构造器，就不用构造对象来使用工具类了，直接通过类名.类方法调用
## 4.static注意事项
类方法中可以直接访问类的成员，不可以直接访问实例成员
实例方法中既可以直接访问类成员，也可以访问实例成员
实例方法中可以出现this关键字，类方法中不可以出现this关键字
## 5.static代码块
![alt text](image-23.png)
![alt text](image-25.png)
static代码块会随着类只加载一次
## 6.实例代码块
![alt text](image-26.png)
![alt text](image-27.png)
## 7.单例设计模式
![alt text](image-28.png)
## 8.继承：使用继承的好处
![alt text](image-29.png)
![alt text](image-30.png)
## 9.继承：权限修饰符
![alt text](image-31.png)
## 10.继承：单继承、Object、方法重写
![alt text](image-32.png)
![alt text](image-33.png)
![alt text](image-34.png)
## 11.继承：子类访问成员的特点
![alt text](image-35.png)
## 12.继承：子类构造器的特点
![alt text](image-36.png)
![alt text](image-38.png)
## 13. 多态
![alt text](image-41.png)
![alt text](image-42.png)
![alt text](image-43.png)
## 14. final 常量
![alt text](image-40.png)
![alt text](image-39.png)
## 15.抽象类
![alt text](image-44.png)
![alt text](image-45.png)
## 16.接口
![alt text](image-46.png)