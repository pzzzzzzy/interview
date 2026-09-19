# C++ 面试核心知识点整理

> 面向应届生的 C++ 技术面试准备指南

---

## 一、基础语法与语言特性

### 1. `const` 关键字的多种用法

#### 修饰变量
```cpp
const int a = 10;  // 常量，不可修改
```

#### 修饰指针
```cpp
// 指向常量的指针（pointer to const）
const int* p1 = &a;      // 不能通过 p1 修改所指向的值，但 p1 本身可以指向别的地址
int const* p2 = &a;      // 同上，写法不同

// 常量指针（const pointer）
int* const p3 = &b;      // p3 本身不能改变指向，但可以通过 p3 修改所指向的值

// 指向常量的常量指针
const int* const p4 = &a; // p4 本身和所指向的值都不能修改
```

**记忆技巧**：`const` 左侧修饰的是**值**，右侧修饰的是**指针本身**。

#### 修饰成员函数
```cpp
class MyClass {
public:
    int getValue() const {  // 常成员函数，不能修改成员变量
        // data = 10;  // 错误！
        return data;
    }
private:
    int data;
};
```

**作用**：
- 承诺不修改对象状态
- 可以被 const 对象调用
- 提高代码安全性和可读性

---

### 2. `static` 关键字的作用

#### 修饰局部变量
```cpp
void func() {
    static int count = 0;  // 只初始化一次，生命周期贯穿整个程序
    count++;
}
```

#### 修饰全局变量/函数
```cpp
static int global_var = 100;  // 限制作用域在当前编译单元（文件），避免命名冲突
static void helper() {}        // 同上
```

#### 修饰类成员
```cpp
class MyClass {
public:
    static int count;                    // 静态成员变量，所有对象共享
    static void printCount() {           // 静态成员函数，无 this 指针
        cout << count << endl;
    }
};
int MyClass::count = 0;  // 类外初始化
```

**核心区别**：
- 静态成员属于**类**，不属于某个对象
- 静态成员函数只能访问静态成员变量

---

### 3. `inline` 内联函数 vs 宏定义

#### inline 函数
```cpp
inline int add(int a, int b) {
    return a + b;
}
```

#### 宏定义
```cpp
#define ADD(a, b) ((a) + (b))
```

**区别对比**：

| 特性 | inline 函数 | 宏定义 |
|------|------------|--------|
| 类型检查 | ✅ 有类型检查 | ❌ 纯文本替换，无类型检查 |
| 调试 | ✅ 可调试 | ❌ 预处理阶段展开，难以调试 |
| 作用域 | ✅ 遵循作用域规则 | ❌ 全局替换 |
| 副作用 | ✅ 参数只求值一次 | ❌ 可能多次求值（如 `ADD(i++, j++)` 会出问题） |

**inline 注意事项**：
- `inline` 只是建议，编译器可能忽略
- 函数体过大或包含递归时不适合内联
- 现代编译器会自动优化，不一定需要手动标记

---

### 4. `sizeof` vs `strlen`

```cpp
char str[] = "hello";
sizeof(str);   // 6 (包含 '\0')
strlen(str);   // 5 (不包含 '\0')
```

**核心区别**：
- `sizeof`：**编译期**确定，计算变量占用的内存字节数
- `strlen`：**运行期**执行，遍历字符串直到 `'\0'`

#### 空类的 sizeof
```cpp
class A {};
sizeof(A);  // 通常是 1
```

**原因**：
- C++ 标准规定每个对象必须有唯一的地址
- 空类也要占用至少 1 字节，以保证不同对象地址不同

---

### 5. `extern "C"` 的作用

```cpp
#ifdef __cplusplus
extern "C" {
#endif

void c_function();

#ifdef __cplusplus
}
#endif
```

**作用**：
- 告诉 C++ 编译器按照 **C 语言的方式** 编译和链接函数
- 避免 C++ 的 **名称修饰（Name Mangling）**

**为什么需要**：
- C++ 支持函数重载，需要在编译时修改函数名（加入参数类型信息）
- C 不支持重载，函数名不变
- 混合编译时，`extern "C"` 保证链接器能找到正确的符号

---

## 二、面向对象与底层机制

### 6. 重载（Overload）、重写（Override）、隐藏（Hide）

#### 重载（Overload）
**同一作用域**内，函数名相同但**参数列表不同**：
```cpp
void print(int a);
void print(double a);
void print(int a, int b);
```

#### 重写（Override / 覆写）
**派生类**重新定义**基类的虚函数**，函数签名必须完全相同：
```cpp
class Base {
public:
    virtual void show() { cout << "Base" << endl; }
};

class Derived : public Base {
public:
    void show() override { cout << "Derived" << endl; }  // 重写
};
```

#### 隐藏（Hide）
派生类定义了与基类**同名的非虚函数**，或者虚函数签名不同：
```cpp
class Base {
public:
    void func(int a) {}
};

class Derived : public Base {
public:
    void func(double a) {}  // 隐藏了 Base::func(int)
};

Derived d;
d.func(10);  // 调用 Derived::func(double)，10 被转换为 10.0
```

**对比总结**：

| 概念 | 作用域 | 是否虚函数 | 函数签名 |
|------|--------|------------|----------|
| 重载 | 同一作用域 | 无关 | 必须不同 |
| 重写 | 派生类 | 必须是虚函数 | 必须相同 |
| 隐藏 | 派生类 | 无关 | 同名即可 |

---

### 7. 多态的实现机制：虚函数表（vtable）与虚函数指针（vptr）

#### 多态示例
```cpp
class Base {
public:
    virtual void func() { cout << "Base::func" << endl; }
};

class Derived : public Base {
public:
    void func() override { cout << "Derived::func" << endl; }
};

Base* p = new Derived();
p->func();  // 输出 "Derived::func"
```

#### 底层机制

1. **编译器为每个包含虚函数的类生成一个虚函数表（vtable）**
   - vtable 是一个函数指针数组
   - 存储该类所有虚函数的地址

2. **每个对象内部包含一个虚函数指针（vptr）**
   - vptr 指向该对象所属类的 vtable
   - vptr 通常位于对象内存布局的最前面

3. **调用虚函数时的过程**：
   ```
   p->func()
   ↓
   通过 p 找到对象的 vptr
   ↓
   通过 vptr 找到 vtable
   ↓
   在 vtable 中查找 func 的地址
   ↓
   调用该地址的函数
   ```

#### 内存布局示意
```
Base 对象:
+--------+
| vptr   | -----> Base vtable: [&Base::func]
+--------+

Derived 对象:
+--------+
| vptr   | -----> Derived vtable: [&Derived::func]
+--------+
```

**关键点**：
- 运行期通过 vptr 动态查找，实现**动态绑定**
- 只有虚函数才会进入 vtable
- 虚函数调用有轻微性能开销（一次间接寻址）

---

### 8. 为什么基类析构函数要声明为 `virtual`

```cpp
class Base {
public:
    ~Base() { cout << "~Base" << endl; }
};

class Derived : public Base {
public:
    ~Derived() { cout << "~Derived" << endl; }
};

Base* p = new Derived();
delete p;  // 只调用 ~Base()，~Derived() 未执行！内存泄漏！
```

**问题**：通过基类指针删除派生类对象时，如果析构函数不是虚函数，只会调用基类的析构函数。

**解决方案**：
```cpp
class Base {
public:
    virtual ~Base() { cout << "~Base" << endl; }
};
```

现在 `delete p` 会正确调用 `~Derived()` → `~Base()`。

**原则**：
- 任何可能被继承的基类，析构函数都应该是 `virtual`
- 如果类不作为基类，可以不声明为 virtual（避免 vtable 开销）

---

### 9. 构造函数的初始化列表

#### 基本用法
```cpp
class MyClass {
public:
    MyClass(int a, int b) : x(a), y(b) {}  // 初始化列表
private:
    int x;
    int y;
};
```

#### 必须使用初始化列表的情况

1. **const 成员变量**
```cpp
class A {
    const int x;
public:
    A(int val) : x(val) {}  // 必须用初始化列表
};
```

2. **引用成员变量**
```cpp
class B {
    int& ref;
public:
    B(int& r) : ref(r) {}  // 必须用初始化列表
};
```

3. **没有默认构造函数的成员对象**
```cpp
class C {
public:
    C(int val) {}  // 无默认构造函数
};

class D {
    C obj;
public:
    D(int val) : obj(val) {}  // 必须在初始化列表中构造 obj
};
```

4. **基类没有默认构造函数**
```cpp
class Base {
public:
    Base(int x) {}
};

class Derived : public Base {
public:
    Derived(int x) : Base(x) {}  // 必须显式调用基类构造函数
};
```

**初始化顺序**：
- 按照成员变量在类中**声明的顺序**初始化，不是初始化列表的顺序
- 基类先于派生类

---

### 10. 编译器默认生成的成员函数

对于一个空类：
```cpp
class Empty {};
```

编译器会自动生成：

1. **默认构造函数**
```cpp
Empty() = default;
```

2. **拷贝构造函数**
```cpp
Empty(const Empty& other) = default;
```

3. **拷贝赋值运算符**
```cpp
Empty& operator=(const Empty& other) = default;
```

4. **析构函数**
```cpp
~Empty() = default;
```

5. **移动构造函数**（C++11）
```cpp
Empty(Empty&& other) = default;
```

6. **移动赋值运算符**（C++11)
```cpp
Empty& operator=(Empty&& other) = default;
```

**注意**：
- 只有在需要时才生成（比如你调用了拷贝构造）
- 如果你自己定义了任何构造函数，默认构造函数不会生成
- 如果定义了拷贝构造，移动构造不会生成

---

## 三、内存管理与指针

### 11. C++ 的内存布局

```
高地址
+-------------------+
|   栈区 (Stack)    |  局部变量、函数参数、返回地址
+-------------------+  ↓ 向下增长
|        ↓          |
|                   |
|        ↑          |
+-------------------+  ↑ 向上增长
|   堆区 (Heap)     |  new/malloc 分配的动态内存
+-------------------+
|  未初始化数据段   |  未初始化的全局变量、静态变量 (BSS)
+-------------------+
|  已初始化数据段   |  已初始化的全局变量、静态变量 (Data)
+-------------------+
|  常量区 (Rodata)  |  字符串字面量、const 全局变量
+-------------------+
|  代码区 (Text)    |  程序的机器指令
+-------------------+
低地址
```

**各区域特点**：

| 区域 | 管理方式 | 生命周期 | 大小 |
|------|----------|----------|------|
| 栈区 | 自动管理 | 函数结束自动释放 | 有限（通常几 MB） |
| 堆区 | 手动管理 | 程序员控制 | 较大 |
| 静态区 | 程序启动时分配 | 程序结束释放 | 固定 |
| 常量区 | 只读 | 程序结束释放 | 固定 |
| 代码区 | 只读 | 程序运行期间 | 固定 |

---

### 12. `new/delete` vs `malloc/free`

#### 核心区别

| 特性 | new/delete | malloc/free |
|------|------------|-------------|
| 语言层面 | C++ 运算符 | C 库函数 |
| 类型安全 | ✅ 返回具体类型指针 | ❌ 返回 void*，需要强制转换 |
| 大小计算 | ✅ 自动 | ❌ 需要手动计算（sizeof） |
| 构造/析构 | ✅ 调用构造函数和析构函数 | ❌ 不调用 |
| 失败返回 | 抛出 std::bad_alloc 异常 | 返回 NULL |
| 重载 | ✅ 可以重载 | ❌ 不能重载 |

#### 示例对比
```cpp
// malloc/free
int* p1 = (int*)malloc(sizeof(int));
*p1 = 10;
free(p1);

// new/delete
int* p2 = new int(10);
delete p2;

// 对象的差异
class MyClass {
public:
    MyClass() { cout << "构造" << endl; }
    ~MyClass() { cout << "析构" << endl; }
};

MyClass* obj1 = (MyClass*)malloc(sizeof(MyClass));  // 不调用构造函数！
free(obj1);  // 不调用析构函数！

MyClass* obj2 = new MyClass();  // 调用构造函数
delete obj2;  // 调用析构函数
```

**重要**：
- 不能混用（`new` 配 `delete`，`malloc` 配 `free`）
- C++ 优先使用 `new/delete` 或智能指针

---

### 13. 深拷贝 vs 浅拷贝

#### 浅拷贝（Shallow Copy）
只复制指针的值，不复制指针指向的内容：
```cpp
class ShallowCopy {
public:
    int* data;
    ShallowCopy(int val) {
        data = new int(val);
    }
    // 默认拷贝构造函数（浅拷贝）
    ShallowCopy(const ShallowCopy& other) = default;
    ~ShallowCopy() {
        delete data;
    }
};

ShallowCopy obj1(10);
ShallowCopy obj2 = obj1;  // obj1.data 和 obj2.data 指向同一块内存
// 析构时会 double free！
```

**潜在风险**：
- 多次释放同一内存（double free）
- 悬空指针
- 一个对象修改会影响另一个对象

#### 深拷贝（Deep Copy）
复制指针指向的内容，创建独立的副本：
```cpp
class DeepCopy {
public:
    int* data;
    DeepCopy(int val) {
        data = new int(val);
    }
    // 深拷贝构造函数
    DeepCopy(const DeepCopy& other) {
        data = new int(*other.data);  // 分配新内存并复制值
    }
    // 深拷贝赋值运算符
    DeepCopy& operator=(const DeepCopy& other) {
        if (this != &other) {
            delete data;
            data = new int(*other.data);
        }
        return *this;
    }
    ~DeepCopy() {
        delete data;
    }
};
```

**原则**：
- 包含指针成员的类，必须实现深拷贝
- 遵循**三/五法则**（后面会提到）

---

### 14. 内存泄漏（Memory Leak）

#### 定义
动态分配的内存没有被释放，导致程序占用的内存不断增长。

#### 常见原因
```cpp
// 1. 忘记 delete
void func() {
    int* p = new int(10);
    // 忘记 delete p;
}

// 2. 异常导致无法释放
void func() {
    int* p = new int(10);
    throw std::exception();  // 异常抛出，delete 未执行
    delete p;
}

// 3. 数组 delete 使用错误
int* arr = new int[10];
delete arr;  // 错误！应该用 delete[]

// 4. 循环引用（智能指针）
struct Node {
    std::shared_ptr<Node> next;
};
std::shared_ptr<Node> a = std::make_shared<Node>();
std::shared_ptr<Node> b = std::make_shared<Node>();
a->next = b;
b->next = a;  // 循环引用，引用计数永远不为 0
```

#### 排查方法
- **Valgrind**（Linux）
- **AddressSanitizer**（编译选项 `-fsanitize=address`）
- **Visual Studio** 内存泄漏检测工具

#### 预防措施
- 使用 **智能指针**（`std::unique_ptr`, `std::shared_ptr`）
- RAII 原则（Resource Acquisition Is Initialization）
- 代码审查
- 使用容器（`std::vector`, `std::string`）代替原始指针

---

### 15. 悬空指针 vs 野指针

#### 悬空指针（Dangling Pointer）
指针指向的内存已经被释放：
```cpp
int* p = new int(10);
delete p;
// p 现在是悬空指针
*p = 20;  // 未定义行为！
```

#### 野指针（Wild Pointer）
未初始化的指针，指向随机地址：
```cpp
int* p;  // 野指针，未初始化
*p = 10;  // 未定义行为！
```

#### 如何避免

1. **释放后置空**
```cpp
delete p;
p = nullptr;
if (p != nullptr) {
    *p = 20;  // 安全检查
}
```

2. **初始化指针**
```cpp
int* p = nullptr;  // 养成习惯
```

3. **使用智能指针**
```cpp
std::unique_ptr<int> p = std::make_unique<int>(10);
// 自动管理，不会悬空
```

4. **作用域控制**
```cpp
{
    int* p = new int(10);
    // ... 使用 p
    delete p;
}  // p 离开作用域
```

---

## 四、现代 C++ 与 STL 标准库

### 16. C++11 智能指针

#### `std::unique_ptr`
**独占所有权**，不可复制，只能移动：
```cpp
std::unique_ptr<int> p1 = std::make_unique<int>(10);
// std::unique_ptr<int> p2 = p1;  // 错误！不能拷贝
std::unique_ptr<int> p2 = std::move(p1);  // 可以移动，p1 变为 nullptr
```

**使用场景**：
- 明确所有权唯一的情况
- 替代原始指针，零开销抽象

#### `std::shared_ptr`
**共享所有权**，使用引用计数：
```cpp
std::shared_ptr<int> p1 = std::make_shared<int>(10);
std::shared_ptr<int> p2 = p1;  // 引用计数变为 2
// 当所有 shared_ptr 都销毁时，内存才被释放
```

**使用场景**：
- 多个对象需要共享同一资源
- 需要在多个地方持有指针

#### `std::weak_ptr`
**弱引用**，不增加引用计数，用于解决**循环引用**问题：
```cpp
struct Node {
    std::shared_ptr<Node> next;
    std::weak_ptr<Node> prev;  // 使用 weak_ptr 打破循环
};

std::shared_ptr<Node> a = std::make_shared<Node>();
std::shared_ptr<Node> b = std::make_shared<Node>();
a->next = b;
b->prev = a;  // weak_ptr 不增加引用计数，避免循环引用
```

**使用 weak_ptr**：
```cpp
std::weak_ptr<int> wp = sp;
if (auto p = wp.lock()) {  // 尝试升级为 shared_ptr
    // 对象存在，可以使用
    *p = 20;
} else {
    // 对象已销毁
}
```

**对比总结**：

| 智能指针 | 所有权 | 拷贝 | 引用计数 | 主要用途 |
|----------|--------|------|----------|----------|
| `unique_ptr` | 独占 | ❌ | ❌ | 替代原始指针 |
| `shared_ptr` | 共享 | ✅ | ✅ | 多方共享资源 |
| `weak_ptr` | 无 | ✅ | ❌ | 打破循环引用 |

---

### 17. 左值、右值与移动语义

#### 左值（Lvalue）vs 右值（Rvalue）

**左值**：有持久状态，可以取地址
```cpp
int a = 10;  // a 是左值
int* p = &a;  // 可以取地址
```

**右值**：临时值，不能取地址
```cpp
int b = 10 + 20;  // 10+20 是右值
// int* p = &(10 + 20);  // 错误！右值不能取地址
```

**简单判断**：能否放在赋值号左边
- 左值：可以
- 右值：不可以

#### 移动语义（Move Semantics）

**问题场景**：
```cpp
std::vector<int> createVector() {
    std::vector<int> v(1000000);
    return v;  // 返回时会拷贝吗？
}

std::vector<int> result = createVector();  // 拷贝开销大！
```

**移动语义**：
- 不拷贝资源，而是**转移所有权**
- 对于临时对象（右值），可以"窃取"其资源

#### `std::move`

```cpp
std::vector<int> v1(1000000);
std::vector<int> v2 = std::move(v1);  // 移动，不拷贝
// 现在 v1 为空，v2 拥有原来 v1 的资源
```

#### 移动构造函数和移动赋值运算符
```cpp
class MyClass {
public:
    int* data;
    
    // 移动构造函数
    MyClass(MyClass&& other) noexcept {
        data = other.data;
        other.data = nullptr;  // "窃取"资源后将源对象置空
    }
    
    // 移动赋值运算符
    MyClass& operator=(MyClass&& other) noexcept {
        if (this != &other) {
            delete data;
            data = other.data;
            other.data = nullptr;
        }
        return *this;
    }
};
```

**关键点**：
- `std::move` 只是类型转换（左值→右值引用），不移动任何东西
- 真正的移动由移动构造/赋值函数完成
- 移动后的对象应处于"有效但未指定"的状态

---

### 18. `std::vector` 的扩容机制

#### 底层实现
- 动态数组，连续内存
- 三个指针：`start`, `finish`, `end_of_storage`

#### 扩容过程
```cpp
std::vector<int> v;
v.push_back(1);  // 容量 0 → 1
v.push_back(2);  // 容量 1 → 2
v.push_back(3);  // 容量 2 → 4
v.push_back(4);
v.push_back(5);  // 容量 4 → 8
```

**扩容策略**：
- GCC：扩容为原来的 **2 倍**
- MSVC：扩容为原来的 **1.5 倍**

**扩容步骤**：
1. 分配新的更大内存
2. 将旧元素**移动/拷贝**到新内存
3. 释放旧内存
4. 更新指针

#### 频繁扩容的问题
- **性能开销**：重新分配内存、拷贝元素
- **迭代器失效**：扩容后原迭代器、指针、引用全部失效

#### 优化方法

**1. 预留容量**
```cpp
std::vector<int> v;
v.reserve(1000);  // 预留 1000 个元素的空间
for (int i = 0; i < 1000; ++i) {
    v.push_back(i);  // 不会触发扩容
}
```

**2. 使用 emplace_back**
```cpp
std::vector<MyClass> v;
v.emplace_back(args);  // 直接在容器内构造，避免临时对象
```

**3. 批量操作**
```cpp
v.insert(v.end(), another_vec.begin(), another_vec.end());
```

**区别**：
- `size()`：当前元素个数
- `capacity()`：当前容量（不触发扩容能容纳的最大元素数）

---

### 19. `std::vector` vs `std::list` 及迭代器失效

#### 对比

| 特性 | std::vector | std::list |
|------|-------------|-----------|
| 底层结构 | 动态数组 | 双向链表 |
| 内存布局 | 连续 | 非连续 |
| 随机访问 | O(1) | O(n) |
| 插入/删除（中间） | O(n) | O(1)（已有迭代器） |
| 插入/删除（末尾） | O(1)（均摊） | O(1) |
| 缓存友好 | ✅ 高 | ❌ 低 |
| 内存开销 | 小 | 大（每个节点额外存储指针） |

**选择原则**：
- **默认选 `vector`**：大多数情况下更快（缓存友好）
- **需要频繁在中间插入/删除** → `list`
- **需要随机访问** → `vector`

#### 迭代器失效（Iterator Invalidation）

**`std::vector`**：

| 操作 | 迭代器失效情况 |
|------|----------------|
| `push_back` | 扩容时全部失效 |
| `insert` | 插入点及之后的失效 |
| `erase` | 删除点及之后的失效 |
| `clear` | 全部失效 |

```cpp
std::vector<int> v = {1, 2, 3, 4, 5};
auto it = v.begin();
v.push_back(6);  // 可能扩容，it 失效
*it;  // 未定义行为！
```

**`std::list`**：

| 操作 | 迭代器失效情况 |
|------|----------------|
| `push_back/push_front` | 不失效 |
| `insert` | 不失效 |
| `erase` | 只有被删除元素的迭代器失效 |

```cpp
std::list<int> l = {1, 2, 3, 4, 5};
auto it = l.begin();
l.push_back(6);  // it 仍然有效
```

**安全实践**：
```cpp
// 删除偶数元素（vector）
for (auto it = v.begin(); it != v.end(); ) {
    if (*it % 2 == 0) {
        it = v.erase(it);  // erase 返回下一个有效迭代器
    } else {
        ++it;
    }
}
```

---

### 20. `std::map` vs `std::unordered_map`

#### 底层数据结构

| 容器 | 底层结构 | 实现 |
|------|----------|------|
| `std::map` | 红黑树 | 自平衡二叉搜索树 |
| `std::unordered_map` | 哈希表 | 数组 + 链表（拉链法） |

#### 性能对比

| 操作 | std::map | std::unordered_map |
|------|----------|---------------------|
| 查找 | O(log n) | O(1) 平均，O(n) 最坏 |
| 插入 | O(log n) | O(1) 平均 |
| 删除 | O(log n) | O(1) 平均 |
| 遍历 | 有序（按 key 排序） | 无序 |
| 内存开销 | 小 | 大（哈希表 + 桶） |

#### 使用场景

**选择 `std::map`**：
- 需要保持 key 的**有序性**
- 需要**范围查询**（`lower_bound`, `upper_bound`）
- 数据量不大，log(n) 性能足够

**选择 `std::unordered_map`**：
- 只需要快速查找，不关心顺序
- 数据量大，追求 O(1) 性能
- key 类型有良好的哈希函数

#### 示例
```cpp
// map（有序）
std::map<int, std::string> m;
m[3] = "three";
m[1] = "one";
m[2] = "two";
for (auto& p : m) {
    cout << p.first << endl;  // 输出：1, 2, 3（有序）
}

// unordered_map（无序）
std::unordered_map<int, std::string> um;
um[3] = "three";
um[1] = "one";
um[2] = "two";
for (auto& p : um) {
    cout << p.first << endl;  // 输出顺序不确定
}
```

#### 自定义类型作为 key

**`std::map`**：需要重载 `operator<`
```cpp
struct Point {
    int x, y;
    bool operator<(const Point& other) const {
        return x < other.x || (x == other.x && y < other.y);
    }
};
std::map<Point, int> m;
```

**`std::unordered_map`**：需要提供哈希函数和相等比较
```cpp
struct Point {
    int x, y;
    bool operator==(const Point& other) const {
        return x == other.x && y == other.y;
    }
};

struct PointHash {
    size_t operator()(const Point& p) const {
        return std::hash<int>()(p.x) ^ (std::hash<int>()(p.y) << 1);
    }
};

std::unordered_map<Point, int, PointHash> um;
```

---

## 附录：面试高频补充知识点

### 三/五法则（Rule of Three/Five）

如果一个类需要自定义以下之一，通常需要自定义全部：

**三法则**（C++11 之前）：
1. 析构函数
2. 拷贝构造函数
3. 拷贝赋值运算符

**五法则**（C++11 及以后）：
1. 析构函数
2. 拷贝构造函数
3. 拷贝赋值运算符
4. 移动构造函数
5. 移动赋值运算符

### RAII 原则

**Resource Acquisition Is Initialization**：资源获取即初始化
- 构造函数中获取资源
- 析构函数中释放资源
- 利用对象生命周期自动管理资源

```cpp
class FileHandler {
    FILE* file;
public:
    FileHandler(const char* name) {
        file = fopen(name, "r");
    }
    ~FileHandler() {
        if (file) fclose(file);
    }
};
// 离开作用域自动关闭文件
```

### nullptr vs NULL

- `NULL`：宏定义为 `0` 或 `(void*)0`
- `nullptr`：C++11 关键字，类型为 `std::nullptr_t`

```cpp
void func(int) { cout << "int" << endl; }
void func(char*) { cout << "char*" << endl; }

func(NULL);     // 可能调用 func(int)，歧义
func(nullptr);  // 明确调用 func(char*)
```

---

## 总结

本文档涵盖了 C++ 应届生技术面试的核心知识点，建议：

1. **理解原理**：不要死记硬背，理解底层机制
2. **动手实践**：每个知识点都写代码验证
3. **对比总结**：善用表格对比相似概念
4. **关注现代 C++**：掌握 C++11/14/17/20 新特性
5. **结合项目经验**：面试时结合实际项目讲解

**推荐学习资源**：
- 《C++ Primer》（第5版）
- 《Effective C++》 / 《Effective Modern C++》
- cppreference.com
- LeetCode C++ 题目练习

祝面试顺利！
