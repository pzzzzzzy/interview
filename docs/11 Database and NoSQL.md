# 数据库深入 + NoSQL 核心知识储备

---

## 一、为什么AI工程师需要数据库知识？

### AI项目中的数据库应用

**实际场景**:

```
1. 训练数据管理
   - 数百万条样本的存储和检索
   - 数据版本控制
   - 标注数据的管理

2. 模型元数据
   - 模型版本、超参数、性能指标
   - 实验追踪（MLflow、Weights & Biases）

3. 特征存储 (Feature Store)
   - 预计算的特征
   - 在线/离线特征一致性

4. 应用层数据
   - 用户数据、行为日志
   - 推荐系统的用户-物品交互
   - RAG系统的向量数据库

5. 生产监控
   - 模型预测日志
   - 性能指标时序数据
   - A/B测试结果
```

**面试中的考察**:
- SQL查询能力（数据分析、特征工程）
- 数据库设计（范式、索引）
- NoSQL的选择（场景匹配）
- 性能优化（慢查询分析）

---

## 二、关系型数据库基础

### 2.1 数据库基本概念

#### 关系型数据库的结构

```
数据库 (Database)
└─ 表 (Table)
   ├─ 行 (Row/Record/Tuple) - 一条数据
   └─ 列 (Column/Field/Attribute) - 一个字段

例子: 用户表
┌────┬──────┬─────┬────────────┐
│ id │ name │ age │   email    │
├────┼──────┼─────┼────────────┤
│ 1  │ 张三 │ 25  │ zhang@xx.com│ ← 一行
│ 2  │ 李四 │ 30  │ li@xx.com  │
└────┴──────┴─────┴────────────┘
  ↑
 一列
```

**关键术语**:

```
主键 (Primary Key):
- 唯一标识一行
- 不能重复、不能为NULL
- 例: id

外键 (Foreign Key):
- 引用另一个表的主键
- 建立表之间的关系
- 例: user_id引用users表的id

索引 (Index):
- 加速查询
- 类似书的目录
- 空间换时间

约束 (Constraint):
- NOT NULL: 不能为空
- UNIQUE: 唯一
- CHECK: 检查条件
- DEFAULT: 默认值
```

---

### 2.2 SQL基础语法

#### DDL (数据定义语言)

**创建表**:

```sql
-- 创建用户表
CREATE TABLE users (
    id INT PRIMARY KEY AUTO_INCREMENT,  -- 主键，自增
    username VARCHAR(50) NOT NULL UNIQUE,  -- 用户名，唯一
    email VARCHAR(100) NOT NULL,
    age INT CHECK (age >= 0),  -- 年龄非负
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,  -- 创建时间
    INDEX idx_email (email)  -- 在email上创建索引
);

-- 创建订单表
CREATE TABLE orders (
    id INT PRIMARY KEY AUTO_INCREMENT,
    user_id INT NOT NULL,
    product_name VARCHAR(100),
    price DECIMAL(10, 2),  -- 10位数字，2位小数
    order_date DATE,
    FOREIGN KEY (user_id) REFERENCES users(id)  -- 外键
);
```

**修改表**:

```sql
-- 添加列
ALTER TABLE users ADD COLUMN phone VARCHAR(20);

-- 删除列
ALTER TABLE users DROP COLUMN phone;

-- 修改列
ALTER TABLE users MODIFY COLUMN age SMALLINT;

-- 重命名表
RENAME TABLE users TO customers;
```

**删除表**:

```sql
DROP TABLE orders;  -- 永久删除，慎用！
```

---

#### DML (数据操作语言)

**插入数据**:

```sql
-- 单行插入
INSERT INTO users (username, email, age) 
VALUES ('alice', 'alice@example.com', 25);

-- 多行插入
INSERT INTO users (username, email, age) VALUES
    ('bob', 'bob@example.com', 30),
    ('charlie', 'charlie@example.com', 28);

-- 从查询结果插入
INSERT INTO users_backup 
SELECT * FROM users WHERE age > 25;
```

**查询数据**:

```sql
-- 基本查询
SELECT * FROM users;  -- 查询所有列

SELECT username, email FROM users;  -- 查询指定列

-- 条件查询
SELECT * FROM users WHERE age > 25;

SELECT * FROM users WHERE age BETWEEN 20 AND 30;

SELECT * FROM users WHERE username IN ('alice', 'bob');

SELECT * FROM users WHERE email LIKE '%@gmail.com';  -- 模糊查询

-- 排序
SELECT * FROM users ORDER BY age DESC;  -- 降序

SELECT * FROM users ORDER BY age ASC, username DESC;  -- 多列排序

-- 限制结果数量
SELECT * FROM users LIMIT 10;  -- 前10条

SELECT * FROM users LIMIT 10 OFFSET 20;  -- 跳过20条，取10条（分页）
```

**更新数据**:

```sql
-- 更新单行
UPDATE users SET age = 26 WHERE username = 'alice';

-- 更新多行
UPDATE users SET age = age + 1 WHERE age < 30;

-- 更新多列
UPDATE users 
SET email = 'newemail@example.com', age = 35 
WHERE id = 1;
```

**删除数据**:

```sql
-- 删除满足条件的行
DELETE FROM users WHERE age < 18;

-- 删除所有行（保留表结构）
DELETE FROM users;  -- 慎用！

-- 截断表（更快，重置自增ID）
TRUNCATE TABLE users;
```

---

### 2.3 SQL进阶查询

#### 聚合函数

```sql
-- 常用聚合函数
SELECT COUNT(*) FROM users;  -- 总行数

SELECT COUNT(DISTINCT age) FROM users;  -- 不重复的年龄数

SELECT AVG(age) FROM users;  -- 平均年龄

SELECT MAX(age), MIN(age) FROM users;  -- 最大、最小年龄

SELECT SUM(price) FROM orders;  -- 总价

-- 结合条件
SELECT AVG(age) FROM users WHERE age > 20;
```

---

#### GROUP BY (分组)

```sql
-- 按年龄分组，统计每个年龄的人数
SELECT age, COUNT(*) as count
FROM users
GROUP BY age;

-- 结果示例:
-- age | count
-- ----+------
-- 25  | 10
-- 30  | 15
-- 28  | 8

-- 多列分组
SELECT age, city, COUNT(*) as count
FROM users
GROUP BY age, city;

-- HAVING: 对分组结果过滤（WHERE是对原始数据过滤）
SELECT age, COUNT(*) as count
FROM users
GROUP BY age
HAVING COUNT(*) > 5;  -- 只显示人数>5的年龄组
```

**GROUP BY vs WHERE vs HAVING**:

```sql
SELECT age, COUNT(*) as count
FROM users
WHERE age > 18          -- ① 先过滤原始数据（年龄>18）
GROUP BY age            -- ② 再分组
HAVING COUNT(*) > 5;    -- ③ 最后过滤分组结果（人数>5）

执行顺序: WHERE → GROUP BY → HAVING
```

---

#### JOIN (表连接)

**准备数据**:

```sql
-- 用户表
users
┌────┬──────────┐
│ id │ username │
├────┼──────────┤
│ 1  │ alice    │
│ 2  │ bob      │
│ 3  │ charlie  │
└────┴──────────┘

-- 订单表
orders
┌────┬─────────┬──────────┐
│ id │ user_id │ product  │
├────┼─────────┼──────────┤
│ 1  │ 1       │ iPhone   │
│ 2  │ 1       │ MacBook  │
│ 3  │ 2       │ iPad     │
└────┴─────────┴──────────┘
```

**1. INNER JOIN (内连接)**:

```sql
-- 只返回两表都有的匹配行
SELECT users.username, orders.product
FROM users
INNER JOIN orders ON users.id = orders.user_id;

-- 结果:
-- username | product
-- ---------+---------
-- alice    | iPhone
-- alice    | MacBook
-- bob      | iPad

-- charlie没有订单，所以不出现
```

**2. LEFT JOIN (左外连接)**:

```sql
-- 返回左表所有行，右表没有匹配的显示NULL
SELECT users.username, orders.product
FROM users
LEFT JOIN orders ON users.id = orders.user_id;

-- 结果:
-- username | product
-- ---------+---------
-- alice    | iPhone
-- alice    | MacBook
-- bob      | iPad
-- charlie  | NULL     ← 左表有，右表没有

-- 找出没有订单的用户
SELECT users.username
FROM users
LEFT JOIN orders ON users.id = orders.user_id
WHERE orders.id IS NULL;
```

**3. RIGHT JOIN (右外连接)**:

```sql
-- 返回右表所有行，左表没有匹配的显示NULL
SELECT users.username, orders.product
FROM users
RIGHT JOIN orders ON users.id = orders.user_id;

-- 实际中少用（可以用LEFT JOIN调换表顺序实现）
```

**4. FULL OUTER JOIN (全外连接)**:

```sql
-- 返回两表所有行，没有匹配的显示NULL
-- MySQL不直接支持，需要用UNION模拟
SELECT users.username, orders.product
FROM users
LEFT JOIN orders ON users.id = orders.user_id
UNION
SELECT users.username, orders.product
FROM users
RIGHT JOIN orders ON users.id = orders.user_id;
```

---

#### 子查询 (Subquery)

```sql
-- 1. 在WHERE中使用子查询
-- 查询年龄大于平均年龄的用户
SELECT username, age
FROM users
WHERE age > (SELECT AVG(age) FROM users);

-- 2. 在FROM中使用子查询（派生表）
-- 查询每个年龄组的平均订单数
SELECT age_group.age, AVG(order_count) as avg_orders
FROM (
    SELECT users.age, COUNT(orders.id) as order_count
    FROM users
    LEFT JOIN orders ON users.id = orders.user_id
    GROUP BY users.id, users.age
) as age_group
GROUP BY age_group.age;

-- 3. 在SELECT中使用子查询
-- 查询每个用户及其订单数
SELECT username,
       (SELECT COUNT(*) 
        FROM orders 
        WHERE orders.user_id = users.id) as order_count
FROM users;

-- 4. EXISTS: 检查子查询是否有结果
-- 查询有订单的用户
SELECT username
FROM users
WHERE EXISTS (
    SELECT 1 FROM orders WHERE orders.user_id = users.id
);
```

---

#### 窗口函数 (Window Functions)

**非常强大，面试常考！**

```sql
-- 准备数据: 员工薪资表
employees
┌────┬──────┬────────┬────────┐
│ id │ name │ dept   │ salary │
├────┼──────┼────────┼────────┤
│ 1  │ Alice│ Sales  │ 5000   │
│ 2  │ Bob  │ Sales  │ 6000   │
│ 3  │ Carol│ Sales  │ 5500   │
│ 4  │ David│ IT     │ 7000   │
│ 5  │ Eve  │ IT     │ 7500   │
└────┴──────┴────────┴────────┘
```

**1. ROW_NUMBER() - 行号**:

```sql
-- 给每行分配一个唯一的序号
SELECT name, dept, salary,
       ROW_NUMBER() OVER (ORDER BY salary DESC) as row_num
FROM employees;

-- 结果:
-- name  | dept  | salary | row_num
-- ------+-------+--------+--------
-- Eve   | IT    | 7500   | 1
-- David | IT    | 7000   | 2
-- Bob   | Sales | 6000   | 3
-- Carol | Sales | 5500   | 4
-- Alice | Sales | 5000   | 5
```

**2. RANK() 和 DENSE_RANK() - 排名**:

```sql
-- RANK: 相同值排名相同，后续排名跳跃
SELECT name, salary,
       RANK() OVER (ORDER BY salary DESC) as rank
FROM employees;

-- 如果有并列（如两个6000）:
-- salary | rank
-- -------+-----
-- 7500   | 1
-- 7000   | 2
-- 6000   | 3
-- 6000   | 3  ← 并列
-- 5500   | 5  ← 跳到5（不是4）

-- DENSE_RANK: 排名不跳跃
SELECT name, salary,
       DENSE_RANK() OVER (ORDER BY salary DESC) as dense_rank
FROM employees;

-- salary | dense_rank
-- -------+-----------
-- 7500   | 1
-- 7000   | 2
-- 6000   | 3
-- 6000   | 3
-- 5500   | 4  ← 紧接着4（不跳）
```

**3. PARTITION BY - 分组窗口**:

```sql
-- 每个部门内部排名
SELECT name, dept, salary,
       RANK() OVER (PARTITION BY dept ORDER BY salary DESC) as rank_in_dept
FROM employees;

-- 结果:
-- name  | dept  | salary | rank_in_dept
-- ------+-------+--------+-------------
-- Bob   | Sales | 6000   | 1  ← Sales部门第1
-- Carol | Sales | 5500   | 2  ← Sales部门第2
-- Alice | Sales | 5000   | 3  ← Sales部门第3
-- Eve   | IT    | 7500   | 1  ← IT部门第1
-- David | IT    | 7000   | 2  ← IT部门第2
```

**4. 累计和 (Running Total)**:

```sql
-- 按薪资排序的累计和
SELECT name, salary,
       SUM(salary) OVER (ORDER BY salary) as running_total
FROM employees;

-- 结果:
-- name  | salary | running_total
-- ------+--------+--------------
-- Alice | 5000   | 5000
-- Carol | 5500   | 10500  (5000+5500)
-- Bob   | 6000   | 16500  (10500+6000)
-- David | 7000   | 23500
-- Eve   | 7500   | 31000
```

**5. 移动平均 (Moving Average)**:

```sql
-- 当前行和前两行的平均薪资（3行窗口）
SELECT name, salary,
       AVG(salary) OVER (
           ORDER BY salary 
           ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
       ) as moving_avg
FROM employees;
```

**6. LAG 和 LEAD - 访问前后行**:

```sql
-- LAG: 获取前一行的值
SELECT name, salary,
       LAG(salary, 1) OVER (ORDER BY salary) as prev_salary,
       salary - LAG(salary, 1) OVER (ORDER BY salary) as diff
FROM employees;

-- 结果:
-- name  | salary | prev_salary | diff
-- ------+--------+-------------+-----
-- Alice | 5000   | NULL        | NULL
-- Carol | 5500   | 5000        | 500
-- Bob   | 6000   | 5500        | 500
-- David | 7000   | 6000        | 1000
-- Eve   | 7500   | 7000        | 500

-- LEAD: 获取后一行的值（用法类似）
```

---

### 2.4 SQL实战题目

#### 题目1: 查询每个部门薪资最高的员工

**表结构**:

```sql
employees (id, name, dept_id, salary)
departments (id, name)
```

**问题**: 列出每个部门薪资最高的员工姓名、部门名称、薪资

---

**做题思路**:

```
核心思路: "每个部门"的"最高薪资" → 分组问题

步骤:
1. 确定分组依据: dept_id
2. 确定过滤条件: 每组内salary最大
3. 需要JOIN获取部门名称

两种方法:
- 方法1: 子查询（相关子查询）
- 方法2: 窗口函数（推荐，更清晰）
```

**注意事项**:
- ⚠️ 如果多个员工薪资并列第一，两种方法结果可能不同
- ⚠️ 方法1的子查询会对每行执行一次，性能较差
- ✅ 窗口函数用RANK()而不是ROW_NUMBER()，可以处理并列情况

**易错点**:
```sql
-- ❌ 错误写法: 直接GROUP BY
SELECT dept_id, MAX(salary)
FROM employees
GROUP BY dept_id;
-- 问题: 只能得到最高薪资，得不到员工姓名（无法同时取name）

-- ❌ 错误写法: 不相关的列
SELECT name, dept_id, MAX(salary)
FROM employees
GROUP BY dept_id;
-- 问题: MySQL会报错（name不在GROUP BY中）
```

**解答**:

```sql
-- 方法1: 使用子查询
SELECT e.name, d.name as dept_name, e.salary
FROM employees e
JOIN departments d ON e.dept_id = d.id
WHERE e.salary = (
    SELECT MAX(salary)
    FROM employees
    WHERE dept_id = e.dept_id
);

-- 方法2: 使用窗口函数（推荐）
SELECT name, dept_name, salary
FROM (
    SELECT e.name, d.name as dept_name, e.salary,
           RANK() OVER (PARTITION BY e.dept_id ORDER BY e.salary DESC) as rk
    FROM employees e
    JOIN departments d ON e.dept_id = d.id
) ranked
WHERE rk = 1;
```

---

#### 题目2: 查询连续登录3天以上的用户

**表结构**:

```sql
user_logins (user_id, login_date)

示例数据:
user_id | login_date
--------+-----------
1       | 2024-01-01
1       | 2024-01-02
1       | 2024-01-03
1       | 2024-01-05
2       | 2024-01-01
2       | 2024-01-03
```

**问题**: 找出至少连续登录3天的用户

---

**做题思路**:

```
核心技巧: 日期 - 行号 = 常数（对于连续日期）

连续日期的特征:
2024-01-01 (第1个) → 2024-01-01 - 1 = 2023-12-31
2024-01-02 (第2个) → 2024-01-02 - 2 = 2023-12-31 ← 相同！
2024-01-03 (第3个) → 2024-01-03 - 3 = 2023-12-31 ← 相同！
2024-01-05 (第4个) → 2024-01-05 - 4 = 2024-01-01 ← 不同（断了）

步骤:
1. 按用户分组，按日期排序
2. 计算 ROW_NUMBER()
3. 计算 date - row_number 得到分组标识
4. 按用户和分组标识统计，找出count >= 3的
```

**注意事项**:
- ⚠️ 需要先去重（同一天可能有多次登录记录）
- ⚠️ ROW_NUMBER()必须在PARTITION BY user_id内部编号
- ✅ 使用DATE_SUB()或DATE()函数处理日期

**易错点**:
```sql
-- ❌ 错误: 没有按用户分区
SELECT login_date,
       ROW_NUMBER() OVER (ORDER BY login_date) as rn
FROM user_logins;
-- 问题: 不同用户的日期会混在一起

-- ❌ 错误: 没有去重
-- 如果一天登录多次，会破坏连续性判断

-- ❌ 错误: 用DATEDIFF比较
SELECT user_id
FROM user_logins t1
JOIN user_logins t2 ON t1.user_id = t2.user_id
WHERE DATEDIFF(t2.login_date, t1.login_date) = 1;
-- 问题: 这只能找到两天连续，无法扩展到N天
```

**解答**:

```sql
-- 核心思想: 日期 - ROW_NUMBER = 常数（对于连续日期）
WITH login_with_row AS (
    SELECT 
        user_id,
        login_date,
        DATE_SUB(login_date, INTERVAL ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY login_date) DAY) as group_date
    FROM user_logins
)
SELECT user_id
FROM login_with_row
GROUP BY user_id, group_date
HAVING COUNT(*) >= 3;

-- 解释:
-- 连续的日期减去行号，结果是相同的
-- 2024-01-01 - 1 = 2023-12-31
-- 2024-01-02 - 2 = 2023-12-31  ← 相同
-- 2024-01-03 - 3 = 2023-12-31  ← 相同
-- 2024-01-05 - 4 = 2024-01-01  ← 不同（不连续）
```

---

#### 题目3: 计算每个用户的留存率

**表结构**:

```sql
user_activity (user_id, activity_date)
```

**问题**: 计算次日留存率（第1天活跃，第2天也活跃的用户比例）

---

**做题思路**:

```
留存率定义:
次日留存率 = (首日活跃且次日也活跃的用户数) / (首日活跃的总用户数)

步骤:
1. 找到每个用户的首次活跃日期（MIN）
2. 检查这些用户在首日+1天是否有活跃记录
3. 计算比例

关键: LEFT JOIN + 日期加1天
```

**注意事项**:
- ⚠️ 首次活跃日期要用MIN()，不能直接用activity_date
- ⚠️ 要用LEFT JOIN，保留所有首日用户（包括未留存的）
- ⚠️ COUNT(DISTINCT user_id)避免重复计数
- ✅ DATE_ADD(date, INTERVAL 1 DAY)计算次日

**易错点**:
```sql
-- ❌ 错误: 直接JOIN（会漏掉未留存的用户）
SELECT COUNT(*) / (SELECT COUNT(DISTINCT user_id) FROM user_activity)
FROM user_activity t1
JOIN user_activity t2 
ON t1.user_id = t2.user_id 
AND t2.activity_date = DATE_ADD(t1.activity_date, INTERVAL 1 DAY);
-- 问题: 分子和分母的基准不一致

-- ❌ 错误: 没有用首次活跃日期
-- 如果用户多天活跃，会重复计算

-- ❌ 错误: 没有DISTINCT
-- 同一用户次日多次活跃会重复计数
```

**解答**:

```sql
-- 次日留存率
SELECT 
    COUNT(DISTINCT t1.user_id) / COUNT(DISTINCT t2.user_id) as retention_rate
FROM (
    -- 所有用户的首次活跃日期
    SELECT user_id, MIN(activity_date) as first_date
    FROM user_activity
    GROUP BY user_id
) t2
LEFT JOIN user_activity t1 
ON t2.user_id = t1.user_id 
AND t1.activity_date = DATE_ADD(t2.first_date, INTERVAL 1 DAY);

-- 7日留存率（类似，把INTERVAL 1 DAY改为INTERVAL 7 DAY）
```

---

#### 题目4: 查询累计销售额超过1000的第一天

**表结构**:

```sql
sales (date, amount)

示例:
date       | amount
-----------+-------
2024-01-01 | 300
2024-01-02 | 400
2024-01-03 | 500  ← 累计1200，超过1000
2024-01-04 | 200
```

**问题**: 找出累计销售额首次超过1000的日期

---

**做题思路**:

```
核心: 累计和 = 窗口函数SUM() OVER (ORDER BY date)

步骤:
1. 使用窗口函数计算累计和
2. 筛选累计和 >= 1000
3. 取最早的日期（ORDER BY + LIMIT 1）

关键点:
- SUM() OVER (ORDER BY date) 会按日期顺序累加
- 窗口函数的结果可以在WHERE中使用（需要子查询）
```

**注意事项**:
- ⚠️ 窗口函数不能直接在WHERE中使用，需要子查询
- ⚠️ 要按日期排序（ORDER BY date），不是PARTITION BY
- ⚠️ 使用LIMIT 1确保只返回第一天
- ✅ 如果有多行同一天，可能需要先GROUP BY聚合

**易错点**:
```sql
-- ❌ 错误: 窗口函数直接在WHERE中
SELECT date
FROM sales
WHERE SUM(amount) OVER (ORDER BY date) >= 1000;
-- 问题: WHERE不能用窗口函数（会报错）

-- ❌ 错误: 忘记排序
SELECT date
FROM (
    SELECT date, SUM(amount) OVER (ORDER BY date) as cumsum
    FROM sales
) t
WHERE cumsum >= 1000
LIMIT 1;
-- 没有ORDER BY，LIMIT 1返回的不一定是最早的

-- ❌ 错误: 用普通SUM
SELECT date FROM sales GROUP BY date HAVING SUM(amount) >= 1000;
-- 问题: 这是每天的总和，不是累计和
```

**解答**:

```sql
SELECT date
FROM (
    SELECT date,
           SUM(amount) OVER (ORDER BY date) as cumulative_sales
    FROM sales
) t
WHERE cumulative_sales >= 1000
ORDER BY date
LIMIT 1;
```

---

#### 题目5: 查询每个用户购买的第二贵的商品

**表结构**:

```sql
purchases (user_id, product_name, price)
```

**问题**: 每个用户购买的商品中，价格第二高的商品名称

---

**做题思路**:

```
核心: "每个用户" → PARTITION BY user_id
     "第二高" → 排名问题

步骤:
1. 使用DENSE_RANK()对每个用户的商品按价格排名
2. 筛选rank = 2的记录

RANK vs DENSE_RANK:
- RANK: 1, 2, 2, 4 (并列后跳号)
- DENSE_RANK: 1, 2, 2, 3 (并列后不跳号) ← 用这个
```

**注意事项**:
- ⚠️ 使用DENSE_RANK而非RANK（处理并列情况）
- ⚠️ PARTITION BY user_id（每个用户独立排名）
- ⚠️ ORDER BY price DESC（从高到低）
- ✅ 如果没有第二贵的商品（只买了1种），该用户不返回

**易错点**:
```sql
-- ❌ 错误: 使用ROW_NUMBER
SELECT user_id, product_name
FROM (
    SELECT user_id, product_name,
           ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY price DESC) as rn
    FROM purchases
) t
WHERE rn = 2;
-- 问题: 如果有多个商品价格相同（并列第一），ROW_NUMBER会随机选一个
-- 导致第二名实际上可能是第一名

-- ❌ 错误: 使用RANK（有问题）
-- 价格: 100, 90, 90, 80
-- RANK:  1,  2,  2,  4
-- rank=2会返回两个90，但如果要"唯一的第二名"就有歧义

-- ❌ 错误: 没有PARTITION BY
-- 会变成全局排名，不是每个用户的排名

-- ✅ 正确: DENSE_RANK
-- 价格: 100, 90, 90, 80
-- DENSE_RANK: 1, 2, 2, 3
-- rank=2准确表示"第二高的价格"
```

**解答**:

```sql
SELECT user_id, product_name, price
FROM (
    SELECT user_id, product_name, price,
           DENSE_RANK() OVER (PARTITION BY user_id ORDER BY price DESC) as rank
    FROM purchases
) ranked
WHERE rank = 2;

-- 注意: 使用DENSE_RANK而不是RANK
-- 如果有多个商品同价（并列第一），DENSE_RANK确保第二名是rank=2
```

---

#### 题目6: 自连接 - 查找管理者

**表结构**:

```sql
employees (id, name, manager_id)

示例:
id | name  | manager_id
---+-------+-----------
1  | Alice | NULL      (CEO)
2  | Bob   | 1         (Alice的下属)
3  | Carol | 1
4  | David | 2         (Bob的下属)
```

**问题**: 列出每个员工及其管理者的姓名

---

**做题思路**:

```
核心: 自连接 = 表和自己JOIN

理解:
- employees表既有员工信息，也有管理者信息
- manager_id指向同一个表的id
- 需要把表"看成两份"：员工表 + 管理者表

步骤:
1. 员工表 e (id, name, manager_id)
2. 管理者表 m (id, name) - 其实是同一个表
3. JOIN条件: e.manager_id = m.id
```

**注意事项**:
- ⚠️ 使用LEFT JOIN而非INNER JOIN（CEO没有管理者）
- ⚠️ 给表起别名（e, m）以示区分
- ⚠️ CEO的manager_id是NULL，会显示NULL
- ✅ 可以用COALESCE处理NULL：COALESCE(m.name, 'No Manager')

**易错点**:
```sql
-- ❌ 错误: 使用INNER JOIN
SELECT e.name as employee, m.name as manager
FROM employees e
INNER JOIN employees m ON e.manager_id = m.id;
-- 问题: CEO (manager_id=NULL) 会被过滤掉

-- ❌ 错误: 没有区分别名
SELECT name, manager_id FROM employees;
-- 只能看到manager_id，看不到管理者姓名

-- ❌ 错误: JOIN条件写反
JOIN employees m ON e.id = m.manager_id;
-- 这会返回"谁管理谁"，不是"谁的管理者是谁"
```

**解答**:

```sql
SELECT 
    e.name as employee_name,
    m.name as manager_name
FROM employees e
LEFT JOIN employees m ON e.manager_id = m.id;

-- 结果:
-- employee_name | manager_name
-- --------------+-------------
-- Alice         | NULL
-- Bob           | Alice
-- Carol         | Alice
-- David         | Bob
```

---

#### 题目7: 行转列 (Pivot)

**表结构**:

```sql
scores (student, subject, score)

示例:
student | subject | score
--------+---------+------
Alice   | Math    | 90
Alice   | English | 85
Bob     | Math    | 80
Bob     | English | 95
```

**问题**: 转换为每行一个学生，列为各科成绩

**期望结果**:

```
student | Math | English
--------+------+--------
Alice   | 90   | 85
Bob     | 80   | 95
```

---

**做题思路**:

```
核心: 行转列 = CASE WHEN + 聚合函数

原理:
- 用CASE WHEN创建"虚拟列"
- 用MAX/SUM聚合（因为每个学生每科只有一个分数）
- GROUP BY student合并同一学生的多行

步骤:
1. CASE WHEN subject = 'Math' THEN score END → Math列
2. CASE WHEN subject = 'English' THEN score END → English列
3. MAX()取出非NULL值
4. GROUP BY student合并
```

**注意事项**:
- ⚠️ 必须用聚合函数（MAX/SUM），否则GROUP BY会报错
- ⚠️ CASE WHEN不匹配时返回NULL，聚合函数会忽略NULL
- ⚠️ 如果科目数量动态变化，这种方法需要手写每个科目（不灵活）
- ✅ MySQL 8.0+没有PIVOT关键字，需要手动实现

**易错点**:
```sql
-- ❌ 错误: 没有聚合函数
SELECT student,
       CASE WHEN subject = 'Math' THEN score END as Math
FROM scores
GROUP BY student;
-- 问题: Math不在GROUP BY中，会报错

-- ❌ 错误: 没有GROUP BY
SELECT student,
       MAX(CASE WHEN subject = 'Math' THEN score END) as Math
FROM scores;
-- 问题: 会把所有学生合并成一行

-- ❌ 错误: 使用SUM但分数可能累加
-- 如果一个学生同一科有多个分数记录，SUM会累加（不符合预期）
-- 应该用MAX或MIN（假设每个学生每科只有一个分数）

-- ❌ 错误: ELSE 0
CASE WHEN subject = 'Math' THEN score ELSE 0 END
-- 问题: 如果学生没考Math，会显示0（实际应该是NULL）
```

**解答**:

```sql
-- MySQL 8.0+
SELECT student,
       MAX(CASE WHEN subject = 'Math' THEN score END) as Math,
       MAX(CASE WHEN subject = 'English' THEN score END) as English
FROM scores
GROUP BY student;

-- 或者使用聚合函数配合IF
SELECT student,
       SUM(IF(subject = 'Math', score, 0)) as Math,
       SUM(IF(subject = 'English', score, 0)) as English
FROM scores
GROUP BY student;
```

---

#### 题目8: 查找重复数据

**表结构**:

```sql
emails (id, email)
```

**问题**: 找出重复的email地址

---

**做题思路**:

```
核心: GROUP BY + HAVING COUNT(*) > 1

理解:
- 重复 = 同一个email出现多次
- GROUP BY email聚合相同的email
- HAVING过滤出现次数 > 1的

步骤:
1. GROUP BY email分组
2. COUNT(*)统计每组的数量
3. HAVING COUNT(*) > 1筛选重复的
```

**注意事项**:
- ⚠️ 如果要列出所有重复记录（包括id），需要JOIN回原表
- ⚠️ HAVING在GROUP BY之后，WHERE在GROUP BY之前
- ⚠️ 删除重复数据时要小心，通常保留id最小的
- ✅ 可以用窗口函数ROW_NUMBER()标记重复

**易错点**:
```sql
-- ❌ 错误: 使用WHERE
SELECT email
FROM emails
WHERE COUNT(*) > 1  -- WHERE不能用聚合函数
GROUP BY email;

-- ❌ 错误: 想同时返回id
SELECT id, email
FROM emails
GROUP BY email
HAVING COUNT(*) > 1;
-- 问题: id不在GROUP BY中（MySQL 8.0+会报错）
-- 即使不报错，返回的id也是随机的一个

-- ❌ 错误: 删除重复数据的错误方法
DELETE FROM emails
WHERE email IN (
    SELECT email FROM emails GROUP BY email HAVING COUNT(*) > 1
);
-- 问题: 会删除所有重复的记录（包括应该保留的）

-- ✅ 正确的删除方法（保留id最小的）
DELETE e1 FROM emails e1
JOIN emails e2 ON e1.email = e2.email AND e1.id > e2.id;
```

**扩展: 找出所有重复记录（包括id）**:
```sql
-- 方法1: 子查询
SELECT e.*
FROM emails e
WHERE email IN (
    SELECT email FROM emails GROUP BY email HAVING COUNT(*) > 1
);

-- 方法2: 窗口函数
SELECT *
FROM (
    SELECT *, ROW_NUMBER() OVER (PARTITION BY email ORDER BY id) as rn
    FROM emails
) t
WHERE rn > 1;  -- rn > 1 表示重复的记录
```

**解答**:

```sql
-- 方法1: GROUP BY + HAVING
SELECT email
FROM emails
GROUP BY email
HAVING COUNT(*) > 1;

-- 方法2: 列出所有重复的记录（包括id）
SELECT e1.*
FROM emails e1
JOIN (
    SELECT email
    FROM emails
    GROUP BY email
    HAVING COUNT(*) > 1
) e2 ON e1.email = e2.email;

-- 方法3: 删除重复数据，只保留id最小的
DELETE e1 
FROM emails e1
JOIN emails e2 
ON e1.email = e2.email AND e1.id > e2.id;
```

---

#### 题目9: 时间序列 - 填充缺失日期

**表结构**:

```sql
daily_sales (date, amount)

示例（缺少2024-01-02）:
date       | amount
-----------+-------
2024-01-01 | 100
2024-01-03 | 150
2024-01-04 | 200
```

**问题**: 填充缺失日期，金额为0

---

**做题思路**:

```
核心: 生成完整日期序列 + LEFT JOIN原表

步骤:
1. 生成连续日期序列（递归CTE）
2. LEFT JOIN实际数据
3. COALESCE处理NULL为0

关键技巧:
- 递归CTE生成序列
- UNION ALL连接递归结果
- WHERE条件控制递归终止
```

**注意事项**:
- ⚠️ MySQL 8.0+才支持递归CTE（WITH RECURSIVE）
- ⚠️ 必须有递归终止条件，否则无限循环
- ⚠️ COALESCE(amount, 0)处理NULL值
- ✅ 也可以用日期表（预先创建的包含所有日期的表）

**易错点**:
```sql
-- ❌ 错误: 忘记RECURSIVE关键字
WITH date_range AS (
    SELECT MIN(date) as date FROM daily_sales
    UNION ALL
    SELECT DATE_ADD(date, INTERVAL 1 DAY) ...
)
-- 问题: 非递归CTE不能引用自己

-- ❌ 错误: 没有终止条件
WITH RECURSIVE date_range AS (
    SELECT MIN(date) as date FROM daily_sales
    UNION ALL
    SELECT DATE_ADD(date, INTERVAL 1 DAY)
    FROM date_range  -- 无限递归！
)

-- ❌ 错误: 使用INNER JOIN
SELECT dr.date, ds.amount
FROM date_range dr
JOIN daily_sales ds ON dr.date = ds.date;
-- 问题: 只返回有数据的日期，缺失的日期不显示

-- ❌ 错误: 没有处理NULL
SELECT dr.date, ds.amount  -- amount会是NULL
FROM date_range dr
LEFT JOIN daily_sales ds ON dr.date = ds.date;
-- 应该用COALESCE(ds.amount, 0)
```

**MySQL旧版本解决方案**:
```sql
-- 如果不支持递归CTE，可以：
-- 1. 创建一个数字表（0-999）
-- 2. 用DATE_ADD生成日期
SELECT 
    DATE_ADD('2024-01-01', INTERVAL n.num DAY) as date,
    COALESCE(ds.amount, 0) as amount
FROM (
    SELECT 0 as num UNION SELECT 1 UNION SELECT 2 ... UNION SELECT 365
) n
LEFT JOIN daily_sales ds 
ON DATE_ADD('2024-01-01', INTERVAL n.num DAY) = ds.date
WHERE DATE_ADD('2024-01-01', INTERVAL n.num DAY) <= '2024-12-31';
```

**解答**:

```sql
-- 生成日期序列（MySQL 8.0+）
WITH RECURSIVE date_range AS (
    SELECT MIN(date) as date FROM daily_sales
    UNION ALL
    SELECT DATE_ADD(date, INTERVAL 1 DAY)
    FROM date_range
    WHERE date < (SELECT MAX(date) FROM daily_sales)
)
SELECT dr.date, COALESCE(ds.amount, 0) as amount
FROM date_range dr
LEFT JOIN daily_sales ds ON dr.date = ds.date
ORDER BY dr.date;
```

---

#### 题目10: 树形结构 - 查询所有子节点

**表结构**:

```sql
categories (id, name, parent_id)

示例（商品分类）:
id | name      | parent_id
---+-----------+----------
1  | 电子产品  | NULL
2  | 手机      | 1
3  | 电脑      | 1
4  | iPhone    | 2
5  | Android   | 2
```

**问题**: 查询"电子产品"(id=1)下的所有子分类（递归）

---

**做题思路**:

```
核心: 递归CTE（Common Table Expression）

树形结构特点:
- 每个节点有parent_id指向父节点
- 需要递归查找所有子孙节点

步骤:
1. 锚点查询（Anchor）：找到起始节点（电子产品）
2. 递归查询（Recursive）：找子节点，再找子节点的子节点...
3. UNION ALL合并结果
4. 自动终止（没有更多子节点时）
```

**注意事项**:
- ⚠️ 必须用WITH RECURSIVE（递归CTE）
- ⚠️ 锚点查询和递归查询用UNION ALL连接
- ⚠️ 递归部分的JOIN条件：c.parent_id = ct.id（子节点的parent_id = 当前节点的id）
- ⚠️ 防止循环引用（A→B→C→A），可以记录路径或深度限制
- ✅ 添加level字段标识层级深度

**易错点**:
```sql
-- ❌ 错误: 忘记RECURSIVE
WITH category_tree AS (...)
-- 非递归CTE不能引用自己

-- ❌ 错误: 使用UNION而非UNION ALL
SELECT id FROM categories WHERE id = 1
UNION  -- ← 错误
SELECT c.id FROM categories c JOIN category_tree ct ...
-- 问题: UNION会去重且效率低，递归CTE必须用UNION ALL

-- ❌ 错误: JOIN条件写反
JOIN category_tree ct ON c.id = ct.parent_id
-- 这是找父节点，不是找子节点
-- 正确: c.parent_id = ct.id（子节点的parent_id等于当前节点的id）

-- ❌ 错误: 没有终止条件
-- 递归CTE会自动终止（当没有更多匹配行时）
-- 但如果数据有循环（A→B→A），会无限递归
-- 解决: 添加深度限制 WHERE ct.level < 10

-- ❌ 错误: 查询根节点
SELECT * FROM category_tree WHERE parent_id IS NULL;
-- 这只能找到根节点，不是递归查所有子节点
```

**防止循环引用**:
```sql
-- 添加路径记录，防止重复访问
WITH RECURSIVE category_tree AS (
    SELECT id, name, parent_id, 
           0 as level,
           CAST(id AS CHAR(200)) as path  -- 记录路径
    FROM categories WHERE id = 1
    
    UNION ALL
    
    SELECT c.id, c.name, c.parent_id, 
           ct.level + 1,
           CONCAT(ct.path, ',', c.id)
    FROM categories c
    JOIN category_tree ct ON c.parent_id = ct.id
    WHERE ct.level < 10  -- 深度限制
    AND FIND_IN_SET(c.id, ct.path) = 0  -- 避免循环
)
SELECT * FROM category_tree;
```

**扩展: 查询从叶子到根的路径**:
```sql
-- 反向递归（从子节点找父节点）
WITH RECURSIVE path_tree AS (
    -- 锚点: 叶子节点
    SELECT id, name, parent_id, name as path
    FROM categories WHERE id = 5  -- Android
    
    UNION ALL
    
    -- 递归: 找父节点
    SELECT c.id, c.name, c.parent_id, 
           CONCAT(c.name, ' > ', pt.path)
    FROM categories c
    JOIN path_tree pt ON c.id = pt.parent_id
)
SELECT path FROM path_tree WHERE parent_id IS NULL;
-- 结果: 电子产品 > 手机 > Android
```

**解答**:

```sql
-- 使用递归CTE
WITH RECURSIVE category_tree AS (
    -- 起始节点
    SELECT id, name, parent_id, 0 as level
    FROM categories
    WHERE id = 1
    
    UNION ALL
    
    -- 递归查找子节点
    SELECT c.id, c.name, c.parent_id, ct.level + 1
    FROM categories c
    JOIN category_tree ct ON c.parent_id = ct.id
)
SELECT * FROM category_tree;

-- 结果:
-- id | name      | parent_id | level
-- ---+-----------+-----------+------
-- 1  | 电子产品  | NULL      | 0
-- 2  | 手机      | 1         | 1
-- 3  | 电脑      | 1         | 1
-- 4  | iPhone    | 2         | 2
-- 5  | Android   | 2         | 2
```

---

### 2.5 索引和性能优化

#### 索引的原理

**什么是索引？**

```
索引 = 书的目录

没有索引:
要找"机器学习"这个词
→ 从第一页翻到最后一页
→ O(n) 线性扫描

有索引:
→ 查目录："机器学习" → 第58页
→ 直接翻到第58页
→ O(log n) 二分查找（B+树）
```

**B+树索引**:

```
B+树特点:
1. 多路平衡树（不是二叉树）
2. 所有数据在叶子节点
3. 叶子节点之间有指针（适合范围查询）
4. 高度很低（3-4层可以存储千万条数据）

结构示意:
         [30, 60]              ← 根节点（索引）
        /    |    \
   [10,20] [30,50] [60,80]     ← 索引节点
    /        |        \
[数据] → [数据] → [数据]       ← 叶子节点（有指针连接）

查询 WHERE id = 45:
1. 根节点: 45在30-60之间，走中间
2. 索引节点: 45在30-50之间
3. 叶子节点: 找到id=45的数据

只需3次IO！
```

---

#### 索引的类型

**1. 主键索引 (Primary Key Index)**

```sql
-- 主键自动创建索引
CREATE TABLE users (
    id INT PRIMARY KEY,  -- 自动创建主键索引
    name VARCHAR(50)
);

-- InnoDB存储引擎中，主键索引就是数据本身（聚簇索引）
```

**2. 唯一索引 (Unique Index)**

```sql
-- 唯一约束自动创建唯一索引
CREATE TABLE users (
    id INT PRIMARY KEY,
    email VARCHAR(100) UNIQUE  -- 自动创建唯一索引
);

-- 或显式创建
CREATE UNIQUE INDEX idx_email ON users(email);
```

**3. 普通索引 (Normal Index)**

```sql
-- 最常用的索引类型
CREATE INDEX idx_name ON users(name);

-- 可以有重复值
```

**4. 组合索引 (Composite Index)**

```sql
-- 多列组合索引
CREATE INDEX idx_name_age ON users(name, age);

-- 遵循"最左前缀"原则:
-- ✓ WHERE name = 'Alice'               (可以使用索引)
-- ✓ WHERE name = 'Alice' AND age = 25  (可以使用索引)
-- ✗ WHERE age = 25                     (不能使用索引)
```

**5. 全文索引 (Full-Text Index)**

```sql
-- 用于文本搜索
CREATE FULLTEXT INDEX idx_content ON articles(content);

-- 使用
SELECT * FROM articles 
WHERE MATCH(content) AGAINST('machine learning');
```

---

#### 何时使用索引

**应该创建索引**:

```sql
✓ 经常出现在WHERE子句的列
✓ 经常用于JOIN的列
✓ 经常需要排序（ORDER BY）的列
✓ 经常用于聚合（GROUP BY）的列

例子:
-- 这个查询很频繁
SELECT * FROM orders WHERE user_id = 100 ORDER BY order_date;

-- 应该创建索引
CREATE INDEX idx_user_date ON orders(user_id, order_date);
```

**不应该创建索引**:

```sql
✗ 小表（几百行）
✗ 频繁更新的列（索引维护开销大）
✗ 区分度低的列（如性别：只有男/女）
✗ 不常用的列

例子:
-- is_deleted列只有0和1，区分度极低
-- 不适合建索引（除非要查的是少数数据）
```

---

#### 索引失效的情况

```sql
-- 1. 使用函数或表达式
-- ✗ 索引失效
SELECT * FROM users WHERE YEAR(created_at) = 2024;

-- ✓ 改写为范围查询
SELECT * FROM users 
WHERE created_at >= '2024-01-01' AND created_at < '2025-01-01';

-- 2. 使用 OR（可能失效）
-- ✗ 可能失效
SELECT * FROM users WHERE name = 'Alice' OR age = 25;

-- ✓ 改用UNION
SELECT * FROM users WHERE name = 'Alice'
UNION
SELECT * FROM users WHERE age = 25;

-- 3. 使用 != 或 <>
-- ✗ 不会使用索引
SELECT * FROM users WHERE age != 25;

-- 4. 使用 LIKE '%xxx'
-- ✗ 前置通配符不会使用索引
SELECT * FROM users WHERE name LIKE '%Alice';

-- ✓ 后置通配符可以使用索引
SELECT * FROM users WHERE name LIKE 'Alice%';

-- 5. 隐式类型转换
-- ✗ phone是VARCHAR，但查询用INT，索引失效
SELECT * FROM users WHERE phone = 12345678;

-- ✓ 使用字符串
SELECT * FROM users WHERE phone = '12345678';

-- 6. 组合索引不符合最左前缀
-- 索引: (name, age)
-- ✗ 跳过name，直接查age
SELECT * FROM users WHERE age = 25;

-- ✓ 从最左列开始
SELECT * FROM users WHERE name = 'Alice' AND age = 25;
```

---

#### 慢查询优化流程

**1. 开启慢查询日志**

```sql
-- 查看慢查询配置
SHOW VARIABLES LIKE 'slow_query%';
SHOW VARIABLES LIKE 'long_query_time';

-- 设置慢查询阈值（2秒）
SET GLOBAL long_query_time = 2;
SET GLOBAL slow_query_log = ON;
```

**2. 使用EXPLAIN分析查询**

```sql
EXPLAIN SELECT * FROM users WHERE age > 25;

-- 结果解读:
-- type: 访问类型
--   ALL: 全表扫描（最慢）❌
--   index: 索引扫描
--   range: 索引范围扫描
--   ref: 索引等值查询
--   const: 主键或唯一索引等值查询（最快）✓

-- key: 实际使用的索引
-- rows: 扫描的行数（越少越好）
-- Extra: 额外信息
--   Using filesort: 需要额外排序❌
--   Using temporary: 使用临时表❌
--   Using index: 覆盖索引✓
```

**3. 优化示例**

```sql
-- 慢查询
EXPLAIN SELECT * FROM orders WHERE user_id = 100 ORDER BY order_date;

-- 结果: type=ALL, rows=1000000, Extra=Using filesort
-- 问题: 全表扫描 + 额外排序

-- 优化: 创建组合索引
CREATE INDEX idx_user_date ON orders(user_id, order_date);

-- 再次EXPLAIN
-- 结果: type=ref, rows=100, Extra=Using index
-- 改善: 使用索引 + 无需额外排序
```

---

### 2.6 事务和ACID

#### 什么是事务？

**定义**: 一组操作，要么全部成功，要么全部失败。

**经典例子: 银行转账**

```sql
-- 场景: Alice给Bob转账100元
-- 两个操作:
-- 1. Alice账户 -100
-- 2. Bob账户 +100

-- 没有事务:
UPDATE accounts SET balance = balance - 100 WHERE user = 'Alice';
-- 此时断电❌
-- Alice的钱扣了，但Bob没收到 → 钱丢失！

-- 使用事务:
START TRANSACTION;  -- 开始事务

UPDATE accounts SET balance = balance - 100 WHERE user = 'Alice';
UPDATE accounts SET balance = balance + 100 WHERE user = 'Bob';

COMMIT;  -- 提交（两个操作都成功）
-- 或
ROLLBACK;  -- 回滚（都不执行）

-- 即使中途断电，事务也会回滚，保证数据一致性
```

---

#### ACID特性

**A - Atomicity (原子性)**

```
事务是不可分割的最小单位
要么全部成功，要么全部失败

例子: 转账的两个操作不可分割
```

**C - Consistency (一致性)**

```
事务前后，数据保持一致状态

例子: 
转账前: Alice 1000元, Bob 500元, 总计1500元
转账后: Alice 900元, Bob 600元, 总计还是1500元
→ 总额不变（一致）
```

**I - Isolation (隔离性)**

```
多个事务并发执行，互不干扰

例子:
事务1: Alice给Bob转100
事务2: Carol给Alice转200

两个事务不能看到对方未提交的中间状态
```

**D - Durability (持久性)**

```
事务提交后，数据永久保存

例子:
事务提交后即使系统崩溃，数据也不会丢失
→ 通过日志（redo log, undo log）实现
```

---

#### 事务隔离级别

**并发问题**:

**1. 脏读 (Dirty Read)**

```
事务A读取了事务B未提交的数据

时间 | 事务A              | 事务B
-----+-------------------+------------------
1    |                   | UPDATE balance=200
2    | SELECT balance    | (未提交)
     | → 读到200         |
3    |                   | ROLLBACK (回滚)
4    | 使用200做决策     |
     | → 基于错误数据❌  |
```

**2. 不可重复读 (Non-Repeatable Read)**

```
事务A在同一事务内两次读取同一数据，结果不同

时间 | 事务A              | 事务B
-----+-------------------+------------------
1    | SELECT balance    |
     | → 读到100         |
2    |                   | UPDATE balance=200
3    |                   | COMMIT
4    | SELECT balance    |
     | → 读到200         |
     | 同一事务，结果不同❌|
```

**3. 幻读 (Phantom Read)**

```
事务A两次查询，第二次查询多了或少了行

时间 | 事务A              | 事务B
-----+-------------------+------------------
1    | SELECT * WHERE    |
     | age>20            |
     | → 返回3行         |
2    |                   | INSERT age=25
3    |                   | COMMIT
4    | SELECT * WHERE    |
     | age>20            |
     | → 返回4行         |
     | 多了一行"幻影"❌  |
```

---

**四种隔离级别**:

```sql
-- 1. READ UNCOMMITTED (读未提交) - 最低级别
-- 可能: 脏读、不可重复读、幻读
-- 性能: 最好，几乎不加锁
-- 使用: 很少

-- 2. READ COMMITTED (读已提交) - PostgreSQL默认
-- 可能: 不可重复读、幻读
-- 避免: 脏读
-- 性能: 较好
-- 使用: 较常见

-- 3. REPEATABLE READ (可重复读) - MySQL默认
-- 可能: 幻读（MySQL的InnoDB通过间隙锁避免了）
-- 避免: 脏读、不可重复读
-- 性能: 一般
-- 使用: 常见

-- 4. SERIALIZABLE (串行化) - 最高级别
-- 避免: 所有并发问题
-- 性能: 最差，事务完全串行执行
-- 使用: 很少

-- 设置隔离级别
SET TRANSACTION ISOLATION LEVEL REPEATABLE READ;
```

**选择建议**:

```
金融系统: SERIALIZABLE（准确性优先）
电商系统: REPEATABLE READ（平衡）
统计分析: READ COMMITTED（性能优先）
```

---

### 2.7 数据库设计和范式

#### 范式 (Normal Form)

**为什么需要范式？**

```
目标:
1. 减少数据冗余
2. 避免异常（插入、更新、删除异常）
3. 提高数据一致性
```

---

**第一范式 (1NF): 列不可再分**

```
✗ 不符合1NF:
students
┌────┬──────┬──────────────────┐
│ id │ name │ phone            │
├────┼──────┼──────────────────┤
│ 1  │ Alice│ 123-456, 789-012 │ ← 一个单元格存多个值
└────┴──────┴──────────────────┘

✓ 符合1NF:
students              student_phones
┌────┬──────┐        ┌────┬─────────┬─────────┐
│ id │ name │        │ id │ student │ phone   │
├────┼──────┤        ├────┼─────────┼─────────┤
│ 1  │ Alice│        │ 1  │ 1       │ 123-456 │
└────┴──────┘        │ 2  │ 1       │ 789-012 │
                     └────┴─────────┴─────────┘
```

---

**第二范式 (2NF): 消除部分依赖**

```
前提: 满足1NF
要求: 非主键列完全依赖主键（不能只依赖主键的一部分）

✗ 不符合2NF:
orders (order_id, product_id, order_date, product_name, product_price)
主键: (order_id, product_id) - 组合主键

问题:
- product_name只依赖product_id（部分依赖）
- product_price只依赖product_id（部分依赖）
- order_date依赖完整主键

✓ 符合2NF: 拆分表
orders (order_id, product_id, order_date)  
products (product_id, product_name, product_price)
```

---

**第三范式 (3NF): 消除传递依赖**

```
前提: 满足2NF
要求: 非主键列不依赖其他非主键列

✗ 不符合3NF:
employees (id, name, dept_id, dept_name, dept_location)

问题:
- dept_name依赖dept_id（传递依赖: id → dept_id → dept_name）
- dept_location依赖dept_id

✓ 符合3NF: 拆分表
employees (id, name, dept_id)
departments (dept_id, dept_name, dept_location)
```

---

#### 反范式化 (Denormalization)

**什么时候打破范式？**

```
完全遵守范式:
✓ 数据一致性好
✓ 存储空间小
✗ 查询需要多表JOIN，慢

反范式化:
✓ 查询快（减少JOIN）
✗ 数据冗余
✗ 更新复杂（需要同步更新多处）

适用场景:
- 读多写少
- 查询性能关键
- 数据不常变化

例子:
-- 电商订单表，反范式化存储商品名称和价格
orders (id, user_id, product_id, 
        product_name,    ← 冗余，但避免JOIN products表
        product_price,   ← 冗余
        quantity, order_date)

优势: 查询订单时不需要JOIN products表
注意: 商品信息变化时，历史订单保持当时的价格（这其实是需求）
```

---

#### 数据库设计最佳实践

**1. 表设计**

```sql
-- ✓ 好的表设计
CREATE TABLE users (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,  -- 使用BIGINT（支持更多数据）
    username VARCHAR(50) NOT NULL UNIQUE,
    email VARCHAR(100) NOT NULL UNIQUE,
    password_hash CHAR(60) NOT NULL,  -- 固定长度用CHAR
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE,  -- 软删除
    INDEX idx_email (email),
    INDEX idx_created_at (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;  -- 支持emoji

-- ✗ 不好的设计
CREATE TABLE users (
    id INT,  -- ✗ INT可能不够用
    username VARCHAR(255),  -- ✗ 太长了
    email TEXT,  -- ✗ TEXT不适合索引
    password VARCHAR(50)  -- ✗ 不应该存明文密码！
);
```

**2. 命名规范**

```
表名: 复数、小写、下划线分隔
✓ users, order_items, user_profiles

列名: 小写、下划线分隔
✓ user_id, created_at, is_active

布尔字段: is_, has_, can_
✓ is_deleted, has_permission

时间字段: _at, _date
✓ created_at, updated_at, birth_date

外键: 表名_id
✓ user_id, order_id
```

**3. 数据类型选择**

```sql
-- 整数
TINYINT   -- 1字节, -128~127
SMALLINT  -- 2字节, -32768~32767
INT       -- 4字节, -21亿~21亿
BIGINT    -- 8字节, -922京~922京 ← 推荐用于ID

-- 小数
DECIMAL(10, 2)  -- 精确小数，用于金额
FLOAT/DOUBLE    -- 近似小数，用于科学计算

-- 字符串
CHAR(10)       -- 定长，浪费空间但快
VARCHAR(100)   -- 变长，常用
TEXT           -- 大文本，不适合索引

-- 时间
DATE           -- 日期: 2024-01-01
DATETIME       -- 日期时间: 2024-01-01 12:30:00
TIMESTAMP      -- 时间戳（自动时区转换）← 推荐

-- 布尔
BOOLEAN        -- 实际是TINYINT(1)
```

---

## 三、NoSQL数据库

### 3.1 NoSQL vs SQL

#### 核心区别

```
SQL (关系型数据库):
- 结构化数据（表、行、列）
- 固定schema（预定义结构）
- ACID事务
- JOIN操作
- 垂直扩展（升级硬件）
- 例: MySQL, PostgreSQL, Oracle

NoSQL (非关系型数据库):
- 半结构化/非结构化数据
- 灵活schema（无schema或动态schema）
- BASE（基本可用、软状态、最终一致性）
- 无JOIN（数据冗余存储）
- 水平扩展（增加服务器）
- 例: MongoDB, Redis, Cassandra, Neo4j
```

#### 何时选择NoSQL？

```
✓ 选择NoSQL:
- 数据量巨大（TB-PB级）
- 需要高并发读写
- 数据结构不固定（如JSON文档）
- 需要水平扩展
- 地理分布式部署
- 实时性要求高

✓ 选择SQL:
- 数据结构稳定
- 需要复杂查询（多表JOIN）
- 需要强一致性（如金融）
- 数据量适中
- 已有SQL技能和工具链

实际:
大型系统通常混合使用！
- SQL: 核心业务数据
- NoSQL: 缓存、日志、会话、推荐
```

---

### 3.2 NoSQL的四大类型

#### 1. 键值存储 (Key-Value Store)

**代表: Redis**

**特点**:

```
数据模型: Key → Value

最简单的NoSQL类型:
"user:1001" → "{"name": "Alice", "age": 25}"
"session:abc123" → "login_data"

优势:
- 极快（内存存储）
- 简单直观
- 易于分片

适用:
- 缓存
- 会话存储
- 排行榜
- 计数器
```

**Redis基本操作**:

```bash
# 字符串
SET user:1001 "Alice"
GET user:1001  # → "Alice"
INCR counter   # 原子自增

# 过期时间
SET session:abc "data" EX 3600  # 1小时后过期

# 哈希 (存储对象)
HSET user:1001 name "Alice"
HSET user:1001 age 25
HGET user:1001 name  # → "Alice"
HGETALL user:1001    # → {name: "Alice", age: 25}

# 列表 (队列、栈)
LPUSH queue task1   # 左侧插入
RPOP queue          # 右侧弹出 → 队列

# 集合 (去重)
SADD tags:article1 "AI" "ML" "Python"
SMEMBERS tags:article1  # → ["AI", "ML", "Python"]

# 有序集合 (排行榜)
ZADD leaderboard 100 "Alice"
ZADD leaderboard 95 "Bob"
ZREVRANGE leaderboard 0 9  # 前10名
```

**AI应用场景**:

```
1. 模型预测结果缓存
   key: "prediction:user123:model_v2"
   value: {"result": 0.85, "timestamp": ...}

2. 特征缓存
   key: "features:user123"
   value: {feature1: val1, feature2: val2, ...}

3. 实时计数
   key: "model:v2:predictions:today"
   INCR 每次预测

4. 分布式锁
   SETNX lock:model_training 1 EX 3600
```

---

#### 2. 文档数据库 (Document Store)

**代表: MongoDB**

**特点**:

```
数据模型: JSON/BSON文档

文档示例:
{
  "_id": ObjectId("507f1f77bcf86cd799439011"),
  "name": "Alice",
  "age": 25,
  "tags": ["ML", "Python"],
  "address": {
    "city": "Beijing",
    "zipcode": "100000"
  }
}

优势:
- 灵活schema（字段可以不同）
- 支持嵌套文档和数组
- 查询功能丰富
- 易于映射到对象

适用:
- 内容管理系统
- 用户画像
- 日志存储
- 移动应用后端
```

**MongoDB基本操作**:

```javascript
// 插入文档
db.users.insertOne({
    name: "Alice",
    age: 25,
    tags: ["ML", "Python"]
})

// 查询
db.users.find({ age: { $gt: 20 } })  // age > 20
db.users.find({ tags: "ML" })  // 包含"ML"标签

// 更新
db.users.updateOne(
    { name: "Alice" },
    { $set: { age: 26 }, $push: { tags: "AI" } }
)

// 删除
db.users.deleteOne({ name: "Alice" })

// 聚合 (类似SQL的GROUP BY)
db.orders.aggregate([
    { $match: { status: "completed" } },  // WHERE
    { $group: {
        _id: "$user_id",  // GROUP BY user_id
        total: { $sum: "$amount" }  // SUM(amount)
    }}
])
```

**AI应用场景**:

```
1. 训练数据存储
{
  "image_id": "img_001",
  "path": "/data/img_001.jpg",
  "labels": ["cat", "sitting"],
  "bbox": [[10, 20, 100, 150]],
  "annotator": "user123",
  "metadata": {...}
}

2. 模型元数据
{
  "model_id": "bert_v2",
  "architecture": "transformer",
  "hyperparameters": {...},
  "metrics": {
    "accuracy": 0.95,
    "f1": 0.93
  },
  "training_history": [...]
}

3. 实验追踪
{
  "experiment_id": "exp_001",
  "config": {...},
  "results": {...},
  "timestamp": "..."
}
```

---

#### 3. 列族数据库 (Column-Family Store)

**代表: Cassandra, HBase**

**特点**:

```
数据模型: 行键 → 列族 → 列

适合写多读少的场景:
- 时序数据（如IoT传感器数据）
- 日志数据
- 消息记录

优势:
- 高写入吞吐量
- 水平扩展好
- 自动分片

例子: 用户行为日志
Row Key: user_id
Column Family: actions
Columns: timestamp:action, timestamp:action, ...
```

**Cassandra CQL示例**:

```sql
-- 创建表
CREATE TABLE user_actions (
    user_id UUID,
    timestamp TIMESTAMP,
    action TEXT,
    details MAP<TEXT, TEXT>,
    PRIMARY KEY (user_id, timestamp)
);

-- 插入
INSERT INTO user_actions (user_id, timestamp, action)
VALUES (uuid(), '2024-01-01 12:00:00', 'login');

-- 查询
SELECT * FROM user_actions 
WHERE user_id = ? AND timestamp > '2024-01-01';
```

**AI应用场景**:

```
1. 模型预测日志
   Row: model_id + timestamp
   Columns: user_id, input, output, latency, ...

2. 特征时序数据
   Row: user_id
   Columns: 2024-01-01:feature_vector, 2024-01-02:feature_vector, ...

3. A/B测试结果
   Row: experiment_id
   Columns: timestamp:user_id:outcome, ...
```

---

#### 4. 图数据库 (Graph Database)

**代表: Neo4j**

**特点**:

```
数据模型: 节点 (Node) + 关系 (Edge)

例子: 社交网络
节点: 用户
关系: 关注、好友

Alice --[关注]--> Bob
Alice --[好友]--> Carol
Bob --[关注]--> David

优势:
- 查询关系极快（不需要JOIN）
- 自然表达关系
- 支持复杂图算法

适用:
- 社交网络
- 推荐系统
- 知识图谱
- 欺诈检测
```

**Neo4j Cypher查询**:

```cypher
-- 创建节点
CREATE (alice:User {name: "Alice", age: 25})
CREATE (bob:User {name: "Bob", age: 30})

-- 创建关系
MATCH (a:User {name: "Alice"}), (b:User {name: "Bob"})
CREATE (a)-[:FOLLOWS]->(b)

-- 查询: Alice关注的所有人
MATCH (alice:User {name: "Alice"})-[:FOLLOWS]->(user)
RETURN user.name

-- 查询: 二度好友（朋友的朋友）
MATCH (alice:User {name: "Alice"})-[:FOLLOWS]->()-[:FOLLOWS]->(fof)
RETURN fof.name

-- 最短路径
MATCH path = shortestPath(
    (alice:User {name: "Alice"})-[*]-(bob:User {name: "Bob"})
)
RETURN path
```

**AI应用场景**:

```
1. 推荐系统
   User --[购买]--> Product
   User --[浏览]--> Product
   User --[相似]--> User
   
   查询: "购买了A的用户还购买了什么？"

2. 知识图谱
   Entity --[关系]--> Entity
   
   例: 
   "机器学习" --[是一种]--> "人工智能"
   "神经网络" --[属于]--> "机器学习"
   "Transformer" --[是一种]--> "神经网络"

3. 欺诈检测
   Account --[转账]--> Account
   检测异常的转账模式（环路、快速分散等）
```

---

### 3.3 向量数据库

**专门为AI设计的NoSQL！**

#### 为什么需要向量数据库？

```
传统数据库查询:
- WHERE name = 'Alice'  (精确匹配)
- WHERE age > 25  (范围查询)

AI场景的需求:
- 找到"语义相似"的文本
- 找到"视觉相似"的图片
- 找到"最相关"的文档

核心: 向量相似度搜索
```

**向量表示**:

```
文本 "I love machine learning"
    ↓ (Embedding模型)
向量: [0.2, -0.5, 0.8, ..., 0.1]  (768维)

查询 "I enjoy AI"
    ↓
向量: [0.3, -0.4, 0.7, ..., 0.2]

计算相似度:
cosine_similarity(v1, v2) = 0.92  (很相似!)
```

---

#### 代表: Pinecone, Milvus, Weaviate, Qdrant

**基本操作 (以Pinecone为例)**:

```python
import pinecone

# 初始化
pinecone.init(api_key="...")
index = pinecone.Index("my-index")

# 插入向量
index.upsert(vectors=[
    ("id1", [0.1, 0.2, ..., 0.9], {"text": "doc1"}),
    ("id2", [0.3, 0.4, ..., 0.8], {"text": "doc2"})
])

# 查询最相似的向量
query_vector = [0.15, 0.25, ..., 0.85]
results = index.query(
    vector=query_vector,
    top_k=10,  # 返回最相似的10个
    include_metadata=True
)

# 结果:
# [
#   {"id": "id1", "score": 0.95, "metadata": {"text": "doc1"}},
#   {"id": "id3", "score": 0.89, ...},
#   ...
# ]
```

---

#### RAG系统中的向量数据库

**典型架构**:

```
用户问题: "什么是Transformer?"

1. 问题编码
   "什么是Transformer?" → 向量

2. 向量数据库检索
   找到最相似的文档向量
   ↓
   返回: ["Transformer论文", "Attention机制", ...]

3. 结合检索结果 + LLM
   Prompt = 检索到的文档 + 用户问题
   ↓
   LLM生成回答

优势:
- LLM基于最新、最相关的信息回答
- 减少幻觉
- 可以引用来源
```

**代码示例**:

```python
from sentence_transformers import SentenceTransformer
import pinecone

# 1. 初始化
model = SentenceTransformer('all-MiniLM-L6-v2')
pinecone.init(api_key="...")
index = pinecone.Index("knowledge-base")

# 2. 存储文档
documents = [
    "Transformer是一种神经网络架构...",
    "Attention机制的核心思想是...",
    # 更多文档
]

for i, doc in enumerate(documents):
    embedding = model.encode(doc).tolist()
    index.upsert([(f"doc{i}", embedding, {"text": doc})])

# 3. 查询
query = "什么是Transformer?"
query_embedding = model.encode(query).tolist()

results = index.query(
    vector=query_embedding,
    top_k=3,
    include_metadata=True
)

# 4. 结合LLM
context = "\n".join([r['metadata']['text'] for r in results['matches']])
prompt = f"基于以下信息回答问题:\n{context}\n\n问题: {query}"

# 调用LLM (如OpenAI API)
# answer = openai.ChatCompletion.create(...)
```

---

### 3.4 NoSQL性能优化

#### 数据建模原则

**1. 为查询而设计**

```
SQL思维: 先设计表结构（范式化），再写查询
NoSQL思维: 先确定查询模式，再设计数据结构

例子 (MongoDB):
查询: "获取用户及其所有订单"

方案1: 关系型思维（不推荐）
users: {_id, name}
orders: {_id, user_id, product, ...}
→ 需要JOIN（MongoDB的$lookup，慢）

方案2: NoSQL思维（推荐）
users: {
  _id, 
  name,
  orders: [  ← 嵌入文档
    {product, price, date},
    {product, price, date}
  ]
}
→ 一次查询搞定！
```

**2. 数据冗余是OK的**

```
SQL: 避免冗余（范式化）
NoSQL: 冗余换性能

例子:
orders: {
  _id,
  user_id,
  user_name,  ← 冗余（users表也有）
  product_id,
  product_name,  ← 冗余（products表也有）
  price,  ← 快照当时的价格
  ...
}

优势:
- 查询订单时不需要JOIN users和products
- 即使用户改名、商品涨价，历史订单保持不变（需求）
```

---

#### 索引优化

**MongoDB索引**:

```javascript
// 创建索引
db.users.createIndex({ email: 1 })  // 1=升序, -1=降序

// 复合索引
db.orders.createIndex({ user_id: 1, order_date: -1 })

// 检查查询是否使用索引
db.users.find({ email: "alice@example.com" }).explain("executionStats")

// 结果: 
// - totalDocsExamined: 扫描的文档数（越少越好）
// - executionTimeMillis: 执行时间
```

**Redis优化**:

```
1. 使用合适的数据结构
   - 计数: String (INCR)
   - 排行榜: Sorted Set (ZADD)
   - 去重: Set (SADD)

2. 设置过期时间
   SET key value EX 3600  # 避免内存爆满

3. 使用Pipeline批量操作
   pipeline = redis.pipeline()
   for i in range(1000):
       pipeline.set(f"key{i}", value)
   pipeline.execute()  # 一次网络请求

4. 避免大Key
   - 单个key不要超过10MB
   - List/Set/Hash不要有太多元素
```

---

## 四、面试高频问题

### Q1: 索引为什么快？底层原理是什么？

**回答框架**:
```
索引快的原因是使用了B+树数据结构:

1. B+树特点:
   - 多路平衡树（不是二叉树）
   - 高度低（3-4层存储千万数据）
   - 所有数据在叶子节点，且叶子节点有指针连接

2. 查询过程:
   - O(log n)复杂度
   - 每层是一次磁盘IO
   - 3-4次IO就能找到数据

3. 范围查询:
   - 叶子节点有指针，顺序扫描即可
   - 非常高效

对比全表扫描:
- 1000万数据，全表扫描需要1000万次比较
- B+树只需要4次IO
```

---

### Q2: 事务的ACID是什么？如何实现的？

**回答框架**:
```
ACID是事务的四个特性:

A - Atomicity (原子性):
- 事务不可分割，要么全成功，要么全失败
- 实现: undo log（记录修改前的值，回滚时恢复）

C - Consistency (一致性):
- 事务前后数据保持一致状态
- 实现: 由原子性、隔离性、持久性共同保证

I - Isolation (隔离性):
- 多个事务并发执行互不干扰
- 实现: 锁机制 + MVCC (多版本并发控制)

D - Durability (持久性):
- 事务提交后永久保存
- 实现: redo log（先写日志，再写数据，崩溃后重放日志）

举例:
银行转账必须是ACID的，否则可能出现钱丢失、重复扣款等问题。
```

---

### Q3: 什么时候用NoSQL？什么时候用SQL？

**回答框架**:
```
选择NoSQL的场景:
1. 数据量巨大（TB-PB级），需要水平扩展
2. 数据结构不固定（如JSON文档、日志）
3. 高并发读写（如缓存、会话）
4. 实时性要求高（如Redis毫秒级响应）
5. 特殊数据模型（如图、向量）

选择SQL的场景:
1. 需要复杂查询（多表JOIN、聚合）
2. 需要强一致性（如金融交易）
3. 数据结构稳定，关系清晰
4. 已有SQL生态和工具

实际项目中通常混合使用:
- MySQL: 用户、订单等核心数据
- Redis: 缓存、会话、计数器
- MongoDB: 日志、用户画像
- 向量数据库: RAG检索
```

---

### Q4: MySQL的InnoDB和MyISAM有什么区别？

**回答框架**:
```
InnoDB (默认，推荐):
✓ 支持事务（ACID）
✓ 支持外键
✓ 行级锁（并发性能好）
✓ 崩溃恢复能力强
✓ MVCC支持高并发读

MyISAM (旧引擎):
✗ 不支持事务
✗ 不支持外键
✗ 表级锁（并发性能差）
✓ 查询速度稍快（无事务开销）
✓ 占用空间小

选择:
- 几乎总是选InnoDB
- 只有只读、不需要事务的场景考虑MyISAM
```

---

### Q5: 如何优化慢查询？

**回答框架**:
```
1. 使用EXPLAIN分析查询计划
   - 检查type（是否全表扫描）
   - 检查key（是否使用索引）
   - 检查rows（扫描行数）

2. 添加索引
   - WHERE、JOIN、ORDER BY的列
   - 注意组合索引的最左前缀原则

3. 避免索引失效
   - 不要在索引列上使用函数
   - 避免隐式类型转换
   - LIKE避免前置通配符

4. 优化查询本身
   - 只查询需要的列（避免SELECT *）
   - 使用LIMIT限制结果
   - 拆分复杂查询

5. 表设计优化
   - 适当的反范式化
   - 垂直/水平分表

6. 其他手段
   - 读写分离（主从复制）
   - 缓存（Redis）
   - 分库分表
```

---

### Q6: Redis为什么这么快？

**回答框架**:
```
1. 内存存储
   - 所有数据在内存，无磁盘IO
   - 访问速度: 纳秒级

2. 单线程模型
   - 避免线程切换开销
   - 避免锁竞争
   - 简化实现

3. IO多路复用
   - 单线程处理多个客户端连接
   - epoll/select/kqueue

4. 高效的数据结构
   - String、Hash、List、Set、Sorted Set
   - 针对不同场景优化

5. 持久化可选
   - 可以纯内存运行（最快）
   - 需要持久化时用RDB或AOF

典型性能:
- 读写: 10万+ QPS
- 延迟: 亚毫秒级
```

---

### Q7: 向量数据库在RAG中的作用是什么？

**回答框架**:
```
RAG (检索增强生成) 的核心是"检索":

传统搜索:
- 关键词匹配
- 问题: "什么是Transformer" 搜不到 "Attention机制"

向量搜索:
- 语义相似度
- "什么是Transformer" 能找到 "Attention is All You Need"

流程:
1. 离线: 文档 → Embedding → 向量数据库
2. 在线:
   - 用户提问 → Embedding → 向量
   - 向量数据库检索最相似的文档
   - 文档 + 问题 → LLM生成答案

优势:
- LLM基于实时、准确的信息回答
- 减少幻觉
- 可追溯来源

实际项目中:
- 知识库问答
- 智能客服
- 文档助手
```

---

### Q8: 数据库的三大范式是什么？

**回答框架**:
```
三大范式用于减少数据冗余、避免异常:

1NF (第一范式):
- 列不可再分
- 每个单元格只存一个值

2NF (第二范式):
- 满足1NF
- 消除部分依赖（非主键列完全依赖主键）

3NF (第三范式):
- 满足2NF
- 消除传递依赖（非主键列不依赖其他非主键列）

但实际中会适度反范式化:
- 为了性能，允许一定冗余
- 减少JOIN
- 例如订单表冗余存储商品名称和价格

权衡: 范式化(一致性) vs 反范式化(性能)
```

---

### Q9: CAP理论是什么？

**回答框架**:
```
CAP理论: 分布式系统只能同时满足两个

C - Consistency (一致性):
- 所有节点同一时间看到相同数据

A - Availability (可用性):
- 每个请求都能得到响应（成功或失败）

P - Partition Tolerance (分区容错):
- 网络分区时系统仍能工作

只能三选二:
- CP: 保证一致性，牺牲可用性（如HBase）
  → 网络分区时，拒绝服务
- AP: 保证可用性，牺牲一致性（如Cassandra）
  → 网络分区时，返回旧数据
- CA: 理论上不存在（分布式系统必须容忍分区）

实际选择:
- 金融: CP（一致性优先）
- 社交: AP（可用性优先，最终一致即可）
```

---

### Q10: 什么是MVCC？

**回答框架**:
```
MVCC (Multi-Version Concurrency Control) - 多版本并发控制

核心思想:
- 读不阻塞写，写不阻塞读
- 通过维护多个版本实现

工作原理 (InnoDB):
1. 每行数据有隐藏列:
   - trx_id: 事务ID
   - roll_ptr: 回滚指针（指向undo log）

2. 读数据时:
   - 根据事务ID判断该版本是否可见
   - 不可见则通过undo log构造旧版本

3. 写数据时:
   - 新版本写入，旧版本保留在undo log

例子:
事务A读 → 看到版本1
事务B写 → 创建版本2
事务A再读 → 仍看到版本1（可重复读）

优势:
- 高并发
- 实现REPEATABLE READ隔离级别
- 无需锁（读写不冲突）
```

---

## 五、总结

### 核心知识点回顾

✅ **SQL基础**:
- DDL、DML语句
- JOIN、子查询
- 窗口函数
- 10个实战题目

✅ **SQL进阶**:
- 索引原理和优化
- 事务和ACID
- 隔离级别
- 数据库设计和范式

✅ **NoSQL**:
- 四大类型: KV、文档、列族、图
- Redis、MongoDB的使用
- 向量数据库和RAG
- CAP理论

✅ **性能优化**:
- 慢查询分析
- 索引设计
- 数据建模
- 缓存策略

---

### 学习建议

```
1. 多练SQL
   - LeetCode数据库题目
   - HackerRank SQL
   - 自己构造数据集练习

2. 理解原理
   - B+树、MVCC、WAL
   - 知道"为什么"比"怎么用"更重要

3. 结合实践
   - 分析实际项目的数据模型
   - 做性能优化
   - 尝试不同的NoSQL

4. 关注AI场景
   - 特征存储
   - 向量检索
   - 实验追踪
```

---

**恭喜你完成Day 11的学习！**

你现在掌握了：
- ✅ SQL查询和优化
- ✅ 数据库设计
- ✅ NoSQL的选择和使用
- ✅ 向量数据库和RAG

**完整的数据管理技能！** 🎉
