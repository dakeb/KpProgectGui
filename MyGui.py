from datetime import time
from tkinter import *
from tkinter.filedialog import askopenfilename
from tkinter.ttk import Combobox
from timeit import default_timer as timer

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import time
import pyodbc
import re

LOG_LINE_NUM = 0

init_window = Tk()

init_window.title("软件工程")  # 窗口名
init_window.geometry('650x500+40+40')

frameX = Frame(init_window, height=200, relief=GROOVE)  # 操作选择界面
frameXy = Frame(init_window, height=200)  # 文件选择界面
frameY = Frame(init_window, width=500, height=100)  # 最优解输出界面
frameYx = Frame(init_window, width=750, height=200, bg="pink", pady=20)  # 日志界面

frameX.grid(row=0, column=0, padx=20)  # 操作选择界面
frameXy.grid(row=1, column=0, columnspan=2)  # 文件选择界面
frameY.grid(row=0, column=1, pady=10)  # 最优解输出界面
frameYx.grid(row=2, column=0, columnspan=3)  # 日志界面

# 功能函数
filename = StringVar()


def fileopen():
    file_sql = askopenfilename()
    if file_sql:
        filename.set(file_sql)
    return filename



Entry(frameXy, width=30, textvariable=filename).grid(row=0, column=0, padx=5)
# Entry(frameXy, width=30).grid(row=0, column=2,padx=5)
# tablename = Entry().get()

def fileoperate():
    # 读取第一行之后的数据
    f = open(filename.get(), 'r')
    res = f.readlines()[1:]
    res = [line.strip("\n") for line in res]
    f1 = open(filename.get(), 'r')
    res1 = f1.readlines()[:1]
    res1 = [line.strip("\n") for line in res1]
    f.close()
    f1.close()
    return res, res1


def painter():
    resx = fileoperate()
    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.sans-serif'] = 'NSimSun,Times New Roman'

    print(resx[0])
    x, y = np.loadtxt(resx[0], delimiter=' ', unpack=True)
    plt.plot(x, y, '.', color='blue')

    plt.xlabel('wight')
    plt.ylabel('value')
    plt.title('scatter plot')
    plt.legend()
    write_log_to_Text("绘制散点图成功")
    plt.show()


# 贪心算法：
def GreedyAlgo(item, c, num):
    data = np.array(item)
    index = np.lexsort([data[:, 0], -1 * data[:, 1]])
    status = [0] * num
    Tw = 0
    Tv = 0

    for i in range(num):
        if data[index[i], 0] <= c:
            Tw += data[index[i], 0]
            Tv += data[index[i], 1]
            status[index[i]] = 1
            c -= data[index[i], 0]
        else:
            continue

    print("贪心算法，最大价值为：")
    return Tv


# 动态规划算法
def Dp(w, v, c, num):
    cnt = [0 for j in range(c + 1)]

    for i in range(1, num + 1):
        for j in range(c, 0, -1):
            if j >= w[i - 2]:
                cnt[j] = max(cnt[j], cnt[j - w[i - 2]] + v[i - 2])

    print("动态规划算法，最大价值为：")
    return cnt[c]


# 回溯法
Nv = 0
Nw = 0
Bv = 0
Bw = 0


def Backtracking(k, c, num):
    m = fileoperate()
    a0 = []
    for line in m[0]:
        line = line.split(' ')  # 以空格划分
        a0.append(line)
    a = np.array(a0)
    a = a.astype(int)
    w = (a[:, 0])
    v = (a[:, 1])
    w = np.array(w)
    v = np.array(v)
    w = w.astype(int)
    v = v.astype(int)
    global Nw, Nv, Bv, Bw
    status = [0 for i in range(num)]

    if k >= num:
        if Bv < Nv:
            Bv = Nv
    else:
        if Nw + w[k] <= c:
            status[k] = 1
            Nw += w[k]
            Nv += v[k]
            Backtracking(k + 1, c, num)
            Nw -= w[k]
            Nv -= v[k]
        status[k] = 0
        Backtracking(k + 1, c, num)


N = 500  ##迭代次数
Pc = 0.8  ##交配概率
Pm = 0.15  ##变异概率


## 遗传算法：
# 初始化,N为种群规模，n为染色体长度
def init(N, n):
    C = []
    for i in range(N):
        c = []
        for j in range(n):
            a = np.random.randint(0, 2)
            c.append(a)
        C.append(c)
    return C


##评估函数
# x(i)取值为1表示被选中，取值为0表示未被选中
# w(i)表示各个分量的重量，v（i）表示各个分量的价值，w表示最大承受重量
def fitness(C, N, n, W, V, w):
    S = []  ##用于存储被选中的下标
    F = []  ## 用于存放当前该个体的最大价值
    for i in range(N):
        s = []
        h = 0  # 重量
        f = 0  # 价值
        for j in range(n):
            if C[i][j] == 1:
                if h + W[j] <= w:
                    h = h + W[j]
                    f = f + V[j]
                    s.append(j)
        S.append(s)
        F.append(f)
    return S, F


##适应值函数,B位返回的种族的基因下标，y为返回的最大值
def best_x(F, S, N):
    y = 0
    x = 0
    B = [0] * N
    for i in range(N):
        if y < F[i]:
            x = i
        y = F[x]
        B = S[x]
    return B, y


## 计算比率
def rate(x):
    p = [0] * len(x)
    s = 0
    for i in x:
        s += i
    for i in range(len(x)):
        p[i] = x[i] / s
    return p


## 选择
def chose(p, X, m, n):
    X1 = X
    r = np.random.rand(m)
    for i in range(m):
        k = 0
        for j in range(n):
            k = k + p[j]
            if r[i] <= k:
                X1[i] = X[j]
                break
    return X1


##交配
def match(X, m, n, p):
    r = np.random.rand(m)
    k = [0] * m
    for i in range(m):
        if r[i] < p:
            k[i] = 1
    u = v = 0
    k[0] = k[0] = 0
    for i in range(m):
        if k[i]:
            if k[u] == 0:
                u = i
            elif k[v] == 0:
                v = i
        if k[u] and k[v]:
            # print(u,v)
            q = np.random.randint(n - 1)
            # print(q)
            for i in range(q + 1, n):
                X[u][i], X[v][i] = X[v][i], X[u][i]
            k[u] = 0
            k[v] = 0
    return X


##变异
def vari(X, m, n, p):
    for i in range(m):
        for j in range(n):
            q = np.random.rand()
            if q < p:
                X[i][j] = np.random.randint(0, 2)

    return X


# 排序
def sort():
    a0 = []
    descending = []
    # 计算重量与价值的比值
    m = fileoperate()
    for line in m[0]:
        line = line.split(' ')
        a0.append(line)
    for i in range(len(a0)):
        F0 = int(a0[i][0])
        S0 = int(a0[i][1])
        T0 = F0 / S0
        descending.append(T0)
    for item in descending:
        LB1.insert(END, item)
    print("非递增排序前为：")
    print(descending)
    descending.sort(reverse=True)
    print("非递增排序后为：")
    print(descending)
    for item in descending:
        LB2.insert(END, item)
    write_log_to_Text("按照重量/质量非递增排列成功")


# 日志动态打印：
def write_log_to_Text(logmsg):
    global LOG_LINE_NUM
    current_time = get_current_time()
    logmsg_in = str(current_time) + " " + str(logmsg) + "\n"  # 换行
    if LOG_LINE_NUM <= 7:
        log_txt.insert(END, logmsg_in)
        LOG_LINE_NUM = LOG_LINE_NUM + 1
    else:
        log_txt.delete(1.0, 2.0)
        log_txt.insert(END, logmsg_in)


# 获取当前时间
def get_current_time():
    current_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    return current_time


def goes():  # 处理事件
    a0 = []
    a1 = []
    m = fileoperate()
    for line in m[0]:
        line = line.split(' ')  # 以空格划分
        a0.append(line)
    for line in m[1]:
        line = line.split(' ')
        a1.append(line)

    c = int(a1[0][0])
    num = int(a1[0][1])

    a = np.array(a0)
    a = a.astype(int)
    w = (a[:, 0])
    v = (a[:, 1])
    w = np.array(w)
    v = np.array(v)
    w = w.astype(int)
    v = v.astype(int)
    item = list(zip(w, v))

    if Contain.get() == "贪心算法":
        time_start = timer()
        ret1 = GreedyAlgo(item, c, num)
        print(ret1)
        time_end = timer()
        time_sum = time_end - time_start
        print("求解时间为：")
        print(time_sum)
        data_Text.delete(1.0, END)
        data_Text.insert(1.0, '最优解为：')
        data_Text.insert(1.8, ret1)
        data_Text.insert(2.0, '\n')
        data_Text.insert(2.0, '运行时间为：')
        data_Text.insert(2.8, time_sum)

        result = "贪心算法：最优解：" + str(ret1) + "，" + "求解时间：" + str(time_sum)
        file = open('resultG.txt', 'w')
        file.write(result)
        file.close()
        write_log_to_Text("使用贪心算法求解成功")
    elif Contain.get() == "动态规划算法":
        time_start = timer()
        ret2 = Dp(w, v, c, num)
        print(ret2)
        time_end = timer()
        time_sum = time_end - time_start
        print("求解时间为：")
        print(time_sum)
        data_Text.delete(1.0, END)
        data_Text.insert(1.0, '最优解为：')
        data_Text.insert(1.8, ret2)
        data_Text.insert(2.0, '\n')
        data_Text.insert(2.0, '运行时间为：')
        data_Text.insert(2.8, time_sum)
        result = "动态规划：最优解：" + str(ret2) + "，" + "求解时间：" + str(time_sum)
        file = open('resultD.txt', 'w')
        file.write(result)
        file.close()
        write_log_to_Text("使用动态规划算法求解成功")
    elif Contain.get() == "回溯法":
        time_start = timer()
        Backtracking(0, c, num)
        print(Bv)
        time_end = timer()
        time_sum = time_end - time_start
        print("求解时间为：")
        print(time_sum)
        data_Text.delete(1.0, END)
        data_Text.insert(1.0, '最优解为：')
        data_Text.insert(1.8, Bv)
        data_Text.insert(2.0, '\n')
        data_Text.insert(2.0, '运行时间为：')
        data_Text.insert(2.8, time_sum)
        result = "回溯法：最优解：" + str(Bv) + "，" + "求解时间：" + str(time_sum)
        file = open('resultD.txt', 'w')
        file.write(result)
        file.close()
        write_log_to_Text("使用回溯法求解成功")
    elif Contain.get() == "遗传算法":
        time_start = timer()
        n = len(w)
        C = init(num, n)
        S, F = fitness(C, num, n, w, v, c)
        B, y = best_x(F, S, num)
        Y = [y]
        for i in range(N):
            p = rate(F)
            C = chose(p, C, num, n)
            C = match(C, num, n, Pc)
            C = vari(C, num, n, Pm)
            S, F = fitness(C, num, n, w, v, c)
            B1, y1 = best_x(F, S, num)
            if y1 > y:
                y = y1
            Y.append(y)
        print("遗传算法，最大价值为：")
        print(y)
        time_end = timer()
        time_sum = time_end - time_start
        print("求解时间为：")
        print(time_sum)
        data_Text.delete(1.0, END)
        data_Text.insert(1.0, '最优解为：')
        data_Text.insert(1.8, y)
        data_Text.insert(2.0, '\n')
        data_Text.insert(2.0, '运行时间为：')
        data_Text.insert(2.8, time_sum)
        result = "遗传算法：最优解：" + str(y) + "，" + "求解时间：" + str(time_sum)
        file = open('resultR.txt', 'w')
        file.write(result)
        file.close()
        write_log_to_Text("使用遗传算法求解成功")


# 清屏
def clear():
    LB1.delete(0, END)
    LB2.delete(0, END)

def db():
    # 数据库
    sqlconn = pyodbc.connect(DRIVER='{ODBC Driver 17 for SQL Server}',
                                 SERVER='xueli',
                                 DATABASE='SR',
                                 Trusted_Connection='yes'
                                 )
    # 连接数据库
    sql_createTb = "CREATE TABLE  yk(\
                       Weight varchar(32) not null,\
                       Value varchar(255) not null,)\
                        "

    cursor = sqlconn.cursor()  # 打开游标
    # cursor.execute("CREATE TABLE"+tablename+" "+"(\
    #                    Weight varchar(32) not null,\
    #                    Value varchar(255) not null,)\
    #                    ")  # 执行SQL语句
    cursor.execute(sql_createTb)  # 执行SQL语句
    with open(filename.get()) as f:
        datas = f.readlines()[1:]

    for data in datas:
        txt = re.split(r'[;,\s]\s*', data)
        Weight = txt[0]
        Value = txt[1]
        cursor.execute("INSERT INTO yk(Weight,Value)VALUES('%s','%s')" % (Weight, Value))
    print("数据库插入完成！")
    cursor.execute("SELECT * FROM  yk")  # 执行sql语句
    queryResult = cursor.fetchall()  # 查询执行的sql操作
    print(queryResult)
    sqlconn.commit()
    cursor.close()  # 关闭游标
    sqlconn.close()  # 关闭数据库连接
    write_log_to_Text("成功写入数据库")


Choose = StringVar()
Contain = Combobox(frameX, textvariable=Choose, width=19)
Contain.grid(row=0, column=1, pady=20)
Contain["values"] = ("点击选择算法", "贪心算法", "动态规划算法", "回溯法", "遗传算法")
Contain.current(0)
Button(frameX, text='求最优解', width=20, height=2, command=goes).grid(row=1, column=1, pady=20)
Button(frameX, text='画散点图', width=20, height=2, command=painter).grid(row=2, column=1, pady=15)  # 画散点图按钮
Button(frameX, text='重量比排序', width=20, height=2, command=sort).grid(row=3, column=1, pady=15)  # 排序按钮
Button(frameXy, text='选择文件', width=10, command=fileopen).grid(row=0, column=1, pady=10, padx=4)  # 选择文件
Button(frameXy, text='提交到数据库', width=10, command=db).grid(row=0, column=2, pady=10, padx=10)
Cleat = Button(frameY, text="清屏", width=20, command=clear)
Cleat.grid(row=4, column=1, columnspan=3)

# 文本框
data_Text = Text(frameY, width=50, height=2)  # 结果输出框
data_Text.grid(row=0, column=1, columnspan=3)

# 日志
Label(frameYx, text="日志: ").grid(row=0, column=0)
log_txt = Text(frameYx, width=90, height=7)  # 日志输出框
log_txt.grid(row=1, column=0, columnspan=30, padx=8)

# 列表框（用来排序的输出）
LB1 = Listbox(frameY)
LB2 = Listbox(frameY)
LB1.grid(row=1, column=1, pady=10, rowspan=3)
LB2.grid(row=1, column=3, pady=10, rowspan=3)

# 标签框：
Label(frameY, text="最优解：").grid(row=0, column=0, padx=10)
Label(frameY, text="排序前：").grid(row=1, column=0)
Label(frameY, text="排序后：").grid(row=1, column=2)

# 设置根窗口默认属性
init_window.mainloop()  # 父窗口进入事件循环，可以理解为保持窗口运行，否则界面不展示
