#%% import
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#%% cfg
f_mode = "F" # Fv, Fh, F
mode = "a" # k, a, theta
loop_num = 5
sample_num = 100

THETA_MODE = "angle"
MAX_K = 1
MAX_A = 2

min_a_dic = {
    "k": 0,
    "a": 1e-8,
    "theta": 0
}
MIN_A = min_a_dic[mode]

#%% funcs
def range_list_to_range(range_list):
    sample_num = range_list[2]
    L = range_list[1] - range_list[0]
    strid = L / sample_num

    for i in range(sample_num+1):
        yield range_list[0] + i*strid

def angle_to_radians(angle):
    """将角度转换为弧度"""
    return np.pi * angle / 180.0

def radians_to_angle(radians):
    """将弧度转换为角度"""
    return radians * 180 / np.pi


def fix_range(range_list:list, defualt_sample_num=100):
    """修复网格参数"""
    n = len(range_list)
    if n > 2:
        return range_list
    
    if n < 2:
        print("error | bad range | need at lest 2 float num !!! | now range_list is",range_list)
        return None
    
    range_list.append(defualt_sample_num)
    return range_list

def get_x_y_z(x_range:list[float], y_range:list[float], func: lambda x,y: x+y, sample_num = 50):
    """创建网格数据"""
    x_range = fix_range(x_range, sample_num)
    if x_range is None:
        return None, None, None
    
    y_range = fix_range(y_range)
    if y_range is None:
        return None, None, None

    x = np.linspace(x_range[0], x_range[1], x_range[2])
    y = np.linspace(y_range[0], y_range[1], y_range[2])

    x, y = np.meshgrid(x, y)
    z = func(x, y)

    return x,y,z

def draw_3d_func_by_xyz(
        x, y, z, 
        title="title", x_name='X axis', y_name='Y axis', z_name='Z axis',
        fig = None, ax = None, show = True
):
    """根据网格数据作图"""
    # 创建3D图形
    if fig is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    # 绘制3D表面图
    surf = ax.plot_surface(x, y, z, cmap='viridis')

    # 添加颜色条
    fig.colorbar(surf, shrink=0.5, aspect=5)

    # 设置轴标签
    ax.set_title(title)
    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    ax.set_zlabel(z_name)

    # 显示图形
    if show:
        plt.show()

    return fig, ax

def draw_3d_func(
        x_range:list[float], y_range:list[float], func: lambda x,y: x+y, 
        title="title", x_name='X axis', y_name='Y axis', z_name='Z axis', 
        sample_num = 50,
        fig = None, ax = None, show = True
):
    """根据 x,y 的网格参数 和 函数 z=func(x,y) 作图"""
    x,y,z = get_x_y_z(x_range, y_range, func, sample_num)
    if x is None:
        return None, None
    
    return draw_3d_func_by_xyz(x, y, z, title, x_name, y_name, z_name, fig, ax, show)

def draw_3d_line_by_xyz(
        x, y, z, 
        line_color="red", linewidth=1.0, 
        title=None, x_name=None, y_name=None, z_name=None,
        fig = None, ax = None, show = True, 
        zorder=10 # 提高线显示的优先级
):
    # 创建3D图形
    if fig is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    
    # 绘制3D线
    ax.plot(x, y, z, color=line_color, linewidth=linewidth, zorder=zorder)

    # 设置轴标签
    if title is not None:
        ax.set_title(title)
        ax.set_xlabel(x_name)
        ax.set_ylabel(y_name)
        ax.set_zlabel(z_name)

    # 显示图形
    if show:
        plt.show()
    
    return fig, ax
#%% 核心运算函数 A(k,a,θ)
def calc_A(k, a, theta):
    c = np.cos(theta)
    s = np.sin(theta)
    return k*(a+c*c)-c*s

#%% 应变量函数 F|Fv|Fh
# 应变量函数 Fv
def Fv_human_by_radians(k, a, theta):
    b = (1+k*k)*a
    return k*calc_A(k, a, theta)/b

# 应变量函数 Fh
def Fh_human_by_radians(k, a, theta):
    b = (1+k*k)*a
    return calc_A(k, a, theta)/b

# 应变量函数 F
def F_human_by_radians(k, a, theta):
    b = np.sqrt((1+k*k))*a
    F1 = np.abs(calc_A(k, a, theta))/b
    F2 = k
    return np.minimum(F1, F2)

# 应变量函数工厂
F_HUMAN_FACTORY = {
    "Fv": Fv_human_by_radians,
    "Fh": Fh_human_by_radians,
    "F": F_human_by_radians
}

# 应变量函数工厂 - 角度制|弧度制
def F_human(k, a, theta, F_mode=f_mode, theta_mode=THETA_MODE):
    if theta_mode == "angle":
        return F_HUMAN_FACTORY[F_mode](k, a, angle_to_radians(theta))
    return F_HUMAN_FACTORY[F_mode](k, a, theta)


#%% 不同参数的定义域
def get_k_D(max_k = MAX_K):
    return [0, max_k]

def get_a_D(max_a = MAX_A):
    return [MIN_A, max_a]

def get_theta_D(theta_mode=THETA_MODE):
    if theta_mode == "angle":
        return [0, 90]
    return [0, np.pi/2]

#%% F = 0 交线
# F = 0 对应的 k(a,θ) 函数 - 固定 a 即得 k(θ)
def f0_k_by_a_theta_line(a, theta, theta_mode=THETA_MODE):
    if theta_mode == "angle":
        theta = angle_to_radians(theta)
    c = np.cos(theta)
    s = np.sin(theta)
    return c*s/(a+c*c)

# F = 0 对应的 a(k,θ) 函数 - 固定 k 即得 a(θ)
def f0_a_by_k_theta_line(k, theta, theta_mode=THETA_MODE):
    if theta_mode == "angle":
        theta = angle_to_radians(theta)
    c = np.cos(theta)
    s = np.sin(theta)
    return c*s/k - c*c

# F = 0 对应的 k(θ,a) 函数 - 固定 θ 即得 k(a)
def f0_k_by_theta_a_line(theta, a, theta_mode=THETA_MODE):
    if theta_mode == "angle":
        theta = angle_to_radians(theta)
    c = np.cos(theta)
    s = np.sin(theta)
    return c*s/(a+c*c)

# F = 0 的交线函数工厂
F_EQ_0_FACTORY = {
    "k": f0_a_by_k_theta_line,
    "a": f0_k_by_a_theta_line,
    "theta": f0_k_by_theta_a_line
}

#%% min F 对应的 θ(k) 函数
def min_theta_by_k(k, theta_mode=THETA_MODE):
    ret = 0
    if k == 0:
        ret = 0.25*np.pi
    else:
        ret = 0.5*(np.pi - np.arctan(1/k))
    
    if theta_mode == "angle":
        ret = radians_to_angle(ret)
    
    return ret

# min F 的线
MIN_THETA_FACTORY = {
    "k": lambda k, a: min_theta_by_k(k),
    "a": lambda a, k: min_theta_by_k(k),
    "theta": lambda theta, k: theta
}

#%% 不同模式(控制变量)下的 函数包
F_FACTORY = {
    "k": [
        get_k_D, get_a_D, get_theta_D, lambda k, a, theta: F_human(k, a, theta), 
        ["k","a","θ"]
    ],
    "a": [
        get_a_D, get_k_D, get_theta_D, lambda a, k, theta: F_human(k, a, theta), 
        ["a","k","θ"]
    ],
    "theta": [
        get_theta_D, get_k_D, get_a_D, lambda theta, k, a: F_human(k, a, theta), 
        ["θ","k","a"]
    ]
}

def get_func_range(loop_range, xyz_func):
    """应变量的函数簇"""
    for i in loop_range:
        yield i, lambda x,y: xyz_func(i, x, y)

def param_factory(mode="k", loop_num=5, sample_num=50):
    """获取: 自变量的区间 以及 应变量的函数簇"""
    loop_range_func, p1_range_func, p2_range_func, xyz_func, info = F_FACTORY[mode]
    loop_range = range_list_to_range(fix_range(loop_range_func(), loop_num))
    p1_range = fix_range(p1_range_func(), sample_num)
    p2_range = fix_range(p2_range_func(), sample_num)

    return p1_range, p2_range, get_func_range(loop_range, xyz_func), info

#%% run demo
def Fv_human_by_keep_one_param(mode="k", loop_num=5, sample_num=50):
    x_range, y_range, func_range, info = param_factory(mode, loop_num, sample_num)
    x = list(range_list_to_range(x_range))
    y = list(range_list_to_range(y_range))

    title_base, x_name, y_name = info
    title_base += " = "
    z_name = "F_human_v"
    for i,func in func_range:
        param_bad = i < 0.001 \
            or ( mode == "theta" and (i>=90 if THETA_MODE=="angle" else i>=np.pi/2) )

        # 1. F|Fv|Fh 函数图像
        fig, ax = draw_3d_func(x_range, y_range, func, title_base+str(round(i,4)), x_name, y_name, z_name, show=param_bad)
        if param_bad:
            continue

        # 2. F = 0 线
        mode_is_theta = mode == "theta"

        f0_line = F_EQ_0_FACTORY[mode]
        x_ = []; y_ = []; z_ = []
        for yi in y:
            xi = f0_line(i, yi)
            if xi < 0:
                continue
            x_.append(xi)
            y_.append(yi)
            z_.append(0)
        draw_3d_line_by_xyz(x_, y_, z_, fig=fig, ax=ax, line_color="blue", show=mode_is_theta)

        if mode_is_theta:
            continue

        # 3. min F|Fv|Fh
        f_min_theta = MIN_THETA_FACTORY[mode]
        y_ = [f_min_theta(i, xi) for xi in x]
        z_ = [func(x[i], y_[i]) for i in range(len(x))]
        draw_3d_line_by_xyz(x, y_, z_, fig=fig, ax=ax, line_color="red", show=True)


#%% cls

#%% funcs

#%% test func


#%% main
if __name__ == "__main__":
    print("start testing ...")

    Fv_human_by_keep_one_param(mode, loop_num, sample_num)

    #%% test
    # draw_3d_func([-6,6,30], [-6,6,30], my_function)

    # print([round(i, 2) for i in range_list_to_range([0,1,10])])
    
    # loop_range = range_list_to_range([0,1,10])
    # f = lambda i, x, y: i*(x + y)
    # f_range = get_func_range(loop_range, f)
    # print(f_range)
    # print([func(1,2) for func in f_range])

    # x = [i*0.1 for i in range(100)]
    # y = x
    # z = [i*i for i in range(100)]
    # draw_3d_line_by_xyz(x,y,z)