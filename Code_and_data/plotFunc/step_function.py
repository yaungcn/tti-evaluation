import numpy as np
import matplotlib.pyplot as plt


def step_function(x, step_point=0, value_before=0, value_after=1, smooth_sigma=0):
    """
    可自定义的阶跃函数
    
    Parameters:
    -----------
    x : array_like
        输入值
    step_point : float, optional
        阶跃点位置，默认为0
    value_before : float, optional
        阶跃前的值，默认为0
    value_after : float, optional
        阶跃后的值，默认为1
    smooth_sigma : float, optional
        平滑度参数，0表示硬阶跃，大于0表示平滑阶跃（Sigmoid类型），默认为0
        
    Returns:
    --------
    ndarray
        函数值
    """
    if smooth_sigma == 0:
        return np.where(x < step_point, value_before, value_after)
    else:
        sigmoid = 1 / (1 + np.exp(-(x - step_point) / smooth_sigma))
        return value_before + (value_after - value_before) * sigmoid


def smooth_step_function(x, center=0, transition_width=0.1, value_before=0, value_after=1):
    """
    以位置和过渡区大小定义的平滑阶跃函数
    
    函数在 [center - transition_width/2, center + transition_width/2] 范围内平滑过渡
    使用平滑的三次多项式（smoothstep）实现
    
    Parameters:
    -----------
    x : array_like
        输入值
    center : float, optional
        阶跃中心位置，默认为0
    transition_width : float, optional
        过渡区宽度，默认为0.1
        值为0时表示硬阶跃
    value_before : float, optional
        阶跃前的值，默认为0
    value_after : float, optional
        阶跃后的值，默认为1
        
    Returns:
    --------
    ndarray
        函数值
    """
    if transition_width == 0:
        return np.where(x < center, value_before, value_after)
    
    normalized = (x - center) / transition_width + 0.5
    mask_before = normalized <= 0
    mask_after = normalized >= 1
    mask_transition = ~mask_before & ~mask_after
    
    result = np.zeros_like(x)
    result[mask_before] = value_before
    result[mask_after] = value_after
    
    t = normalized[mask_transition]
    result[mask_transition] = value_before + (value_after - value_before) * (3*t**2 - 2*t**3)
    
    return result


def visualize_step(x_range=(-5, 5), num_points=1000, step_point=0, value_before=0, 
                   value_after=1, smooth_sigma=0, title="Step Function", 
                   xlabel="x", ylabel="f(x)", grid=True, figsize=(10, 6)):
    """
    可视化阶跃函数
    
    Parameters:
    -----------
    x_range : tuple, optional
        x轴范围，默认为(-5, 5)
    num_points : int, optional
        采样点数，默认为1000
    step_point : float, optional
        阶跃点位置，默认为0
    value_before : float, optional
        阶跃前的值，默认为0
    value_after : float, optional
        阶跃后的值，默认为1
    smooth_sigma : float, optional
        平滑度参数，默认为0
    title : str, optional
        图表标题，默认为"Step Function"
    xlabel : str, optional
        x轴标签，默认为"x"
    ylabel : str, optional
        y轴标签，默认为"f(x)"
    grid : bool, optional
        是否显示网格，默认为True
    figsize : tuple, optional
        图表大小，默认为(10, 6)
        
    Returns:
    --------
    tuple
        (figure, axes)
    """
    x = np.linspace(x_range[0], x_range[1], num_points)
    y = step_function(x, step_point, value_before, value_after, smooth_sigma)
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(x, y, linewidth=2)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if grid:
        ax.grid(True, alpha=0.3)
    ax.axvline(x=step_point, color='red', linestyle='--', alpha=0.5, label=f'Step at x={step_point}')
    ax.legend()
    
    return fig, ax


def visualize_smooth_step(x_range=(-5, 5), num_points=1000, center=0, 
                          transition_width=0.1, value_before=0, value_after=1,
                          title="Smooth Step Function", xlabel="x", ylabel="f(x)", 
                          grid=True, figsize=(10, 6)):
    """
    可视化平滑阶跃函数（基于过渡区宽度）
    
    Parameters:
    -----------
    x_range : tuple, optional
        x轴范围，默认为(-5, 5)
    num_points : int, optional
        采样点数，默认为1000
    center : float, optional
        阶跃中心位置，默认为0
    transition_width : float, optional
        过渡区宽度，默认为0.1
    value_before : float, optional
        阶跃前的值，默认为0
    value_after : float, optional
        阶跃后的值，默认为1
    title : str, optional
        图表标题，默认为"Smooth Step Function"
    xlabel : str, optional
        x轴标签，默认为"x"
    ylabel : str, optional
        y轴标签，默认为"f(x)"
    grid : bool, optional
        是否显示网格，默认为True
    figsize : tuple, optional
        图表大小，默认为(10, 6)
        
    Returns:
    --------
    tuple
        (figure, axes)
    """
    x = np.linspace(x_range[0], x_range[1], num_points)
    y = smooth_step_function(x, center, transition_width, value_before, value_after)
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(x, y, linewidth=2, color='blue')
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if grid:
        ax.grid(True, alpha=0.3)
    
    ax.axvline(x=center, color='red', linestyle='--', alpha=0.5, label=f'中心 x={center}')
    if transition_width > 0:
        ax.axvspan(center - transition_width/2, center + transition_width/2, 
                   alpha=0.1, color='green', label=f'过渡区 宽度={transition_width}')
    
    ax.legend()
    
    return fig, ax


def combine_step_functions(x, func_list, operation=None, use_smooth_step=False):
    """
    组合多个阶跃函数
    
    Parameters:
    -----------
    x : array_like
        输入值
    func_list : list of dict or list of callable
        阶跃函数列表，每个元素可以是：
        - dict: 包含 step_point/center, value_before, value_after, smooth_sigma/transition_width 的字典
        - callable: 直接的函数
    operation : callable, optional
        组合操作函数，默认为乘法
        例如: lambda results: np.prod(results, axis=0) 或 lambda results: results[0] * (1 - results[1])
    use_smooth_step : bool, optional
        是否使用基于过渡区宽度的平滑阶跃函数，默认为False
        
    Returns:
    --------
    ndarray
        组合后的函数值
    """
    results = []
    for item in func_list:
        if callable(item):
            results.append(item(x))
        elif isinstance(item, dict):
            if use_smooth_step:
                center = item.get('center', 0)
                transition_width = item.get('transition_width', 0.1)
                value_before = item.get('value_before', 0)
                value_after = item.get('value_after', 1)
                results.append(smooth_step_function(x, center, transition_width, value_before, value_after))
            else:
                step_point = item.get('step_point', 0)
                value_before = item.get('value_before', 0)
                value_after = item.get('value_after', 1)
                smooth_sigma = item.get('smooth_sigma', 0)
                results.append(step_function(x, step_point, value_before, value_after, smooth_sigma))
    
    if operation is None:
        return np.prod(results, axis=0)
    else:
        return operation(results)


def visualize_combined_step(x_range=(-2, 3), num_points=1000, func_list=None, 
                            operation=None, title="Combined Step Function",
                            xlabel="x", ylabel="f(x)", grid=True, figsize=(10, 6),
                            use_smooth_step=False):
    """
    可视化组合后的阶跃函数
    
    Parameters:
    -----------
    x_range : tuple, optional
        x轴范围，默认为(-2, 3)
    num_points : int, optional
        采样点数，默认为1000
    func_list : list, optional
        阶跃函数列表
    operation : callable, optional
        组合操作函数
    title : str, optional
        图表标题
    xlabel : str, optional
        x轴标签，默认为"x"
    ylabel : str, optional
        y轴标签，默认为"f(x)"
    grid : bool, optional
        是否显示网格，默认为True
    figsize : tuple, optional
        图表大小，默认为(10, 6)
    use_smooth_step : bool, optional
        是否使用基于过渡区宽度的平滑阶跃函数，默认为False
        
    Returns:
    --------
    tuple
        (figure, axes)
    """
    x = np.linspace(x_range[0], x_range[1], num_points)
    y = combine_step_functions(x, func_list, operation, use_smooth_step)
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(x, y, linewidth=2, color='blue')
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if grid:
        ax.grid(True, alpha=0.3)
    
    for item in func_list:
        if isinstance(item, dict):
            if use_smooth_step:
                center = item.get('center', 0)
                transition_width = item.get('transition_width', 0.1)
                ax.axvline(x=center, color='red', linestyle='--', alpha=0.5)
                if transition_width > 0:
                    ax.axvspan(center - transition_width/2, center + transition_width/2, 
                              alpha=0.1, color='green')
            else:
                step_point = item.get('step_point', 0)
                ax.axvline(x=step_point, color='red', linestyle='--', alpha=0.5)
    
    return fig, ax


if __name__ == "__main__":
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    print("示例1: 硬阶跃函数")
    visualize_step(step_point=0, value_before=0, value_after=1, smooth_sigma=0, 
                   title="硬阶跃函数")
    # plt.show()
    
    print("\n示例2: 平滑阶跃函数 (Sigmoid)")
    visualize_step(step_point=0, value_before=0, value_after=1, smooth_sigma=0.5, 
                   title="平滑阶跃函数 (σ=0.5)")
    # plt.show()
    
    print("\n示例3: 多个平滑度对比")
    x = np.linspace(-5, 5, 1000)
    plt.figure(figsize=(10, 6))
    sigmas = [0, 0.2, 0.5, 1]
    colors = ['blue', 'orange', 'green', 'red']
    for sigma, color in zip(sigmas, colors):
        y = step_function(x, step_point=0, value_before=0, value_after=1, smooth_sigma=sigma)
        label = f'σ={sigma}' if sigma > 0 else '硬阶跃'
        plt.plot(x, y, linewidth=2, label=label, color=color)
    plt.title('不同平滑度的阶跃函数对比')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()
    
    print("\n示例4: 自定义阶跃点和阶跃值")
    visualize_step(x_range=(-2, 6), step_point=2, value_before=1, value_after=3, smooth_sigma=0.3, 
                   title="自定义阶跃函数 (阶跃点=2, 阶跃前=1, 阶跃后=3)")
    # plt.show()
    
    print("\n示例5: 阶跃函数组合 - 矩形脉冲")
    print("组合函数: step1(x) * (1 - step1(x-1))")
    print("效果: 在区间[0, 1]内为1，其他地方为0")
    
    func_list = [
        {'step_point': 0, 'value_before': 0, 'value_after': 1, 'smooth_sigma': 0},
        {'step_point': 1, 'value_before': 0, 'value_after': 1, 'smooth_sigma': 0}
    ]
    operation = lambda results: results[0] * (1 - results[1])
    
    visualize_combined_step(x_range=(-1, 2), func_list=func_list, operation=operation,
                           title="阶跃函数组合 - 矩形脉冲 (step1(x)*(1-step1(x-1)))")
    # plt.show()
    
    print("\n示例6: 平滑矩形脉冲")
    func_list_smooth = [
        {'step_point': 0, 'value_before': 0, 'value_after': 1, 'smooth_sigma': 0.1},
        {'step_point': 1, 'value_before': 0, 'value_after': 1, 'smooth_sigma': 0.1}
    ]
    operation = lambda results: results[0] * (1 - results[1])
    
    visualize_combined_step(x_range=(-0.5, 1.5), func_list=func_list_smooth, operation=operation,
                           title="平滑矩形脉冲 (σ=0.1)")
    # plt.show()
    
    print("\n示例7: 多个矩形脉冲")
    x = np.linspace(-1, 5, 1000)
    plt.figure(figsize=(10, 6))
    
    pulses = [
        {'start': 0, 'end': 1, 'sigma': 0},
        {'start': 2, 'end': 3, 'sigma': 0},
        {'start': 3.5, 'end': 4.5, 'sigma': 0}
    ]
    
    for i, pulse in enumerate(pulses):
        func_list = [
            {'step_point': pulse['start'], 'value_before': 0, 'value_after': 1, 'smooth_sigma': pulse['sigma']},
            {'step_point': pulse['end'], 'value_before': 0, 'value_after': 1, 'smooth_sigma': pulse['sigma']}
        ]
        operation = lambda results: results[0] * (1 - results[1])
        y = combine_step_functions(x, func_list, operation)
        plt.plot(x, y, linewidth=2, label=f'脉冲 {i+1}: [{pulse["start"]}, {pulse["end"]}]')
    
    y_combined = sum(combine_step_functions(x, [
        {'step_point': p['start'], 'value_before': 0, 'value_after': 1, 'smooth_sigma': p['sigma']},
        {'step_point': p['end'], 'value_before': 0, 'value_after': 1, 'smooth_sigma': p['sigma']}
    ], lambda results: results[0] * (1 - results[1])) for p in pulses)
    
    plt.plot(x, y_combined, linewidth=3, color='black', label='组合叠加')
    plt.title('多个矩形脉冲及其叠加')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()
    
    print("\n示例8: 基于过渡区宽度的平滑阶跃函数")
    visualize_smooth_step(x_range=(-1, 2), center=0.5, transition_width=0.4, 
                          title="平滑阶跃函数 (中心=0.5, 过渡区宽度=0.4)")
    # plt.show()
    
    print("\n示例9: 不同过渡区宽度对比")
    x = np.linspace(-1, 2, 1000)
    plt.figure(figsize=(10, 6))
    widths = [0, 0.1, 0.3, 0.6]
    colors = ['blue', 'orange', 'green', 'red']
    for width, color in zip(widths, colors):
        y = smooth_step_function(x, center=0.5, transition_width=width)
        label = f'宽度={width}' if width > 0 else '硬阶跃'
        plt.plot(x, y, linewidth=2, label=label, color=color)
    plt.axvspan(0.5 - 0.3, 0.5 + 0.3, alpha=0.1, color='green', label='过渡区示例')
    plt.title('不同过渡区宽度的阶跃函数对比')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    # plt.show()
    
    print("\n示例10: 使用过渡区宽度的矩形脉冲组合")
    func_list_smooth2 = [
        {'center': 0, 'transition_width': 0.3, 'value_before': 0, 'value_after': 1},
        {'center': 1, 'transition_width': 0.3, 'value_before': 0, 'value_after': 1}
    ]
    operation = lambda results: results[0] * (1 - results[1])
    
    visualize_combined_step(x_range=(-0.5, 1.5), func_list=func_list_smooth2, operation=operation,
                           title="基于过渡区宽度的矩形脉冲组合", use_smooth_step=True)
    plt.show()
    
    print("\n示例11: Sigmoid vs Smoothstep 对比")
    x = np.linspace(-1, 2, 1000)
    plt.figure(figsize=(10, 6))
    
    y_sigmoid = step_function(x, step_point=0.5, smooth_sigma=0.1)
    y_smoothstep = smooth_step_function(x, center=0.5, transition_width=0.5)
    
    plt.plot(x, y_sigmoid, linewidth=2, label='Sigmoid (σ=0.1)', color='blue')
    plt.plot(x, y_smoothstep, linewidth=2, label='Smoothstep (宽度=0.5)', color='red', linestyle='--')
    
    plt.axvline(x=0.5, color='green', linestyle=':', alpha=0.7, label='阶跃中心')
    plt.axvspan(0.5 - 0.25, 0.5 + 0.25, alpha=0.1, color='red', label='Smoothstep过渡区')
    
    plt.title('Sigmoid 与 Smoothstep 对比')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()
    
    print("\n示例12: 多个不同过渡区的脉冲组合")
    func_list_multi = [
        {'center': 0.5, 'transition_width': 0.2, 'value_before': 0, 'value_after': 1},
        {'center': 1.5, 'transition_width': 0.2, 'value_before': 0, 'value_after': 1},
        {'center': 2.5, 'transition_width': 0.4, 'value_before': 0, 'value_after': 1},
        {'center': 3.5, 'transition_width': 0.4, 'value_before': 0, 'value_after': 1}
    ]
    
    def pulse_operation(results):
        return sum(results[i] * (1 - results[i+1]) for i in range(0, len(results), 2))
    
    visualize_combined_step(x_range=(0, 4), func_list=func_list_multi, operation=pulse_operation,
                           title="多个不同过渡区的脉冲组合", use_smooth_step=True)
    # plt.show()
