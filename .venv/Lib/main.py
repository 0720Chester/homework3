import taichi as ti
import numpy as np

# 初始化Taichi
ti.init(arch=ti.gpu)

# 常量定义
WIDTH, HEIGHT = 800, 800
NUM_SEGMENTS = 1000  # 曲线采样点数
MAX_CONTROL_POINTS = 100
CURVE_POINTS_SIZE = NUM_SEGMENTS + 1  # 曲线上的点数量

# 颜色定义
WHITE = ti.Vector([1.0, 1.0, 1.0])
BLACK = ti.Vector([0.0, 0.0, 0.0])
RED = ti.Vector([1.0, 0.0, 0.0])
GREEN = ti.Vector([0.0, 1.0, 0.0])
GRAY = ti.Vector([0.5, 0.5, 0.5])

# GPU缓冲区
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(WIDTH, HEIGHT))
curve_points_field = ti.Vector.field(2, dtype=ti.f32, shape=CURVE_POINTS_SIZE)
gui_points = ti.Vector.field(2, dtype=ti.f32, shape=MAX_CONTROL_POINTS)


@ti.func
def draw_pixel(x, y, color):
    """在指定像素位置绘制颜色"""
    if 0 <= x < WIDTH and 0 <= y < HEIGHT:
        pixels[x, y] = color


@ti.kernel
def draw_curve_kernel(n: ti.i32):
    """GPU内核：绘制曲线"""
    for i in range(n):
        pos = curve_points_field[i]
        # 将归一化坐标映射到像素坐标
        px = int(pos[0] * WIDTH)
        py = int(pos[1] * HEIGHT)
        draw_pixel(px, py, GREEN)


@ti.kernel
def draw_control_points_kernel(n: ti.i32):
    """GPU内核：绘制控制点"""
    # 绘制控制点连线（控制多边形）
    for i in range(n - 1):
        p1 = gui_points[i]
        p2 = gui_points[i + 1]
        # 只绘制可见的点（不在屏幕外）
        if p1[0] >= 0 and p1[1] >= 0 and p2[0] >= 0 and p2[1] >= 0:
            # 使用Bresenham算法绘制线段
            x1 = int(p1[0] * WIDTH)
            y1 = int(p1[1] * HEIGHT)
            x2 = int(p2[0] * WIDTH)
            y2 = int(p2[1] * HEIGHT)

            # Bresenham线段绘制算法
            dx = abs(x2 - x1)
            dy = abs(y2 - y1)
            sx = 1 if x1 < x2 else -1
            sy = 1 if y1 < y2 else -1
            err = dx - dy

            while True:
                if 0 <= x1 < WIDTH and 0 <= y1 < HEIGHT:
                    pixels[x1, y1] = GRAY
                if x1 == x2 and y1 == y2:
                    break
                e2 = 2 * err
                if e2 > -dy:
                    err -= dy
                    x1 += sx
                if e2 < dx:
                    err += dx
                    y1 += sy

    # 绘制控制点（红色圆点）
    for i in range(n):
        pos = gui_points[i]
        if pos[0] >= 0 and pos[1] >= 0:
            cx = int(pos[0] * WIDTH)
            cy = int(pos[1] * HEIGHT)
            radius = 5
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if dx * dx + dy * dy <= radius * radius:
                        x = cx + dx
                        y = cy + dy
                        if 0 <= x < WIDTH and 0 <= y < HEIGHT:
                            pixels[x, y] = RED


@ti.kernel
def clear_pixels():
    """清空屏幕"""
    for i, j in pixels:
        pixels[i, j] = BLACK


def de_casteljau(points, t):
    """
    De Casteljau算法计算贝塞尔曲线上的点
    points: 控制点列表
    t: 参数 [0, 1]
    返回: 曲线上的点坐标 [x, y]
    """
    if not points:
        return [0, 0]

    # 复制控制点列表
    temp_points = points.copy()

    # 递归插值直到只剩一个点
    while len(temp_points) > 1:
        new_points = []
        for i in range(len(temp_points) - 1):
            # 线性插值
            x = (1 - t) * temp_points[i][0] + t * temp_points[i + 1][0]
            y = (1 - t) * temp_points[i][1] + t * temp_points[i + 1][1]
            new_points.append([x, y])
        temp_points = new_points

    return temp_points[0]


def generate_curve_points(control_points):
    """
    生成贝塞尔曲线上的所有点
    control_points: 控制点列表
    返回: numpy数组，包含曲线上的点
    """
    if len(control_points) < 2:
        return np.zeros((CURVE_POINTS_SIZE, 2), dtype=np.float32)

    curve_points = []
    for i in range(CURVE_POINTS_SIZE):
        t = i / (CURVE_POINTS_SIZE - 1)
        point = de_casteljau(control_points, t)
        curve_points.append(point)

    return np.array(curve_points, dtype=np.float32)


def prepare_gui_points(control_points):
    """
    准备GUI控制点数据（对象池技巧）
    返回: numpy数组，大小为MAX_CONTROL_POINTS，无效点放在屏幕外
    """
    gui_points_array = np.full((MAX_CONTROL_POINTS, 2), -10.0, dtype=np.float32)

    for i, point in enumerate(control_points):
        if i < MAX_CONTROL_POINTS:
            gui_points_array[i] = point

    return gui_points_array


def main():
    # 创建窗口
    window = ti.ui.Window("贝塞尔曲线绘制 - De Casteljau算法", (WIDTH, HEIGHT))
    canvas = window.get_canvas()

    # 控制点列表（在CPU端维护）
    control_points = []

    print("=" * 50)
    print("贝塞尔曲线绘制程序")
    print("=" * 50)
    print("操作说明:")
    print("  - 鼠标左键点击: 添加控制点")
    print("  - 按 'C' 键: 清空所有控制点")
    print("  - 按 'ESC' 键: 退出程序")
    print("=" * 50)

    # 主循环
    while window.running:
        # 清空屏幕
        clear_pixels()

        # 处理交互事件
        for event in window.get_events(ti.ui.PRESS):
            if event.key == ti.ui.ESCAPE:
                window.running = False
                break
            elif event.key == 'c':
                # 清空控制点
                control_points.clear()
                print(f"\n✓ 已清空所有控制点")
            elif event.key == ti.ui.LMB:
                # 获取鼠标位置并添加控制点
                mouse_pos = window.get_cursor_pos()
                # 将鼠标位置从屏幕坐标转换为归一化坐标
                if len(control_points) < MAX_CONTROL_POINTS:
                    control_points.append([mouse_pos[0], mouse_pos[1]])
                    print(f"添加控制点 {len(control_points)}: ({mouse_pos[0]:.3f}, {mouse_pos[1]:.3f})")

        # 绘制控制点和控制多边形
        if len(control_points) > 0:
            gui_points_array = prepare_gui_points(control_points)
            gui_points.from_numpy(gui_points_array)
            draw_control_points_kernel(len(control_points))

        # 绘制贝塞尔曲线
        if len(control_points) >= 2:
            # 在CPU端生成曲线上的所有点
            curve_points_array = generate_curve_points(control_points)
            # 批量拷贝到GPU
            curve_points_field.from_numpy(curve_points_array)
            # 调用GPU内核绘制曲线
            draw_curve_kernel(CURVE_POINTS_SIZE)

        # 显示画面
        canvas.set_image(pixels)
        window.show()


    print("\n\n程序退出")


if __name__ == "__main__":
    main()