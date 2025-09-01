import random

def generate_maze(size=10, obstacle_ratio=0.2):
    """生成一个size x size的迷宫，包含随机障碍物"""
    # 初始化迷宫，0表示通路，1表示障碍物
    maze = [[0 for _ in range(size)] for _ in range(size)]
    
    # 起点(0,0)和终点(size-1, size-1)不能是障碍物
    start = (0, 0)
    end = (size-1, size-1)
    
    # 随机放置障碍物
    obstacle_count = int(size * size * obstacle_ratio)
    obstacles_placed = 0
    
    while obstacles_placed < obstacle_count:
        x = random.randint(0, size-1)
        y = random.randint(0, size-1)
        
        # 确保不是起点、终点，且还没放置障碍物
        if (x, y) != start and (x, y) != end and maze[x][y] == 0:
            maze[x][y] = 1
            obstacles_placed += 1
    
    return maze, start, end

def print_maze(maze, player_pos):
    """打印迷宫，显示玩家位置"""
    size = len(maze)
    for i in range(size):
        row = []
        for j in range(size):
            if (i, j) == player_pos:
                row.append("P")  # 玩家位置
            elif maze[i][j] == 1:
                row.append("#")  # 障碍物
            elif (i, j) == (0, 0):
                row.append("S")  # 起点
            elif (i, j) == (size-1, size-1):
                row.append("E")  # 终点
            else:
                row.append(" ")  # 通路
        print("|".join(row))
        print("-" * (size * 2 - 1))

def main():
    size = 10
    maze, start, end = generate_maze(size)
    player_pos = start
    steps = 0
    
    print("欢迎来到迷宫游戏！")
    print("使用WASD键移动，Q键退出游戏")
    print("目标：从起点S到达终点E，避开障碍物#")
    
    while True:
        print(f"\n步数：{steps}")
        print_maze(maze, player_pos)
        
        # 检查是否到达终点
        if player_pos == end:
            print(f"\n恭喜你，你所使用的步数为{steps}。")
            break
        
        # 获取用户输入
        move = input("请输入移动方向(W/A/S/D)：").strip().lower()
        
        if move == 'q':
            print("游戏结束。")
            break
        
        # 计算新位置
        x, y = player_pos
        if move == 'w':  # 上
            new_pos = (x-1, y)
        elif move == 's':  # 下
            new_pos = (x+1, y)
        elif move == 'a':  # 左
            new_pos = (x, y-1)
        elif move == 'd':  # 右
            new_pos = (x, y+1)
        else:
            print("无效的输入，请使用W/A/S/D移动，Q键退出。")
            continue
        
        # 检查新位置是否有效
        new_x, new_y = new_pos
        if 0 <= new_x < size and 0 <= new_y < size and maze[new_x][new_y] == 0:
            player_pos = new_pos
            steps += 1
        else:
            print("无法移动到该位置，请重新选择方向。")

if __name__ == "__main__":
    main()
    