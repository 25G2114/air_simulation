import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# シミュレーションパラメータの定義.
WIDTH = 50.0                  # 空間の幅
HEIGHT = 50.0                 # 空間の高さ

# ★★★ エージェント総数：1000個 ★★★
AGENTS_NUMBER = 1000       
SIMULATION_TIME = 600         # シミュレーション時間
AGENT_SPEED = 1.5             # エージェントの基本移動速度（熱運動）

# 【風のパラメータ】
WIND_SPEED = 3.5              # 風の強さ
WIND_PROBABILITY = 0.6        # 風の影響を受ける確率 (60%)
WIND_ROUTE_WIDTH = 6.0        # 風の通り道の太さ（中心線からの許容距離）

# エージェントの状態の定義
OXYGEN = 0   # 酸素
CO2 = 1      # 二酸化炭素

# ★★★ 開口部（窓・ドア）の座標の範囲（ここを自由に変えても連動します） ★★★
LEFT_WINDOW = (4.0, 12.0)    # 左側の窓の位置
RIGHT_DOOR = (8.0, 16.0)      # 右側のドアの位置

# 【自動計算】窓の中心とドアの中心座標
window_center = np.array([0.0, (LEFT_WINDOW[0] + LEFT_WINDOW[1]) / 2.0])
door_center = np.array([WIDTH, (RIGHT_DOOR[0] + RIGHT_DOOR[1]) / 2.0])

# 【自動計算】窓からドアへの風の方向ベクトル（単位ベクトル）
wind_dir = door_center - window_center
wind_dir = wind_dir / np.linalg.norm(wind_dir)  # 正規化

# 初期配置と状態の初期化
pos = np.random.rand(AGENTS_NUMBER, 2) * [WIDTH, HEIGHT]

# 初期割合を 酸素 0 : 二酸化炭素 10 に設定
states = np.ones(AGENTS_NUMBER, dtype=np.int8) * CO2

# ログ記録用リスト
log_time = []
log_o2_count = []
log_co2_count = []

# 描画設定
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# 左側：教室の空気分布
scatter = ax1.scatter(pos[:, 0], pos[:, 1], c=['green' for _ in range(AGENTS_NUMBER)], s=15, alpha=0.7, edgecolors='black', linewidths=0.2)
ax1.set_xlim(-2, WIDTH + 2)
ax1.set_ylim(-2, HEIGHT + 2)
ax1.set_aspect('equal')

# 壁のプロット
ax1.plot([0, 0], [0, LEFT_WINDOW[0]], color='black', linewidth=3)
ax1.plot([0, 0], [LEFT_WINDOW[1], HEIGHT], color='black', linewidth=3)
ax1.text(-5, window_center[1], 'Window', fontsize=10, color='green', weight='bold', va='center')

ax1.plot([WIDTH, WIDTH], [0, RIGHT_DOOR[0]], color='black', linewidth=3)
ax1.plot([WIDTH, WIDTH], [RIGHT_DOOR[1], HEIGHT], color='black', linewidth=3)
ax1.text(WIDTH+1, door_center[1], 'Door', fontsize=10, color='brown', weight='bold', va='center')

ax1.plot([0, WIDTH], [0, 0], color='black', linewidth=3)
ax1.plot([0, WIDTH], [HEIGHT, HEIGHT], color='black', linewidth=3)

# 右側：分子の個数・割合推移グラフ
ax2.set_xlim(0, SIMULATION_TIME)
ax2.set_ylim(0, AGENTS_NUMBER) 
ax2.set_xlabel('Time Steps')
ax2.set_ylabel('Number of Molecules')
ax2.set_title('Indoor Molecular Distribution (O2 vs CO2)')

line_o2, = ax2.plot([], [], color='blue', label='Oxygen (O2)', linewidth=2)
line_co2, = ax2.plot([], [], color='green', label='Carbon Dioxide (CO2)', linewidth=2)
ax2.legend(loc='upper right')
ax2.grid(True)

ax2_twin = ax2.twinx()
ax2_twin.set_ylim(0, 100)
ax2_twin.set_ylabel('Percentage (%)')

# アニメーション更新関数
def animate(frame):
    global pos, states
    
    # ★★★ 新しい風の通り道判定（点と直線の距離を利用） ★★★
    # 窓の中心から各エージェントへのベクトル
    v_agent = pos - window_center
    # 風の方向ベクトルに対する投影（中心線に沿った進捗度）
    proj_len = np.dot(v_agent, wind_dir)
    # 中心線からエージェントまでの最短距離（垂直距離）を計算
    closest_points = window_center + np.outer(proj_len, wind_dir)
    dist_from_route = np.linalg.norm(pos - closest_points, axis=1)
    
    # 風の通り道のエリア（中心線から指定幅以内）
    in_wind_route = dist_from_route <= WIND_ROUTE_WIDTH
    
    # 確率判定
    is_affected = np.random.rand(AGENTS_NUMBER) < WIND_PROBABILITY
    wind_mask = in_wind_route & is_affected
    
    # 基本の熱運動
    angles = np.random.rand(AGENTS_NUMBER) * 2 * np.pi
    dx = AGENT_SPEED * np.cos(angles)
    dy = AGENT_SPEED * np.sin(angles)
    
    # ★★★ 窓からドアへの方向ベクトルを風の速度に適用 ★★★
    dx[wind_mask] = WIND_SPEED * wind_dir[0]
    dy[wind_mask] = WIND_SPEED * wind_dir[1]
    
    pos[:, 0] += dx
    pos[:, 1] += dy
    
    # 壁の判定
    pos[:, 1] = np.clip(pos[:, 1], 0, HEIGHT)
    
    left_wall = pos[:, 0] < 0
    in_window = (pos[:, 1] >= LEFT_WINDOW[0]) & (pos[:, 1] <= LEFT_WINDOW[1])
    pos[left_wall & ~in_window, 0] = 0.0
    
    right_wall = pos[:, 0] >= WIDTH
    in_door = (pos[:, 1] >= RIGHT_DOOR[0]) & (pos[:, 1] <= RIGHT_DOOR[1])
    pos[right_wall & ~in_door, 0] = WIDTH
    
    # 換気処理（外からは新鮮な酸素が入ってくる）
    exited_right = right_wall & in_door
    exited_left = left_wall & in_window
    exited_mask = exited_right | exited_left
    exit_count = np.sum(exited_mask)
    
    if exit_count > 0:
        pos[exited_mask, 0] = 0.0
        pos[exited_mask, 1] = np.random.uniform(LEFT_WINDOW[0], LEFT_WINDOW[1], size=exit_count)
        states[exited_mask] = OXYGEN  
        
    # 各分子の個数カウント
    current_co2_count = np.sum(states == CO2)
    current_o2_count = AGENTS_NUMBER - current_co2_count
    
    log_time.append(frame)
    log_o2_count.append(current_o2_count)
    log_co2_count.append(current_co2_count)
    
    # 描画の更新
    scatter.set_offsets(pos)
    scatter.set_facecolors(['blue' if s==OXYGEN else 'green' for s in states])
    
    o2_pct = (current_o2_count / AGENTS_NUMBER) * 100
    co2_pct = (current_co2_count / AGENTS_NUMBER) * 100
    ax1.set_title(f"Dynamic Directional Ventilation (Frame: {frame})\nO2: {current_o2_count} ({o2_pct:.1f}%) | CO2: {current_co2_count} ({co2_pct:.1f}%)", fontsize=10)
    
    line_o2.set_data(log_time, log_o2_count)
    line_co2.set_data(log_time, log_co2_count)
    
    return scatter, line_o2, line_co2,

ani = animation.FuncAnimation(fig, animate, frames=SIMULATION_TIME, blit=True, repeat=False)
plt.tight_layout()
print("窓とドアの位置に追従する風向シミュレーションを表示します...")
plt.show()