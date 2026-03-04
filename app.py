from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/evaluate', methods=['POST'])
def evaluate():
    data = request.json
    n = data.get('n')
    start = data.get('start')
    end = data.get('end')
    obstacles = data.get('obstacles', [])
    policy = data.get('policy')

    # 基本校驗
    if not all([n, end, policy]):
        return jsonify({"error": "Missing required data"}), 400

    # 初始化 V(s) = 0
    V = [[0.0 for _ in range(n)] for _ in range(n)]
    gamma = 0.9
    theta = 1e-5

    obs_set = set(tuple(o) for o in obstacles)
    end_tuple = tuple(end)
    
    # Policy Evaluation (Bellman Equation)
    while True:
        delta = 0
        new_V = [[0.0 for _ in range(n)] for _ in range(n)]
        
        for r in range(n):
            for c in range(n):
                # 終點或障礙物的價值保持為 0
                if (r, c) == end_tuple or (r, c) in obs_set:
                    continue
                
                action = policy[r][c]
                nr, nc = r, c
                
                if action == 'U': nr -= 1
                elif action == 'D': nr += 1
                elif action == 'L': nc -= 1
                elif action == 'R': nc += 1
                
                reward = 0
                next_v = 0
                
                # 若撞牆或遇到障礙物
                if nr < 0 or nr >= n or nc < 0 or nc >= n or (nr, nc) in obs_set:
                    reward = -10
                    next_v = V[r][c]  # 退回原狀態
                # 若到達終點
                elif (nr, nc) == end_tuple:
                    reward = 100
                    next_v = 0  # 終點本身無後續狀態，價值為 0
                # 正常步進
                else:
                    reward = 0
                    next_v = V[nr][nc]
                
                # 計算新值
                v_new = reward + gamma * next_v
                new_V[r][c] = v_new
                
                delta = max(delta, abs(V[r][c] - v_new))
                
        V = new_V
        if delta < theta:
            break

    # 取小數點後兩位返回
    V_rounded = [[round(V[r][c], 2) for c in range(n)] for r in range(n)]
    
    return jsonify({"values": V_rounded})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
