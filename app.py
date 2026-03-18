import numpy as np
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

ACTIONS = ['U', 'D', 'L', 'R']
ACTION_DELTAS = {'U': (-1, 0), 'D': (1, 0), 'L': (0, -1), 'R': (0, 1)}

def get_next_state(r, c, a, n, obstacles_set):
    dr, dc = ACTION_DELTAS[a]
    nr, nc = r + dr, c + dc
    if 0 <= nr < n and 0 <= nc < n and (nr, nc) not in obstacles_set:
        return (nr, nc)
    return (r, c)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/evaluate', methods=['POST'])
def evaluate():
    data = request.json
    n = data.get('n', 5)
    start = tuple(data.get('start', [0, 0]))
    end = tuple(data.get('end', [n-1, n-1]))
    obstacles = data.get('obstacles', [])
    policy = data.get('policy', [])
    
    obstacles_set = set(tuple(obs) for obs in obstacles)
    
    gamma = 0.9
    theta = 1e-4
    
    V = np.zeros((n, n))
    
    # Policy Evaluation
    while True:
        delta = 0
        V_new = np.copy(V)
        for r in range(n):
            for c in range(n):
                s = (r, c)
                if s == end or s in obstacles_set:
                    continue
                
                a = policy[r][c]
                next_s = get_next_state(r, c, a, n, obstacles_set)
                
                reward = 1.0 if next_s == end else 0.0
                v_next = V[next_s] if next_s != end else 0.0
                v_curr = reward + gamma * v_next
                
                delta = max(delta, abs(v_curr - V[r, c]))
                V_new[r, c] = v_curr
                
        V = V_new
        if delta < theta:
            break
            
    return jsonify({'V': V.tolist()})

@app.route('/api/optimize', methods=['POST'])
def optimize():
    data = request.json
    n = data.get('n', 5)
    start = tuple(data.get('start', [0, 0]))
    end = tuple(data.get('end', [n-1, n-1]))
    obstacles = data.get('obstacles', [])
    
    obstacles_set = set(tuple(obs) for obs in obstacles)
    
    gamma = 0.9
    theta = 1e-4
    
    V = np.zeros((n, n))
    policy = [['U' for _ in range(n)] for _ in range(n)]
    
    # Value Iteration
    while True:
        delta = 0
        V_new = np.copy(V)
        for r in range(n):
            for c in range(n):
                s = (r, c)
                if s == end or s in obstacles_set:
                    continue
                
                max_v = -float('inf')
                best_a = 'U'
                
                for a in ACTIONS:
                    next_s = get_next_state(r, c, a, n, obstacles_set)
                    reward = 1.0 if next_s == end else 0.0
                    v_next = V[next_s] if next_s != end else 0.0
                    val = reward + gamma * v_next
                    
                    if val > max_v:
                        max_v = val
                        best_a = a
                        
                delta = max(delta, abs(max_v - V[r, c]))
                V_new[r, c] = max_v
                policy[r][c] = best_a
                
        V = V_new
        if delta < theta:
            break
            
    return jsonify({'V': V.tolist(), 'policy': policy})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
