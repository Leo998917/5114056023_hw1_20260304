const ARROWS = {
    'U': '↑',
    'D': '↓',
    'L': '←',
    'R': '→'
};

const RANDOM_ACTIONS = ['U', 'D', 'L', 'R'];

let n = 5;
let state = 'SET_START'; // SET_START -> SET_END -> SET_OBSTACLE -> DONE
let startCell = null;
let endCell = null;
let obstacles = [];

const configGrid = document.getElementById('config-grid');
const gridSizeInput = document.getElementById('grid-size');
const statusText = document.getElementById('status-text');
const btnReset = document.getElementById('btn-reset');
const btnHw12 = document.getElementById('btn-hw1-2');
const btnHw13 = document.getElementById('btn-hw1-3');

const gridHw12 = document.getElementById('grid-hw1-2');
const sectionHw12 = document.getElementById('section-hw1-2');

const gridHw13 = document.getElementById('grid-hw1-3');
const sectionHw13 = document.getElementById('section-hw1-3');

function init() {
    n = parseInt(gridSizeInput.value) || 5;
    if (n < 5) n = 5;
    if (n > 9) n = 9;
    gridSizeInput.value = n;
    
    state = 'SET_START';
    startCell = null;
    endCell = null;
    obstacles = [];
    
    updateStatus();
    renderGrid(configGrid, n, true);
    
    btnHw12.disabled = true;
    btnHw13.disabled = true;
    sectionHw12.style.display = 'none';
    sectionHw13.style.display = 'none';
}

function updateStatus() {
    if (state === 'SET_START') {
        statusText.innerText = "State: Click a cell to set Start (Green).";
    } else if (state === 'SET_END') {
        statusText.innerText = "State: Click a cell to set End (Red).";
    } else if (state === 'SET_OBSTACLE') {
        let maxObs = n - 2;
        statusText.innerText = `State: Click cells to set Obstacles (${obstacles.length} / ${maxObs}) (Gray).`;
    } else {
        statusText.innerText = "State: Ready. You can solve HW1-2 and HW1-3 now.";
    }
}

function handleCellClick(r, c) {
    if (state === 'SET_START') {
        startCell = [r, c];
        state = 'SET_END';
    } else if (state === 'SET_END') {
        if (r === startCell[0] && c === startCell[1]) return;
        endCell = [r, c];
        state = 'SET_OBSTACLE';
    } else if (state === 'SET_OBSTACLE') {
        if (r === startCell[0] && c === startCell[1]) return;
        if (r === endCell[0] && c === endCell[1]) return;
        
        let existingIdx = obstacles.findIndex(obs => obs[0] === r && obs[1] === c);
        if (existingIdx >= 0) {
            obstacles.splice(existingIdx, 1);
        } else {
            if (obstacles.length < n - 2) {
                obstacles.push([r, c]);
            }
        }
        
        if (obstacles.length === n - 2) {
            state = 'DONE';
            btnHw12.disabled = false;
            btnHw13.disabled = false;
        }
    }
    
    updateStatus();
    renderGrid(configGrid, n, true);
}

function createDOMCell(r, c, isInteractive) {
    const cell = document.createElement('div');
    cell.classList.add('cell');
    
    if (startCell && startCell[0] === r && startCell[1] === c) {
        cell.classList.add('start');
        cell.innerText = 'S';
    } else if (endCell && endCell[0] === r && endCell[1] === c) {
        cell.classList.add('end');
        cell.innerText = 'E';
    } else if (obstacles.some(obs => obs[0] === r && obs[1] === c)) {
        cell.classList.add('obstacle');
    }
    
    if (isInteractive && state !== 'DONE') {
        cell.addEventListener('click', () => handleCellClick(r, c));
    }
    
    return cell;
}

function renderGrid(container, n, isInteractive) {
    container.innerHTML = '';
    container.style.gridTemplateColumns = `repeat(${n}, 60px)`;
    container.style.gridTemplateRows = `repeat(${n}, 60px)`;
    
    for (let r = 0; r < n; r++) {
        for (let c = 0; c < n; c++) {
            container.appendChild(createDOMCell(r, c, isInteractive));
        }
    }
}

function renderResultGrid(container, n, policy, V) {
    container.innerHTML = '';
    container.style.gridTemplateColumns = `repeat(${n}, 60px)`;
    container.style.gridTemplateRows = `repeat(${n}, 60px)`;
    
    for (let r = 0; r < n; r++) {
        for (let c = 0; c < n; c++) {
            let cell = createDOMCell(r, c, false);
            
            // Do not show arrows on obstacles or end cell
            let isEnd = endCell && endCell[0] === r && endCell[1] === c;
            let isObstacle = obstacles.some(obs => obs[0] === r && obs[1] === c);
            
            if (!isObstacle) {
                if (!isEnd) {
                    let arrowDiv = document.createElement('div');
                    arrowDiv.classList.add('arrow');
                    arrowDiv.innerText = ARROWS[policy[r][c]] || '';
                    cell.prepend(arrowDiv);
                }
                
                let valDiv = document.createElement('div');
                valDiv.classList.add('value');
                valDiv.innerText = V[r][c].toFixed(2);
                cell.appendChild(valDiv);
            }
            
            container.appendChild(cell);
        }
    }
}

btnReset.addEventListener('click', init);
gridSizeInput.addEventListener('change', init);

btnHw12.addEventListener('click', async () => {
    btnHw12.disabled = true;
    let randomPolicy = [];
    for (let r = 0; r < n; r++) {
        let row = [];
        for (let c = 0; c < n; c++) {
            row.push(RANDOM_ACTIONS[Math.floor(Math.random() * RANDOM_ACTIONS.length)]);
        }
        randomPolicy.push(row);
    }
    
    try {
        const response = await fetch('/api/evaluate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                n, start: startCell, end: endCell, obstacles, policy: randomPolicy
            })
        });
        const data = await response.json();
        
        sectionHw12.style.display = 'block';
        renderResultGrid(gridHw12, n, randomPolicy, data.V);
    } catch (e) {
        alert("Error connecting to backend");
    } finally {
        btnHw12.disabled = false;
    }
});

btnHw13.addEventListener('click', async () => {
    btnHw13.disabled = true;
    try {
        const response = await fetch('/api/optimize', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                n, start: startCell, end: endCell, obstacles
            })
        });
        const data = await response.json();
        
        sectionHw13.style.display = 'block';
        renderResultGrid(gridHw13, n, data.policy, data.V);
    } catch (e) {
        alert("Error connecting to backend");
    } finally {
        btnHw13.disabled = false;
    }
});

// Initialize on page load
init();
