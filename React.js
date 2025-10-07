import React, { useState, useCallback, useEffect, useMemo } from 'react';

// Configuration
const MCTS_ITERATIONS = 2000;
const EXPLORATION_CONSTANT = Math.sqrt(2);

// Player enum
const Player = {
  X: 'X',
  O: 'O'
};

// Outcome constants for precomputed minimax
const OUTCOME_ONGOING = 0;
const OUTCOME_X_WINS = 1;
const OUTCOME_O_WINS = 2;
const OUTCOME_DRAW = 3;

const getOpponent = (player) => player === Player.X ? Player.O : Player.X;

// Game State class
class GameState {
  constructor(board = Array(9).fill(null), currentPlayer = Player.X) {
    this.board = [...board];
    this.currentPlayer = currentPlayer;
  }

  display() {
    return this.board;
  }

  getLegalMoves() {
    return this.board
      .map((cell, index) => cell === null ? index : null)
      .filter(index => index !== null);
  }

  makeMove(action) {
    const newBoard = [...this.board];
    newBoard[action] = this.currentPlayer;
    return new GameState(newBoard, getOpponent(this.currentPlayer));
  }

  getWinner() {
    const lines = [
      [0, 1, 2], [3, 4, 5], [6, 7, 8], // Rows
      [0, 3, 6], [1, 4, 7], [2, 5, 8], // Columns
      [0, 4, 8], [2, 4, 6]             // Diagonals
    ];

    for (const line of lines) {
      const [a, b, c] = line;
      if (this.board[a] && this.board[a] === this.board[b] && this.board[a] === this.board[c]) {
        return this.board[a];
      }
    }
    return null;
  }

  isTerminal() {
    return this.getWinner() !== null || this.getLegalMoves().length === 0;
  }

  clone() {
    return new GameState([...this.board], this.currentPlayer);
  }
}

// Board encoding: convert board to base-3 index
function boardToIndex(board) {
  let index = 0;
  for (let i = 0; i < 9; i++) {
    const cell = board[i];
    const value = cell === null ? 0 : cell === Player.X ? 1 : 2;
    index += value * Math.pow(3, i);
  }
  return index;
}

// Precompute all game outcomes using minimax with dynamic programming
function precomputeAllGames() {
  const totalStates = Math.pow(3, 9); // 19,683 possible board states
  const outcomes = new Uint8Array(totalStates);
  
  // Memoization cache
  const memo = new Map();
  
  function minimax(state) {
    const index = boardToIndex(state.board);
    
    // Check memo
    if (memo.has(index)) {
      return memo.get(index);
    }
    
    // Check terminal states
    if (state.isTerminal()) {
      const winner = state.getWinner();
      let outcome;
      if (winner === Player.X) {
        outcome = OUTCOME_X_WINS;
      } else if (winner === Player.O) {
        outcome = OUTCOME_O_WINS;
      } else {
        outcome = OUTCOME_DRAW;
      }
      memo.set(index, outcome);
      outcomes[index] = outcome;
      return outcome;
    }
    
    const legalMoves = state.getLegalMoves();
    const isXTurn = state.currentPlayer === Player.X;
    
    let bestOutcome;
    if (isXTurn) {
      // X wants to maximize (looking for X win)
      bestOutcome = OUTCOME_O_WINS; // Worst case for X
      for (const move of legalMoves) {
        const newState = state.makeMove(move);
        const outcome = minimax(newState);
        
        if (outcome === OUTCOME_X_WINS) {
          bestOutcome = OUTCOME_X_WINS;
          break; // Best possible
        } else if (outcome === OUTCOME_DRAW && bestOutcome !== OUTCOME_X_WINS) {
          bestOutcome = OUTCOME_DRAW;
        }
      }
    } else {
      // O wants to minimize (looking for O win)
      bestOutcome = OUTCOME_X_WINS; // Worst case for O
      for (const move of legalMoves) {
        const newState = state.makeMove(move);
        const outcome = minimax(newState);
        
        if (outcome === OUTCOME_O_WINS) {
          bestOutcome = OUTCOME_O_WINS;
          break; // Best possible
        } else if (outcome === OUTCOME_DRAW && bestOutcome !== OUTCOME_O_WINS) {
          bestOutcome = OUTCOME_DRAW;
        }
      }
    }
    
    memo.set(index, bestOutcome);
    outcomes[index] = bestOutcome;
    return bestOutcome;
  }
  
  // Precompute from initial state
  const initialState = new GameState();
  minimax(initialState);
  
  return outcomes;
}

// MCTS Node class (kept for difficult mode)
class Node {
  constructor(state, parentIndex = null, moveFromParent = null) {
    this.state = state.clone();
    this.parentIndex = parentIndex;
    this.childrenIndices = [];
    this.visits = 0;
    this.wins = 0;
    this.untriedMoves = [...state.getLegalMoves()];
    this.moveFromParent = moveFromParent;
  }

  ucb1(parentVisits) {
    if (this.visits === 0) {
      return Infinity;
    }
    const winRate = this.wins / this.visits;
    const exploration = EXPLORATION_CONSTANT * Math.sqrt(Math.log(parentVisits) / this.visits);
    return winRate + exploration;
  }

  isFullyExpanded() {
    return this.untriedMoves.length === 0;
  }
}

// MCTS class (kept for difficult mode)
class MCTS {
  constructor(rootState) {
    const rootNode = new Node(rootState);
    this.tree = [rootNode];
  }

  run() {
    for (let i = 0; i < MCTS_ITERATIONS; i++) {
      const selectedIndex = this.select();
      const expandedIndex = this.expand(selectedIndex);
      const result = this.simulate(this.tree[expandedIndex].state);
      this.backpropagate(expandedIndex, result);
    }
  }

  select() {
    let currentIndex = 0;
    while (true) {
      if (this.tree[currentIndex].state.isTerminal()) {
        return currentIndex;
      }

      if (!this.tree[currentIndex].isFullyExpanded()) {
        return currentIndex;
      }

      const parentVisits = this.tree[currentIndex].visits;
      currentIndex = this.tree[currentIndex].childrenIndices.reduce((best, childIndex) => {
        const bestUcb1 = this.tree[best].ucb1(parentVisits);
        const childUcb1 = this.tree[childIndex].ucb1(parentVisits);
        return childUcb1 > bestUcb1 ? childIndex : best;
      });
    }
  }

  expand(nodeIndex) {
    if (this.tree[nodeIndex].state.isTerminal()) {
      return nodeIndex;
    }

    const action = this.tree[nodeIndex].untriedMoves.pop();
    if (action === undefined) {
      throw new Error("Trying to expand but no untried moves available");
    }

    const newState = this.tree[nodeIndex].state.makeMove(action);
    const newNode = new Node(newState, nodeIndex, action);
    const newNodeIndex = this.tree.length;
    this.tree.push(newNode);

    this.tree[nodeIndex].childrenIndices.push(newNodeIndex);
    return newNodeIndex;
  }

  simulate(startState) {
    let currentState = startState.clone();
    while (!currentState.isTerminal()) {
      const legalMoves = currentState.getLegalMoves();
      const action = legalMoves[Math.floor(Math.random() * legalMoves.length)];
      currentState = currentState.makeMove(action);
    }

    return currentState.getWinner() ? 1.0 : 0.5;
  }

  backpropagate(nodeIndex, result) {
    let currentIndex = nodeIndex;
    let currentResult = result;

    while (currentIndex !== undefined) {
      this.tree[currentIndex].visits++;
      this.tree[currentIndex].wins += currentResult;

      const parentIndex = this.tree[currentIndex].parentIndex;
      if (parentIndex !== null) {
        currentIndex = parentIndex;
        currentResult = 1.0 - currentResult;
      } else {
        break;
      }
    }
  }

  bestMove() {
    const root = this.tree[0];
    if (root.childrenIndices.length === 0) {
      return null;
    }

    const bestChildIndex = root.childrenIndices.reduce((best, childIndex) => {
      return this.tree[childIndex].visits > this.tree[best].visits ? childIndex : best;
    });

    return this.tree[bestChildIndex].moveFromParent;
  }
}

// React Component
export default function TicTacToeGame() {
  const [gameState, setGameState] = useState(() => new GameState());
  const [isAiThinking, setIsAiThinking] = useState(false);
  const [gameStatus, setGameStatus] = useState('');
  const [aiThinkingTime, setAiThinkingTime] = useState(0);
  const [gameMode, setGameMode] = useState(null);
  const [isPrecomputing, setIsPrecomputing] = useState(true);
  const [precomputeTime, setPrecomputeTime] = useState(0);

  const humanPlayer = Player.X;
  const aiPlayer = Player.O;

  // Precompute all games on mount
  const precomputedOutcomes = useMemo(() => {
    const startTime = performance.now();
    const outcomes = precomputeAllGames();
    const endTime = performance.now();
    setPrecomputeTime(endTime - startTime);
    setIsPrecomputing(false);
    return outcomes;
  }, []);

  const updateGameStatus = useCallback((state) => {
    const winner = state.getWinner();
    if (winner === humanPlayer) {
      setGameStatus(`You win! 🎉`);
    } else if (winner === aiPlayer) {
      setGameStatus(`AI wins! 🤖`);
    } else if (state.isTerminal()) {
      setGameStatus(`It's a draw! 🤝`);
    } else if (state.currentPlayer === humanPlayer) {
      setGameStatus('Your turn');
    } else {
      setGameStatus('AI is thinking...');
    }
  }, [humanPlayer, aiPlayer]);

  const makeAiMove = useCallback(async (currentState) => {
    setIsAiThinking(true);
    const startTime = performance.now();
    
    setTimeout(() => {
      let action;
      
      if (gameMode === 'easy') {
        // Easy mode: random move
        const legalMoves = currentState.getLegalMoves();
        action = legalMoves[Math.floor(Math.random() * legalMoves.length)];
      } else if (gameMode === 'precomputed') {
        // Precomputed mode: use lookup table
        const legalMoves = currentState.getLegalMoves();
        const isOTurn = currentState.currentPlayer === Player.O;
        
        let bestMove = legalMoves[0];
        let bestOutcome = isOTurn ? OUTCOME_X_WINS : OUTCOME_O_WINS; // Worst case
        
        for (const move of legalMoves) {
          const newState = currentState.makeMove(move);
          const index = boardToIndex(newState.board);
          const outcome = precomputedOutcomes[index];
          
          if (isOTurn) {
            // O wants OUTCOME_O_WINS
            if (outcome === OUTCOME_O_WINS) {
              bestMove = move;
              bestOutcome = OUTCOME_O_WINS;
              break;
            } else if (outcome === OUTCOME_DRAW && bestOutcome !== OUTCOME_O_WINS) {
              bestMove = move;
              bestOutcome = OUTCOME_DRAW;
            }
          } else {
            // X wants OUTCOME_X_WINS
            if (outcome === OUTCOME_X_WINS) {
              bestMove = move;
              bestOutcome = OUTCOME_X_WINS;
              break;
            } else if (outcome === OUTCOME_DRAW && bestOutcome !== OUTCOME_X_WINS) {
              bestMove = move;
              bestOutcome = OUTCOME_DRAW;
            }
          }
        }
        
        action = bestMove;
      } else {
        // Difficult mode: use MCTS
        const mcts = new MCTS(currentState);
        mcts.run();
        action = mcts.bestMove();
      }
      
      const endTime = performance.now();
      setAiThinkingTime(endTime - startTime);
      
      if (action !== null && action !== undefined) {
        const newState = currentState.makeMove(action);
        setGameState(newState);
      }
      setIsAiThinking(false);
    }, 50);
  }, [gameMode, precomputedOutcomes]);

  const handleCellClick = useCallback((index) => {
    if (gameState.board[index] !== null || gameState.isTerminal() || isAiThinking || gameState.currentPlayer !== humanPlayer) {
      return;
    }

    const newState = gameState.makeMove(index);
    setGameState(newState);
  }, [gameState, isAiThinking, humanPlayer]);

  const resetGame = useCallback(() => {
    setGameState(new GameState());
    setIsAiThinking(false);
    setAiThinkingTime(0);
    setGameMode(null);
  }, []);

  useEffect(() => {
    updateGameStatus(gameState);
    
    if (!gameState.isTerminal() && gameState.currentPlayer === aiPlayer && !isAiThinking && gameMode !== null) {
      makeAiMove(gameState);
    }
  }, [gameState, aiPlayer, isAiThinking, makeAiMove, updateGameStatus, gameMode]);

  const renderCell = (index) => {
    const value = gameState.board[index];
    const isClickable = !value && !gameState.isTerminal() && !isAiThinking && gameState.currentPlayer === humanPlayer;
    
    return (
      <button
        key={index}
        className={`
          w-20 h-20 border-2 border-gray-400 text-2xl font-bold rounded-lg
          transition-all duration-200 transform
          ${isClickable 
            ? 'hover:bg-blue-50 hover:scale-105 cursor-pointer active:scale-95 bg-gray-50' 
            : 'cursor-not-allowed'
          }
          ${value === Player.X ? 'text-blue-600 bg-white' : value === Player.O ? 'text-red-600 bg-white' : 'bg-gray-100 opacity-60'}
          ${!value && isClickable ? 'hover:shadow-md text-gray-400' : ''}
        `}
        onClick={() => handleCellClick(index)}
        disabled={!isClickable}
      >
        {value || (isClickable ? index + 1 : '')}
      </button>
    );
  };

  if (isPrecomputing) {
    return (
      <div className="flex flex-col items-center justify-center min-h-screen bg-gray-50 p-4">
        <div className="bg-white rounded-lg shadow-lg p-8 max-w-md w-full text-center">
          <div className="animate-spin w-12 h-12 border-4 border-blue-500 border-t-transparent rounded-full mx-auto mb-4"></div>
          <h2 className="text-xl font-bold mb-2">Precomputing Game States...</h2>
          <p className="text-gray-600">Using dynamic programming to compute optimal moves</p>
          <p className="text-sm text-gray-500 mt-2">Computing 19,683 board states with minimax</p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gray-50 p-4">
      <div className="bg-white rounded-lg shadow-lg p-8 max-w-md w-full">
        <h1 className="text-3xl font-bold text-center mb-2">Tic-Tac-Toe</h1>
        <p className="text-center text-gray-600 mb-2">Human (X) vs AI (O)</p>
        <p className="text-xs text-gray-500 text-center mb-4">
          Precomputed in {precomputeTime.toFixed(2)}ms using minimax DP
        </p>
        
        {gameMode === null ? (
          <div className="flex flex-col gap-3">
            <h2 className="text-xl font-semibold text-center mb-2">Choose Difficulty</h2>
            <button
              onClick={() => setGameMode('easy')}
              className="w-full bg-green-500 hover:bg-green-600 text-white font-bold py-4 px-6 rounded-lg transition-all duration-200 transform hover:scale-105"
            >
              <div className="text-2xl mb-1">😊 Easy Mode</div>
              <div className="text-sm opacity-90">Random AI moves</div>
            </button>
            <button
              onClick={() => setGameMode('precomputed')}
              className="w-full bg-purple-500 hover:bg-purple-600 text-white font-bold py-4 px-6 rounded-lg transition-all duration-200 transform hover:scale-105"
            >
              <div className="text-2xl mb-1">⚡ Perfect Mode</div>
              <div className="text-sm opacity-90">Precomputed minimax (instant moves)</div>
            </button>
            <button
              onClick={() => setGameMode('difficult')}
              className="w-full bg-red-500 hover:bg-red-600 text-white font-bold py-4 px-6 rounded-lg transition-all duration-200 transform hover:scale-105"
            >
              <div className="text-2xl mb-1">🤖 MCTS Mode</div>
              <div className="text-sm opacity-90">Monte Carlo Tree Search ({MCTS_ITERATIONS} simulations)</div>
            </button>
          </div>
        ) : (
          <>
            <div className="text-center mb-4">
              <span className={`inline-block px-3 py-1 rounded-full text-sm font-semibold ${
                gameMode === 'easy' ? 'bg-green-100 text-green-800' : 
                gameMode === 'precomputed' ? 'bg-purple-100 text-purple-800' :
                'bg-red-100 text-red-800'
              }`}>
                {gameMode === 'easy' ? '😊 Easy Mode' : 
                 gameMode === 'precomputed' ? '⚡ Perfect Mode' :
                 '🤖 MCTS Mode'}
              </span>
            </div>
            
            <div className="grid grid-cols-3 gap-2 mb-6">
              {Array.from({ length: 9 }, (_, index) => renderCell(index))}
            </div>

            <div className="text-center mb-4">
              <p className={`text-lg font-semibold ${isAiThinking ? 'text-orange-600' : 'text-gray-800'}`}>
                {gameStatus}
              </p>
              {isAiThinking && (
                <div className="mt-2">
                  <div className="animate-spin w-6 h-6 border-2 border-orange-500 border-t-transparent rounded-full mx-auto"></div>
                </div>
              )}
              {aiThinkingTime > 0 && !isAiThinking && (
                <p className="text-sm text-gray-500 mt-1">
                  {gameMode === 'precomputed' ? 
                    `Lookup time: ${aiThinkingTime.toFixed(2)}ms` :
                    gameMode === 'difficult' ?
                    `AI computed in ${aiThinkingTime.toFixed(2)}ms (${MCTS_ITERATIONS} sims)` :
                    `Move time: ${aiThinkingTime.toFixed(2)}ms`
                  }
                </p>
              )}
            </div>

            <button
              onClick={resetGame}
              className="w-full bg-blue-600 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded-lg transition-colors duration-200"
            >
              New Game
            </button>
          </>
        )}
      </div>
    </div>
  );
}
