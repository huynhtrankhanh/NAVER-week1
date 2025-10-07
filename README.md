# NAVER-week1

## Tic-Tac-Toe AI Game

A sophisticated Tic-Tac-Toe game built with React, TypeScript, and Vite featuring multiple AI difficulty levels.

### Features

- **Three AI Difficulty Levels:**
  - **Easy Mode**: AI makes random moves
  - **Perfect Mode**: Unbeatable AI using precomputed minimax with dynamic programming
  - **MCTS Mode**: Advanced AI using Monte Carlo Tree Search (2000 simulations)

- **Modern Tech Stack:**
  - React 18
  - TypeScript
  - Vite
  - Tailwind CSS

### Getting Started

1. Install dependencies:
   ```bash
   npm install
   ```

2. Run development server:
   ```bash
   npm run dev
   ```

3. Build for production:
   ```bash
   npm run build
   ```

4. Preview production build:
   ```bash
   npm run preview
   ```

### How to Play

1. Choose your preferred AI difficulty level
2. You play as X, the AI plays as O
3. Click on the numbered cells to make your move
4. Try to get three in a row horizontally, vertically, or diagonally!

### Technical Details

The game uses sophisticated AI algorithms:
- **Minimax with Dynamic Programming**: Precomputes all 19,683 possible board states for instant, perfect moves
- **Monte Carlo Tree Search (MCTS)**: Uses UCB1 exploration strategy with 2000 simulations per move