import time
import statistics
import random

class SudokuSolverB:
    def __init__(self, board):
        self.board = [row[:] for row in board]
        self.nodes_expanded = 0

    def print_board(self):
        for i in range(9):
            if i % 3 == 0 and i != 0:
                print("- - - - - - - - - - -")
            print(" ".join(str(self.board[i][j]) if self.board[i][j] != 0 else "." for j in range(9)))

    def is_consistent(self, r, c, val):
        for i in range(9):
            if self.board[r][i] == val or self.board[i][c] == val:
                return False
        sr, sc = 3 * (r // 3), 3 * (c // 3)
        for i in range(sr, sr + 3):
            for j in range(sc, sc + 3):
                if self.board[i][j] == val:
                    return False
        return True

    def solve(self):
        self.nodes_expanded += 1
        for r in range(9):
            for c in range(9):
                if self.board[r][c] == 0:

                    # 🔥 RANDOMIZE VALUE ORDER HERE
                    vals = list(range(1, 10))
                    random.shuffle(vals)

                    for val in vals:
                        if self.is_consistent(r, c, val):
                            self.board[r][c] = val
                            if self.solve():
                                return True
                            self.board[r][c] = 0
                    return False
        return True

# ALL FOUR PUZZLES 
EASY = [[0,6,1,0,0,0,0,5,2],[8,0,0,0,0,0,0,0,1],[7,0,0,5,0,0,4,0,0],[9,0,3,6,0,2,0,4,7],[0,0,6,7,0,1,5,0,0],[5,7,0,9,0,3,2,0,6],[0,0,4,0,0,9,0,0,5],[1,0,0,0,0,0,0,0,8],[6,2,0,0,0,0,9,3,0]]
MEDIUM = [[8,3,0,6,0,0,0,0,7],[0,0,7,0,2,0,0,5,0],[0,2,1,0,0,9,0,8,0],[6,0,0,0,8,0,0,0,9],[0,0,0,4,6,5,0,0,0],[3,0,0,0,9,0,0,0,2],[0,8,0,2,0,0,3,9,0],[0,5,0,0,4,0,2,0,0],[2,0,0,0,0,8,0,1,6]]
HARD = [[0,4,0,0,2,5,9,0,0],[0,0,0,0,3,9,0,4,0],[0,0,0,0,0,0,0,6,1],[0,1,7,0,0,0,0,0,0],[6,0,0,7,5,4,0,0,9],[0,0,0,0,0,0,7,3,0],[4,2,0,0,0,0,0,0,0],[0,9,0,5,4,0,0,0,0],[0,0,8,9,6,0,0,5,0]]
EVIL = [[0,6,0,8,2,0,0,0,0],[0,0,2,0,0,0,8,0,1],[0,0,0,7,0,0,0,5,0],[4,0,0,5,0,0,0,0,6],[0,9,0,6,0,7,0,3,0],[2,0,0,0,0,1,0,0,7],[0,2,0,0,0,9,0,0,0],[8,0,4,0,0,0,7,0,0],[0,0,0,0,4,8,0,2,0]]

if __name__ == "__main__":
    for name, grid in {"Easy": EASY, "Medium": MEDIUM, "Hard": HARD, "Evil": EVIL}.items():
        times, nodes = [], []
        print(f"\n{'='*8} Backtracking Search| Puzzle: {name} {'='*8}")
        for i in range(1, 31):
            s = SudokuSolverB(grid)
            t0 = time.perf_counter()
            s.solve()
            elapsed = time.perf_counter() - t0
            times.append(elapsed)
            nodes.append(s.nodes_expanded)
            print(f"Run {i:02d}: Time {elapsed:.5f}s | Nodes {s.nodes_expanded}")
            if i == 1:
                print("\nSolution:")
                s.print_board()
                print("-" * 30)
        
        print(f"\nTABLE STATISTICS for {name}:")
        print(f"Avg Time: {statistics.mean(times):.6f}s (StdDev: {statistics.stdev(times):.6f}s)")
        print(f"Avg Nodes: {statistics.mean(nodes):.2f} (StdDev: {statistics.stdev(nodes):.2f})")
