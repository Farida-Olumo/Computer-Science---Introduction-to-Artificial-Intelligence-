import time
import statistics
import random

class SudokuSolverFC:
    def __init__(self, board):
        self.board = [row[:] for row in board]
        self.nodes_expanded = 0
        self.domains = {(r, c): [self.board[r][c]] if self.board[r][c] != 0 else list(range(1, 10))
                        for r in range(9) for c in range(9)}

    def print_board(self):
        for i in range(9):
            if i % 3 == 0 and i != 0:
                print("- - - - - - - - - - -")
            print(" ".join(str(self.board[i][j]) for j in range(9)))

    def get_neighbors(self, r, c):
        neighbors = set()
        for i in range(9):
            neighbors.add((r, i))
            neighbors.add((i, c))
        sr, sc = 3 * (r // 3), 3 * (c // 3)
        for i in range(sr, sr + 3):
            for j in range(sc, sc + 3):
                neighbors.add((i, j))
        neighbors.discard((r, c))
        return neighbors

    def forward_check(self, r, c, val, current_domains):
        new_domains = {k: list(v) for k, v in current_domains.items()}
        for nr, nc in self.get_neighbors(r, c):
            if val in new_domains[(nr, nc)]:
                new_domains[(nr, nc)].remove(val)
                if not new_domains[(nr, nc)]:
                    return None
        return new_domains

    def solve(self, current_domains):
        self.nodes_expanded += 1

        var = next(((r, c) for r in range(9) for c in range(9)
                    if self.board[r][c] == 0), None)
        if not var:
            return True

        r, c = var

        # 🔥 RANDOMIZE DOMAIN ORDER HERE
        values = list(current_domains[(r, c)])
        random.shuffle(values)

        for val in values:
            self.board[r][c] = val
            next_doms = self.forward_check(r, c, val, current_domains)
            if next_doms is not None and self.solve(next_doms):
                return True
            self.board[r][c] = 0

        return False

# ALL FOUR PUZZLES
EASY = [[0,6,1,0,0,0,0,5,2],[8,0,0,0,0,0,0,0,1],[7,0,0,5,0,0,4,0,0],[9,0,3,6,0,2,0,4,7],[0,0,6,7,0,1,5,0,0],[5,7,0,9,0,3,2,0,6],[0,0,4,0,0,9,0,0,5],[1,0,0,0,0,0,0,0,8],[6,2,0,0,0,0,9,3,0]]
MEDIUM = [[8,3,0,6,0,0,0,0,7],[0,0,7,0,2,0,0,5,0],[0,2,1,0,0,9,0,8,0],[6,0,0,0,8,0,0,0,9],[0,0,0,4,6,5,0,0,0],[3,0,0,0,9,0,0,0,2],[0,8,0,2,0,0,3,9,0],[0,5,0,0,4,0,2,0,0],[2,0,0,0,0,8,0,1,6]]
HARD = [[0,4,0,0,2,5,9,0,0],[0,0,0,0,3,9,0,4,0],[0,0,0,0,0,0,0,6,1],[0,1,7,0,0,0,0,0,0],[6,0,0,7,5,4,0,0,9],[0,0,0,0,0,0,7,3,0],[4,2,0,0,0,0,0,0,0],[0,9,0,5,4,0,0,0,0],[0,0,8,9,6,0,0,5,0]]
EVIL = [[0,6,0,8,2,0,0,0,0],[0,0,2,0,0,0,8,0,1],[0,0,0,7,0,0,0,5,0],[4,0,0,5,0,0,0,0,6],[0,9,0,6,0,7,0,3,0],[2,0,0,0,0,1,0,0,7],[0,2,0,0,0,9,0,0,0],[8,0,4,0,0,0,7,0,0],[0,0,0,0,4,8,0,2,0]]

if __name__ == "__main__":
    for name, grid in {"Easy": EASY, "Medium": MEDIUM, "Hard": HARD, "Evil": EVIL}.items():
        times, nodes = [], []
        print(f"\n{'='*8} Backtracking and Forward checking | Puzzle: {name} {'='*8}")
        for i in range(1, 31):
            s = SudokuSolverFC(grid)
            t0 = time.perf_counter()
            s.solve(s.domains)
            elapsed = time.perf_counter() - t0
            times.append(elapsed)
            nodes.append(s.nodes_expanded)
            print(f"Run {i:02d}: Time {elapsed:.5f}s | Nodes {s.nodes_expanded}")
            if i == 1:
                print("\nSolution")
                s.print_board()
                print("-" * 30)

        print(f"\nFINAL STATS for {name}:")
        print(f"Avg Time: {statistics.mean(times):.6f}s (StdDev: {statistics.stdev(times):.6f}s)")
        print(f"Avg Nodes: {statistics.mean(nodes):.2f} (StdDev: {statistics.stdev(nodes):.2f})")
