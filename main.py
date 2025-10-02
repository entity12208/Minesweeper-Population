# minesweeper.py
# Single pygame window showing a population of Minesweeper AIs learning together.
# Each thumbnail displays the agent's current rank and points (score).
#
# Save and run: python minesweeper_viz_population_rank.py
# Requirements: pip install pygame

import pygame
import threading
import time
import random
import math
from dataclasses import dataclass, field
from typing import List, Tuple
from collections import defaultdict
import queue

# ---------- CONFIG ----------
POPULATION_SIZE = 5     # changeable; set to 1000 if you really want that but expect heavy load
ROWS = 15
COLS = 15
MINES = 20

WINDOW_W = 1400
WINDOW_H = 900

THUMB_MARGIN = 4
LEADERBOARD_WIDTH = 360

EVOLUTION_INTERVAL_SEC = 10.0
ELITE_COUNT = max(2, POPULATION_SIZE // 50)
MUTATION_RATE = 0.15

STATS_PRINT_INTERVAL = 5.0

# Map speed 1..10 to delay seconds: 1 => near 0, 10 => ~2.0 sec
def speed_to_delay(speed_val: int) -> float:
    s = max(1, min(10, int(speed_val)))
    return ((s - 1) / 9.0) * 2.0

GLOBAL_SLOWDOWN = 1.0
DEFAULT_ENUM_LIMIT = 20
# ---------- END CONFIG ----------

# Directions
DIRS = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]

# ---------- Minesweeper headless model ----------
class MineModel:
    def __init__(self, rows:int, cols:int, mines:int, safe_first_click:bool = True):
        self.rows = rows; self.cols = cols; self.mines = mines
        self.safe_first_click = safe_first_click
        self.reset()

    def reset(self):
        self.is_mine = [[False]*self.cols for _ in range(self.rows)]
        self.nbr = [[0]*self.cols for _ in range(self.rows)]
        self.revealed = [[False]*self.cols for _ in range(self.rows)]
        self.flagged = [[False]*self.cols for _ in range(self.rows)]
        self.placed = False
        self.game_over = False
        self.win = False
        self.revealed_count = 0

    def place_mines(self, avoid_r:int, avoid_c:int):
        cells = [(r,c) for r in range(self.rows) for c in range(self.cols)]
        if self.safe_first_click:
            banned = set()
            banned.add((avoid_r, avoid_c))
            for dr,dc in DIRS:
                rr,cc = avoid_r+dr, avoid_c+dc
                if 0 <= rr < self.rows and 0 <= cc < self.cols:
                    banned.add((rr,cc))
            cells = [cell for cell in cells if cell not in banned]
        random.shuffle(cells)
        chosen = cells[:self.mines]
        for r,c in chosen:
            self.is_mine[r][c] = True
        for r in range(self.rows):
            for c in range(self.cols):
                if self.is_mine[r][c]:
                    self.nbr[r][c] = -1
                else:
                    cnt = 0
                    for dr,dc in DIRS:
                        rr,cc = r+dr, c+dc
                        if 0 <= rr < self.rows and 0 <= cc < self.cols and self.is_mine[rr][cc]:
                            cnt += 1
                    self.nbr[r][c] = cnt
        self.placed = True

    def reveal(self, r:int, c:int):
        if self.game_over or self.flagged[r][c]:
            return
        if not self.placed:
            self.place_mines(r,c)
        if self.is_mine[r][c]:
            self.revealed[r][c] = True
            self.game_over = True
            self.win = False
            return
        # flood fill
        stack = [(r,c)]
        while stack:
            rr,cc = stack.pop()
            if self.revealed[rr][cc]:
                continue
            self.revealed[rr][cc] = True
            self.revealed_count += 1
            if self.nbr[rr][cc] == 0:
                for dr,dc in DIRS:
                    nr,nc = rr+dr, cc+dc
                    if 0 <= nr < self.rows and 0 <= nc < self.cols and not self.revealed[nr][nc] and not self.flagged[nr][nc]:
                        if not self.is_mine[nr][nc]:
                            stack.append((nr,nc))
        self.check_win()

    def toggle_flag(self, r:int, c:int):
        if self.game_over or self.revealed[r][c]:
            return
        self.flagged[r][c] = not self.flagged[r][c]

    def check_win(self):
        total = self.rows * self.cols
        if self.revealed_count == total - self.mines:
            self.game_over = True
            self.win = True

# ---------- Perceived board builder (what agent sees) ----------
def build_perceived(model: MineModel):
    # -1 covered, -2 flagged, 0..8 revealed
    p = [[-1]*model.cols for _ in range(model.rows)]
    for r in range(model.rows):
        for c in range(model.cols):
            if model.flagged[r][c]:
                p[r][c] = -2
            elif model.revealed[r][c]:
                p[r][c] = model.nbr[r][c]
            else:
                p[r][c] = -1
    return p

# ---------- Solver (only sees perceived board) ----------
def in_bounds(r,c,rows,cols): return 0 <= r < rows and 0 <= c < cols

def apply_simple(perceived, rows, cols):
    to_click = set(); to_flag = set()
    for r in range(rows):
        for c in range(cols):
            v = perceived[r][c]
            if v >= 0:
                flagged = 0; covered = []
                for dr,dc in DIRS:
                    rr,cc = r+dr, c+dc
                    if not in_bounds(rr,cc,rows,cols): continue
                    pv = perceived[rr][cc]
                    if pv == -2: flagged += 1
                    elif pv == -1: covered.append((rr,cc))
                if not covered: continue
                if v - flagged == len(covered):
                    for cell in covered: to_flag.add(cell)
                if v == flagged:
                    for cell in covered: to_click.add(cell)
    return to_click, to_flag

def enumerate_frontier(frontier_cells, constraints, limit):
    n = len(frontier_cells)
    if n == 0: return [dict()]
    if n > limit: return None
    idx = {frontier_cells[i]: i for i in range(n)}
    valid = []
    for mask in range(1 << n):
        ok = True
        for cells, req in constraints:
            s = 0
            for c in cells:
                s += 1 if ((mask >> idx[c]) & 1) else 0
            if s != req:
                ok = False; break
        if ok:
            assign = {}
            for cell,i in idx.items():
                assign[cell] = ((mask >> i) & 1) == 1
            valid.append(assign)
    return valid

def build_frontier(perceived, rows, cols):
    frontier = set(); constraints = []
    for r in range(rows):
        for c in range(cols):
            v = perceived[r][c]
            if v >= 0:
                flagged = 0; covered = []
                for dr,dc in DIRS:
                    rr,cc = r+dr, c+dc
                    if not in_bounds(rr,cc,rows,cols): continue
                    pv = perceived[rr][cc]
                    if pv == -2: flagged += 1
                    elif pv == -1: covered.append((rr,cc))
                req = v - flagged
                if covered and req >= 0:
                    for cell in covered: frontier.add(cell)
                    constraints.append((tuple(covered), req))
    return list(frontier), constraints

def plan_move(perceived, rows, cols, enum_limit, tie_breaker, guess_randomness):
    to_click, to_flag = apply_simple(perceived, rows, cols)
    if to_click or to_flag:
        return to_click, to_flag, None
    frontier, raw_constraints = build_frontier(perceived, rows, cols)
    if not frontier:
        covered = [(r,c) for r in range(rows) for c in range(cols) if perceived[r][c] == -1]
        if not covered: return set(), set(), None
        covered.sort(key=lambda x: (abs(x[0]-rows//2)+abs(x[1]-cols//2), random.random()))
        return {covered[0]}, set(), None
    constraints = []
    for cells, req in raw_constraints:
        f_cells = [c for c in cells if c in frontier]
        if not f_cells: continue
        constraints.append((tuple(f_cells), req))
    assignments = enumerate_frontier(frontier, constraints, enum_limit)
    if assignments is None:
        cand = min(list(frontier), key=lambda cell: sum(1 for dr,dc in DIRS if in_bounds(cell[0]+dr, cell[1]+dc, rows, cols)))
        return {cand}, set(), None
    if not assignments:
        return {random.choice(list(frontier))}, set(), None
    freq = defaultdict(int)
    for a in assignments:
        for cell,val in a.items():
            if val: freq[cell] += 1
    total = len(assignments)
    certain_flags = set(); certain_safe = set()
    for cell in frontier:
        if freq[cell] == total: certain_flags.add(cell)
        elif freq[cell] == 0: certain_safe.add(cell)
    if certain_flags or certain_safe:
        return certain_safe, certain_flags, (freq, total)
    probs = {cell: (freq[cell]/total) for cell in frontier}
    if tie_breaker == "random":
        return {random.choice(list(frontier))}, set(), (freq,total)
    if tie_breaker == "min_prob":
        minp = min(probs.values())
        candidates = [c for c,p in probs.items() if abs(p-minp) < 1e-9]
        if random.random() < guess_randomness:
            chosen = random.choice(list(frontier))
        else:
            chosen = random.choice(candidates)
        return {chosen}, set(), (freq,total)
    if tie_breaker == "entropy":
        best = None; best_score = -1
        for cell,p in probs.items():
            if p <= 0 or p >= 1: s = 0
            else: s = -(p*math.log(p+1e-12) + (1-p)*math.log(1-p+1e-12))
            if s > best_score:
                best_score = s; best = cell
        return {best}, set(), (freq,total)
    # default
    chosen = min(probs.items(), key=lambda kv: kv[1])[0]
    return {chosen}, set(), (freq,total)

# ---------- Genome, AgentState, Worker ----------
@dataclass
class Genome:
    speed: int = field(default_factory=lambda: random.randint(1,10))
    enum_limit: int = field(default_factory=lambda: DEFAULT_ENUM_LIMIT)
    tie_breaker: str = field(default_factory=lambda: random.choice(["min_prob","random","entropy"]))
    guess_randomness: float = field(default_factory=lambda: random.random()*0.3)
    initial_click: str = field(default_factory=lambda: random.choice(["center","corner","random"]))
    def mutate(self):
        if random.random() < MUTATION_RATE:
            delta = int(round(random.gauss(0,1.2)))
            self.speed = max(1, min(10, self.speed + delta))
        if random.random() < MUTATION_RATE:
            factor = 1.0 + random.gauss(0,0.2)
            newv = int(max(2, min(40, round(self.enum_limit * factor))))
            self.enum_limit = newv
        if random.random() < MUTATION_RATE:
            self.tie_breaker = random.choice(["min_prob","random","entropy"])
        if random.random() < MUTATION_RATE:
            self.guess_randomness = max(0.0, min(0.9, self.guess_randomness + random.gauss(0,0.06)))
        if random.random() < (MUTATION_RATE * 0.5):
            self.initial_click = random.choice(["center","corner","random"])
    def copy(self):
        return Genome(speed=self.speed, enum_limit=self.enum_limit,
                      tie_breaker=self.tie_breaker, guess_randomness=self.guess_randomness,
                      initial_click=self.initial_click)

@dataclass
class AgentState:
    id: int
    genome: Genome
    score: int = 0
    games_played: int = 0
    wins: int = 0
    losses: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)
    # visual fields updated by worker for renderer to show
    perceived: List[List[int]] = field(default_factory=list)
    model_snapshot_revealed: List[List[bool]] = field(default_factory=list)
    model_snapshot_mines: List[List[bool]] = field(default_factory=list)
    model_over: bool = False
    model_win: bool = False

def agent_worker(agent: AgentState, stop_event: threading.Event):
    rng = random.Random(agent.id + int(time.time()))
    while not stop_event.is_set():
        model = MineModel(ROWS, COLS, MINES, safe_first_click=True)
        g = agent.genome
        # initial click
        if g.initial_click == "center":
            r0,c0 = ROWS//2, COLS//2
        elif g.initial_click == "corner":
            r0,c0 = 0,0
        else:
            r0,c0 = rng.randrange(0,ROWS), rng.randrange(0,COLS)
        model.reveal(r0,c0)
        # update visual once at start
        with agent.lock:
            agent.perceived = build_perceived(model)
            agent.model_snapshot_revealed = [row.copy() for row in model.revealed]
            agent.model_snapshot_mines = [row.copy() for row in model.is_mine]
            agent.model_over = model.game_over
            agent.model_win = model.win

        # play until end or stop
        while not model.game_over and not stop_event.is_set():
            perceived = build_perceived(model)
            to_click, to_flag, _ = plan_move(perceived, ROWS, COLS, g.enum_limit, g.tie_breaker, g.guess_randomness)
            for fr,fc in list(to_flag):
                model.toggle_flag(fr,fc)
                # update visual quickly
                with agent.lock:
                    agent.perceived = build_perceived(model)
                delay = speed_to_delay(g.speed) * GLOBAL_SLOWDOWN
                t0 = time.time()
                while time.time() - t0 < delay and not stop_event.is_set():
                    time.sleep(0.01)
            for cr,cc in list(to_click):
                if model.flagged[cr][cc]:
                    continue
                model.reveal(cr,cc)
                with agent.lock:
                    agent.perceived = build_perceived(model)
                delay = speed_to_delay(g.speed) * GLOBAL_SLOWDOWN
                t0 = time.time()
                while time.time() - t0 < delay and not stop_event.is_set():
                    time.sleep(0.01)
                if model.game_over:
                    break
            if not to_click and not to_flag:
                covered = [(r,c) for r in range(ROWS) for c in range(COLS) if perceived[r][c] == -1]
                if not covered:
                    break
                cand = rng.choice(covered)
                model.reveal(cand[0], cand[1])
                with agent.lock:
                    agent.perceived = build_perceived(model)
                delay = speed_to_delay(g.speed) * GLOBAL_SLOWDOWN
                t0 = time.time()
                while time.time() - t0 < delay and not stop_event.is_set():
                    time.sleep(0.01)
        # report result
        with agent.lock:
            agent.games_played += 1
            if model.win:
                agent.score += 1; agent.wins += 1
            else:
                agent.score -= 1; agent.losses += 1
            # snapshot for rendering: show mines only when game over
            agent.model_snapshot_revealed = [row.copy() for row in model.revealed]
            agent.model_snapshot_mines = [row.copy() for row in model.is_mine]
            agent.model_over = model.game_over
            agent.model_win = model.win
            # ensure perceived up to date
            agent.perceived = build_perceived(model)
        # loop restarts immediately for a new game

# ---------- Manager + Evolution ----------
def evolution_manager(agents: List[AgentState], stop_event: threading.Event):
    last_evolve = time.time()
    last_stats = time.time()
    # spawn threads
    threads = []
    for a in agents:
        t = threading.Thread(target=agent_worker, args=(a, stop_event), daemon=True)
        t.start(); threads.append(t)

    try:
        while not stop_event.is_set():
            now = time.time()
            if now - last_evolve >= EVOLUTION_INTERVAL_SEC:
                snapshot = sorted(agents, key=lambda ag: ag.score, reverse=True)
                for i in range(ELITE_COUNT):
                    src = snapshot[i]
                    dst = snapshot[-1 - i]
                    newg = src.genome.copy()
                    newg.mutate()
                    with dst.lock:
                        dst.genome = newg
                        dst.score = 0; dst.games_played = 0; dst.wins = 0; dst.losses = 0
                last_evolve = now
            if now - last_stats >= STATS_PRINT_INTERVAL:
                topk = min(10, len(agents))
                snapshot = sorted(agents, key=lambda ag: ag.score, reverse=True)
                print("\n=== Leaderboard Top {} ===".format(topk))
                for i in range(topk):
                    a = snapshot[i]
                    with a.lock:
                        avg = a.score / a.games_played if a.games_played else 0.0
                        print(f"#{i+1:2d} ID {a.id:4d} score={a.score:4d} games={a.games_played:4d} wins={a.wins:3d} losses={a.losses:3d} genome=[spd{a.genome.speed} enum{a.genome.enum_limit} tie={a.genome.tie_breaker} gr={a.genome.guess_randomness:.2f}]")
                total_games = sum(a.games_played for a in agents)
                total_score = sum(a.score for a in agents)
                avg_score = total_score / len(agents) if agents else 0.0
                print(f"Total games: {total_games}, total score: {total_score}, avg agent score: {avg_score:.3f}")
                last_stats = now
            time.sleep(0.25)
    finally:
        stop_event.set()
        for t in threads:
            t.join(timeout=1.0)

# ---------- Rendering helpers ----------
COLOR_BG = (22,22,22)
COLOR_GRID = (55,55,55)
COLOR_COVERED = (140,140,140)
COLOR_REVEALED = (220,220,220)
COLOR_FLAG = (200,40,40)
NUMBER_COLORS = {
    1: (0, 0, 160),
    2: (0, 150, 0),
    3: (150, 0, 0),
    4: (0, 150, 150),
    5: (150, 0, 150),
    6: (150, 80, 0),
    7: (0,0,0),
    8: (90,90,90)
}

def draw_agent_thumbnail(surf, x, y, w, h, agent: AgentState, rank:int=None, points:int=None, font=None):
    # compute per-cell size
    cell_w = max(2, w // COLS)
    cell_h = max(2, h // ROWS)
    # background rect
    pygame.draw.rect(surf, COLOR_GRID, (x, y, w, h))
    with agent.lock:
        perceived = agent.perceived if agent.perceived else [[-1]*COLS for _ in range(ROWS)]
        revealed_snapshot = agent.model_snapshot_revealed if agent.model_snapshot_revealed else [[False]*COLS for _ in range(ROWS)]
        mines_snapshot = agent.model_snapshot_mines if agent.model_snapshot_mines else [[False]*COLS for _ in range(ROWS)]
        over = agent.model_over
        win = agent.model_win
        # draw cells
        for r in range(ROWS):
            for c in range(COLS):
                px = x + c*cell_w
                py = y + r*cell_h
                rect = pygame.Rect(px, py, cell_w-1, cell_h-1)
                pv = perceived[r][c]
                if pv == -1:
                    pygame.draw.rect(surf, COLOR_COVERED, rect)
                elif pv == -2:
                    pygame.draw.rect(surf, COLOR_COVERED, rect)
                    # flag circle center
                    cx = px + cell_w//2; cy = py + cell_h//2
                    pygame.draw.circle(surf, COLOR_FLAG, (cx,cy), max(1, min(cell_w,cell_h)//4))
                else:
                    # revealed
                    pygame.draw.rect(surf, COLOR_REVEALED, rect)
                    if pv > 0:
                        # paint small center block with the number color
                        cx = px + cell_w//2; cy = py + cell_h//2
                        color = NUMBER_COLORS.get(pv, (0,0,0))
                        block = pygame.Rect(cx-1, cy-1, 3, 3)
                        surf.fill(color, block)
                # optionally show mines if game over (only for visualization)
                if over and not win and mines_snapshot[r][c]:
                    cx = px + cell_w//2; cy = py + cell_h//2
                    pygame.draw.circle(surf, (0,0,0), (cx,cy), max(1, min(cell_w,cell_h)//3))
    # border
    pygame.draw.rect(surf, (40,40,40), (x,y,w,h), 1)

    # draw rank & points badge top-left (contrasting small box)
    if font is not None and rank is not None and points is not None:
        badge_text = f"#{rank}  {points}"
        # badge background and border
        bx = x + 4
        by = y + 4
        bw = max(36, font.size(badge_text)[0] + 8)
        bh = max(16, font.get_linesize())
        pygame.draw.rect(surf, (10,10,10), (bx-1, by-1, bw+2, bh+2))  # slightly darker border
        pygame.draw.rect(surf, (30,30,30), (bx, by, bw, bh))
        txt_surf = font.render(badge_text, True, (220,220,220))
        surf.blit(txt_surf, (bx+4, by + (bh - font.get_linesize())//2))

# ---------- Main application ----------
def main():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    pygame.display.set_caption("Minesweeper Population — Single Window Visualization (rank+points)")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Consolas", 12)
    large_font = pygame.font.SysFont("Consolas", 18)

    # create agents
    agents = []
    for i in range(POPULATION_SIZE):
        genome = Genome()
        agents.append(AgentState(id=i, genome=genome))

    stop_event = threading.Event()
    # start evolution manager in separate thread (it will spawn agent worker threads)
    manager_thread = threading.Thread(target=evolution_manager, args=(agents, stop_event), daemon=True)
    manager_thread.start()

    # layout thumbnails in grid (compute columns/rows to fit)
    thumb_area_w = WINDOW_W - LEADERBOARD_WIDTH - 20
    thumb_area_h = WINDOW_H - 20
    # choose columns approx sqrt(pop * aspect)
    aspect = thumb_area_w / max(1, thumb_area_h)
    cols_thumb = max(1, int(math.ceil(math.sqrt(POPULATION_SIZE * aspect))))
    rows_thumb = int(math.ceil(POPULATION_SIZE / cols_thumb))
    thumb_w = thumb_area_w // cols_thumb
    thumb_h = thumb_area_h // rows_thumb

    # safety clamp to reasonable small thumbnails
    if thumb_w < 32 or thumb_h < 24:
        # if too small, reduce pop used for visual (we still run all agents headless; we will visualize a subset)
        VISUAL_LIMIT = min(POPULATION_SIZE, max(64, (thumb_area_w // 32) * (thumb_area_h // 24)))
        # pick VISUAL_LIMIT agents indices to display (rotate over time to show different ones)
        visual_indices = list(range(VISUAL_LIMIT))
        rotating = True
        rotation_index = 0
        cols_thumb = max(1, thumb_area_w // 32)
        rows_thumb = max(1, thumb_area_h // 24)
        thumb_w = thumb_area_w // cols_thumb
        thumb_h = thumb_area_h // rows_thumb
    else:
        VISUAL_LIMIT = POPULATION_SIZE
        visual_indices = list(range(POPULATION_SIZE))
        rotating = False

    print(f"Rendering {VISUAL_LIMIT} thumbnails ({cols_thumb}x{rows_thumb}), thumb size {thumb_w}x{thumb_h}")

    last_rotate = time.time()
    rotate_interval = 3.0

    running = True
    while running and not stop_event.is_set():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False; stop_event.set()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    print("SPACE pressed -> terminating...")
                    running = False; stop_event.set()

        # rotate visualized subset if needed
        if rotating and time.time() - last_rotate > rotate_interval:
            rotation_index = (rotation_index + 1) % POPULATION_SIZE
            visual_indices = [(rotation_index + i) % POPULATION_SIZE for i in range(VISUAL_LIMIT)]
            last_rotate = time.time()

        # Precompute ranking map and score map for badges (frame-consistent snapshot)
        snapshot_full = sorted(agents, key=lambda a: a.score, reverse=True)
        rank_by_id = {}
        score_by_id = {}
        # assign ranks (1..N), ties are broken by order in sorted list
        for idx, agent_obj in enumerate(snapshot_full, start=1):
            rank_by_id[agent_obj.id] = idx
            score_by_id[agent_obj.id] = agent_obj.score

        screen.fill(COLOR_BG)
        # draw thumbnails grid
        start_x = 10; start_y = 10
        idx = 0
        for row in range(rows_thumb):
            for col in range(cols_thumb):
                if idx >= VISUAL_LIMIT: break
                agent_idx = visual_indices[idx]
                ax = start_x + col * thumb_w
                ay = start_y + row * thumb_h
                # shrink slightly inside cell
                pad = THUMB_MARGIN
                # pass rank and points for overlay
                rnk = rank_by_id.get(agent_idx, None)
                pts = score_by_id.get(agent_idx, None)
                draw_agent_thumbnail(screen, ax + pad, ay + pad, thumb_w - pad*2, thumb_h - pad*2, agents[agent_idx], rank=rnk, points=pts, font=font)
                idx += 1
            if idx >= VISUAL_LIMIT: break

        # draw leaderboard area on right
        lb_x = WINDOW_W - LEADERBOARD_WIDTH + 10
        lb_y = 10
        pygame.draw.rect(screen, (18,18,18), (WINDOW_W - LEADERBOARD_WIDTH, 0, LEADERBOARD_WIDTH, WINDOW_H))
        title = large_font.render("Leaderboard (Top 20)", True, (235,235,235))
        screen.blit(title, (lb_x, lb_y))
        # sort snapshot
        snapshot = snapshot_full[:20]
        y = lb_y + 28
        for i,a in enumerate(snapshot):
            with a.lock:
                txt = f"#{i+1:2d} ID{a.id:4d} S{a.score:4d} G{a.games_played:4d} W{a.wins:3d} L{a.losses:3d} spd{a.genome.speed} enum{a.genome.enum_limit} tie:{a.genome.tie_breaker[:3]} gr:{a.genome.guess_randomness:.2f}"
            surf = font.render(txt, True, (220,220,220))
            screen.blit(surf, (lb_x, y))
            y += 18
            if y > WINDOW_H - 40: break

        # info text
        info_lines = [
            f"Population: {POPULATION_SIZE}  Visualized: {VISUAL_LIMIT}   Elites: {ELITE_COUNT}",
            f"Evolution every {EVOLUTION_INTERVAL_SEC}s  MutationRate: {MUTATION_RATE}",
            "Press SPACE to stop."
        ]
        iy = WINDOW_H - 60
        for line in info_lines:
            surf = font.render(line, True, (200,200,200))
            screen.blit(surf, (10, iy))
            iy += 16

        pygame.display.flip()
        clock.tick(30)

    # stop and join manager (which stops agent threads)
    stop_event.set()
    manager_thread.join(timeout=2.0)
    pygame.quit()
    print("Exited cleanly.")

if __name__ == "__main__":
    main()
