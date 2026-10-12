"""Interactive pygame viewer for the 2-link arm.

Run with ``python -m src.viz`` from the repo root.

Shows the RK4 ground truth (the same ``simulate`` used to build the dataset) for
any payload / friction / torque setting, with live re-solving as sliders move.
Slider tracks shade the in-range domain; going past it is out-of-range territory.

Keys:
    SPACE pause/play    R restart      [ / ]  slower / faster
    G     toggle the passive ghost arm (tau = 0, i.e. a pure double pendulum)
    K     toggle RK45 reference overlay (adaptive solver, should sit on RK4)
    N     load a random test sample from data/dataset.npz (O: out-of-range sample)
    D     reset sliders to defaults      P  save a screenshot to results/
Click or drag on any time-series plot to scrub.

Surrogates (DeepONet / MLP) plug in through ``SURROGATES``: a name -> callable
``f(params, coef) -> traj (N_T, 4)`` mapping. They are drawn as extra arms and
traces once registered.
"""
from dataclasses import replace
from pathlib import Path

import numpy as np
import pygame

from .data import RANGES, OOD_RANGES, T_FINAL, T_GRID, X0, load, simulate, torque_fn
from .dynamics import ArmParams, energy, state_derivative
from .solvers import rk45_integrate

SURROGATES: dict = {}  # e.g. SURROGATES["DeepONet"] = lambda params, coef: model_predict(...)

W, H = 1180, 720
ARM_W = 560
PX_PER_M = 300
BASE = (ARM_W // 2, 330)
BASE_ARM = ArmParams()

BG = (24, 26, 32)
PANEL = (34, 37, 46)
FG = (225, 228, 235)
DIM = (130, 136, 150)
C_GT = (90, 190, 255)
C_RK45 = (255, 200, 80)
C_GHOST = (120, 124, 138)
C_Q1, C_Q2 = (110, 220, 150), (255, 130, 130)
SUR_COLORS = [(210, 130, 255), (255, 160, 90), (130, 255, 230)]
RESULTS = Path(__file__).resolve().parent.parent / "results"


class Slider:
    """Horizontal slider. The in-range part of the track is drawn brighter."""

    def __init__(self, rect, label, lo, hi, val, safe=None, fmt="{:.2f}"):
        self.rect = pygame.Rect(rect)
        self.label, self.lo, self.hi, self.val = label, lo, hi, val
        self.default, self.safe, self.fmt = val, safe, fmt
        self.drag = False

    def _x(self, v):
        return self.rect.x + (v - self.lo) / (self.hi - self.lo) * self.rect.w

    def handle(self, ev) -> bool:
        """Process a mouse event; True if the value changed."""
        if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
            if self.rect.inflate(0, 14).collidepoint(ev.pos):
                self.drag = True
        elif ev.type == pygame.MOUSEBUTTONUP and ev.button == 1:
            self.drag = False
        if self.drag and ev.type in (pygame.MOUSEBUTTONDOWN, pygame.MOUSEMOTION):
            frac = np.clip((ev.pos[0] - self.rect.x) / self.rect.w, 0, 1)
            new = self.lo + frac * (self.hi - self.lo)
            changed = new != self.val
            self.val = float(new)
            return changed
        return False

    def draw(self, surf, font):
        r = self.rect
        pygame.draw.rect(surf, (55, 60, 72), (r.x, r.centery - 2, r.w, 4), border_radius=2)
        if self.safe:
            x0, x1 = self._x(self.safe[0]), self._x(self.safe[1])
            pygame.draw.rect(surf, (70, 120, 160), (x0, r.centery - 2, x1 - x0, 4), border_radius=2)
        pygame.draw.circle(surf, FG, (int(self._x(self.val)), r.centery), 7)
        surf.blit(font.render(self.label, True, DIM), (r.x, r.y - 17))
        txt = font.render(self.fmt.format(self.val), True, FG)
        surf.blit(txt, (r.right - txt.get_width(), r.y - 17))


def draw_plot(surf, rect, font, t, series, cursor_t, title, unit=""):
    """Draw line series [(label, color, y_array)] against t with a time cursor."""
    pygame.draw.rect(surf, PANEL, rect, border_radius=6)
    ys = np.concatenate([s[2] for s in series])
    lo, hi = float(ys.min()), float(ys.max())
    pad = 0.1 * (hi - lo) if hi > lo else 1.0
    lo, hi = lo - pad, hi + pad
    inner = rect.inflate(-16, -30)
    inner.y += 8

    def px(tt, yy):
        return (inner.x + tt / T_FINAL * inner.w, inner.bottom - (yy - lo) / (hi - lo) * inner.h)

    if lo < 0 < hi:
        pygame.draw.line(surf, (60, 65, 78), px(0, 0), px(T_FINAL, 0))
    for _, col, y, width in series:
        pygame.draw.lines(surf, col, False, [px(a, b) for a, b in zip(t, y)], width)
    cx = px(cursor_t, 0)[0]
    pygame.draw.line(surf, (200, 200, 210), (cx, inner.y), (cx, inner.bottom))
    surf.blit(font.render(f"{title} {unit}", True, DIM), (rect.x + 8, rect.y + 4))
    lx = rect.right - 8
    seen = set()
    for label, col, *_ in reversed(series):
        if label in seen:
            continue
        seen.add(label)
        img = font.render(label, True, col)
        lx -= img.get_width()
        surf.blit(img, (lx, rect.y + 4))
        lx -= 12
    surf.blit(font.render(f"{hi:.1f}", True, DIM), (rect.x + 8, inner.y))
    surf.blit(font.render(f"{lo:.1f}", True, DIM), (rect.x + 8, inner.bottom - 12))


def fk(q, p: ArmParams):
    """Joint positions (elbow, tip) in metres, y up."""
    e = np.array([p.l1 * np.cos(q[0]), p.l1 * np.sin(q[0])])
    tip = e + np.array([p.l2 * np.cos(q[0] + q[1]), p.l2 * np.sin(q[0] + q[1])])
    return e, tip


def to_px(pt):
    return (BASE[0] + pt[0] * PX_PER_M, BASE[1] - pt[1] * PX_PER_M)


def draw_arm(surf, q, p, color, width=10, alpha_tip=True):
    e, tip = fk(q, p)
    pts = [BASE, to_px(e), to_px(tip)]
    pygame.draw.lines(surf, color, False, pts, width)
    for pt in pts:
        pygame.draw.circle(surf, color, pt, width // 2 + 2)
    if alpha_tip and p.m_p > 0:
        pygame.draw.circle(surf, color, pts[2], int(6 + 14 * p.m_p), 2)


def draw_torque(surf, center, tau, color):
    """Arc whose sweep is proportional to the torque (anticlockwise = positive)."""
    sweep = np.clip(tau / RANGES["A"][1], -1, 1) * np.pi
    if abs(sweep) < 0.05:
        return
    r = pygame.Rect(0, 0, 44, 44)
    r.center = center
    a0, a1 = (0.0, sweep) if sweep > 0 else (sweep, 0.0)
    pygame.draw.arc(surf, color, r, a0, a1, 3)


class App:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((W, H))
        pygame.display.set_caption("2-link arm: DeepONet project visualiser")
        self.font = pygame.font.SysFont("consolas", 14)
        self.big = pygame.font.SysFont("consolas", 18, bold=True)
        self.clock = pygame.time.Clock()
        self.data = None  # lazily loaded dataset

        sx, sw = ARM_W + 30, 250
        specs = [
            ("m_p (kg)", 0.0, OOD_RANGES["m_p"][1], 0.5, RANGES["m_p"], "{:.2f}"),
            ("b (N m s/rad)", RANGES["b"][0], OOD_RANGES["b"][1], 0.2, RANGES["b"], "{:.2f}"),
        ]
        for j in (1, 2):
            specs += [
                (f"A{j} (N m)", 0.0, RANGES["A"][1], 8.0, RANGES["A"], "{:.2f}"),
                (f"f{j} (Hz)", RANGES["f"][0], RANGES["f"][1], 0.5 + 0.2 * j, RANGES["f"], "{:.2f}"),
                (f"phi{j} (rad)", 0.0, 2 * np.pi, 0.0 if j == 1 else 1.0, None, "{:.2f}"),
            ]
        self.sliders = []
        for i, (lab, lo, hi, v, safe, fmt) in enumerate(specs):
            if i < 2:  # m_p and b share the top row
                col, row = i, 0
            else:  # torque sliders: joint 1 left, joint 2 right
                col, row = divmod(i - 2, 3)[0], divmod(i - 2, 3)[1] + 1
            rect = (sx + col * (sw + 30), 50 + row * 42, sw, 14)
            self.sliders.append(Slider(rect, lab, lo, hi, v, safe, fmt))

        self.t = 0.0
        self.speed = 1.0
        self.paused = False
        self.show_rk45 = False
        self.show_ghost = True
        self.trail_len = 80
        self.solve()

    # ---- model ----------------------------------------------------------
    def inputs(self):
        v = [s.val for s in self.sliders]
        params = np.array([v[0], v[1]])
        coef = np.array([[v[2], v[3], v[4]], [v[5], v[6], v[7]]])
        return params, coef

    def set_inputs(self, params, coef):
        vals = [params[0], params[1], *coef[0], *coef[1]]
        for s, v in zip(self.sliders, vals):
            s.val = float(np.clip(v, s.lo, s.hi))

    def solve(self):
        params, coef = self.inputs()
        self.p = replace(BASE_ARM, m_p=float(params[0]), b=float(params[1]))
        self.traj = simulate(params, coef)
        self.tau = torque_fn(coef)
        self.passive = simulate(params, np.zeros_like(coef))
        self.tau_grid = np.array([self.tau(t) for t in T_GRID])
        self.energy = np.array([energy(x, self.p) for x in self.traj])
        self.rk45 = None
        if self.show_rk45:
            self.solve_rk45()
        self.surr = {n: f(params, coef) for n, f in SURROGATES.items()}

    def solve_rk45(self):
        f = lambda t, x: state_derivative(t, x, self.p, self.tau)
        self.rk45 = rk45_integrate(f, X0, T_GRID)

    def state_at(self, traj):
        return np.array([np.interp(self.t, T_GRID, traj[:, k]) for k in range(4)])

    def load_sample(self, split):
        if self.data is None:
            try:
                self.data = load()
            except OSError:
                return
        i = np.random.randint(len(self.data[f"{split}_params"]))
        self.set_inputs(self.data[f"{split}_params"][i], self.data[f"{split}_tau_coef"][i])
        self.t = 0.0
        self.solve()

    # ---- events ---------------------------------------------------------
    def handle(self, ev):
        changed = False
        for s in self.sliders:
            changed |= s.handle(ev)
        if changed:
            self.solve()
        if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1 and ev.pos[0] > ARM_W and ev.pos[1] > 290:
            self.scrub = True
        if ev.type == pygame.MOUSEBUTTONUP:
            self.scrub = False
        if getattr(self, "scrub", False) and ev.type in (pygame.MOUSEBUTTONDOWN, pygame.MOUSEMOTION):
            frac = np.clip((ev.pos[0] - (ARM_W + 24)) / (W - ARM_W - 56), 0, 1)
            self.t = float(frac * T_FINAL)
        if ev.type == pygame.KEYDOWN:
            k = ev.key
            if k == pygame.K_SPACE:
                self.paused = not self.paused
            elif k == pygame.K_r:
                self.t = 0.0
            elif k == pygame.K_LEFTBRACKET:
                self.speed = max(0.125, self.speed / 2)
            elif k == pygame.K_RIGHTBRACKET:
                self.speed = min(8.0, self.speed * 2)
            elif k == pygame.K_g:
                self.show_ghost = not self.show_ghost
            elif k == pygame.K_k:
                self.show_rk45 = not self.show_rk45
                if self.show_rk45:
                    self.solve_rk45()
            elif k == pygame.K_n:
                self.load_sample("test")
            elif k == pygame.K_o:
                self.load_sample("ood")
            elif k == pygame.K_d:
                for s in self.sliders:
                    s.val = s.default
                self.solve()
            elif k == pygame.K_p:
                RESULTS.mkdir(exist_ok=True)
                path = RESULTS / "viz_screenshot.png"
                pygame.image.save(self.screen, str(path))
                print("saved", path)

    # ---- drawing --------------------------------------------------------
    def draw_arm_panel(self):
        s = self.screen
        pygame.draw.rect(s, PANEL, (10, 10, ARM_W - 10, H - 20), border_radius=8)
        # reach circle and ground reference
        pygame.draw.circle(s, (45, 49, 60), BASE, int((self.p.l1 + self.p.l2) * PX_PER_M), 1)
        pygame.draw.line(s, (45, 49, 60), (20, BASE[1]), (ARM_W - 10, BASE[1]))

        idx = int(round(self.t / T_FINAL * (len(T_GRID) - 1)))
        lo = max(0, idx - self.trail_len)
        for name, traj, col in self.arm_sets():
            tips = [to_px(fk(x[:2], self.p)[1]) for x in traj[lo : idx + 1]]
            if len(tips) > 1:
                pygame.draw.lines(s, tuple(c // 2 for c in col), False, tips, 2)
        if self.show_ghost:
            draw_arm(s, self.state_at(self.passive)[:2], self.p, C_GHOST, width=4, alpha_tip=False)
        for name, traj, col in reversed(self.arm_sets()):
            q = self.state_at(traj)[:2]
            draw_arm(s, q, self.p, col, width=10 if name == "RK4" else 5, alpha_tip=name == "RK4")
        x = self.state_at(self.traj)
        tau = self.tau(self.t)
        draw_torque(s, BASE, tau[0], C_Q1)
        draw_torque(s, to_px(fk(x[:2], self.p)[0]), tau[1], C_Q2)

        s.blit(self.big.render("RK4 ground truth", True, C_GT), (24, 22))
        y = 46
        legend = self.arm_sets()[1:] + ([("passive (tau = 0)", None, C_GHOST)] if self.show_ghost else [])
        for name, _, col in legend:
            s.blit(self.font.render(name, True, col), (24, y))
            y += 18
        info = [
            f"t = {self.t:4.2f} / {T_FINAL:.1f} s   speed x{self.speed:g}" + ("  [paused]" if self.paused else ""),
            f"q1 {x[0]:+.2f}  q2 {x[1]:+.2f} rad    qd1 {x[2]:+.2f}  qd2 {x[3]:+.2f} rad/s",
            f"tau1 {tau[0]:+.2f}  tau2 {tau[1]:+.2f} N m   E = {energy(x, self.p):+.2f} J",
        ]
        in_range = self.in_range()
        info.append("in-range" if in_range else "OUT-OF-RANGE (surrogate untrained here)")
        for i, line in enumerate(info):
            col = (255, 120, 120) if i == 3 and not in_range else DIM
            s.blit(self.font.render(line, True, col), (24, H - 100 + i * 18))

    def in_range(self):
        return all(sl.safe is None or sl.safe[0] <= sl.val <= sl.safe[1] for sl in self.sliders)

    def arm_sets(self):
        sets = [("RK4", self.traj, C_GT)]
        if self.show_rk45 and self.rk45 is not None:
            sets.append(("RK45 (adaptive)", self.rk45, C_RK45))
        for i, (n, tr) in enumerate(self.surr.items()):
            sets.append((n, tr, SUR_COLORS[i % len(SUR_COLORS)]))
        return sets

    def draw_panels(self):
        s = self.screen
        s.blit(self.big.render("Parameters (bright track = training range)", True, FG), (ARM_W + 30, 12))
        for sl in self.sliders:
            sl.draw(s, self.font)
        x0, wid = ARM_W + 10, W - ARM_W - 20
        rects = [pygame.Rect(x0, 290 + i * 142, wid, 134) for i in range(3)]
        q = [("q1", C_Q1, self.traj[:, 0], 2), ("q2", C_Q2, self.traj[:, 1], 2)]
        qd = [("qd1", C_Q1, self.traj[:, 2], 2), ("qd2", C_Q2, self.traj[:, 3], 2)]
        for n, tr in self.surr.items():
            col = SUR_COLORS[list(self.surr).index(n) % len(SUR_COLORS)]
            q += [(n, col, tr[:, 0], 1), (n, col, tr[:, 1], 1)]
            qd += [(n, col, tr[:, 2], 1), (n, col, tr[:, 3], 1)]
        if self.show_rk45 and self.rk45 is not None:
            q += [("RK45", C_RK45, self.rk45[:, 0], 1), ("RK45", C_RK45, self.rk45[:, 1], 1)]
            qd += [("RK45", C_RK45, self.rk45[:, 2], 1), ("RK45", C_RK45, self.rk45[:, 3], 1)]
        draw_plot(s, rects[0], self.font, T_GRID, q, self.t, "joint angles", "(rad)")
        draw_plot(s, rects[1], self.font, T_GRID, qd, self.t, "joint velocities", "(rad/s)")
        e_series = [("E", FG, self.energy, 2)]
        draw_plot(s, rects[2], self.font, T_GRID, e_series, self.t, "total energy", "(J)")

    def run(self, max_frames=None, screenshot=None):
        self.scrub = False
        frames = 0
        running = True
        while running:
            dt = self.clock.tick(60) / 1000
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    running = False
                else:
                    self.handle(ev)
            if not self.paused:
                self.t += dt * self.speed
                if self.t > T_FINAL:
                    self.t = 0.0
            self.screen.fill(BG)
            self.draw_arm_panel()
            self.draw_panels()
            pygame.display.flip()
            frames += 1
            if max_frames and frames >= max_frames:
                if screenshot:
                    pygame.image.save(self.screen, str(screenshot))
                running = False
        pygame.quit()


def main():
    App().run()


if __name__ == "__main__":
    main()
