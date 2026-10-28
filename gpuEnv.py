'''
GPU evaluation of many games (numba.cuda): one GPU thread plays one whole game, from the reset to the end, including the network's forward pass.

The game code is NOT copied: the device functions (Mersenne Twister, level generation, one frame, observation with landing prediction, counters) are
compiled from fastEnv.py's own source (ast), with `@njit` turned into `@cuda.jit(device=True)` and np.floor / np.ceil / np.sqrt / np.inf into the math module.
A change of the rules in fastEnv.py therefore reaches the GPU version automatically. Only the driver (reset, forward pass, decision loop) lives here.

Differences to the CPU version: the forward pass sums in a different order and the GPU fuses multiply-adds, so a game is not bit-identical to the CPU game
(the integer parts - levels, random numbers - are); the games diverge after some hundreds of frames, the statistics are the same (see gpuCheck.py).

Needs the environment ~/venv-gpu (numba, numba-cuda, nvidia-cuda-nvcc / nvrtc / runtime, numpy < 2.4, scipy). The kernel runs in chunks of decisions
(Windows stops GPU kernels that run longer than ~2 s).

    from gpuEnv import GpuBatch
    gb = GpuBatch(children=800, games=50)
    r = gb.evaluate(weights, seeds)      # weights: 8 float32 arrays [C, ...] (see GpuBatch.evaluate); returns scores and counters per game, shape [C, games]
'''
import ast
import math
import os
import re
import sys

import numpy as np
from numba import cuda, float32, int64

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

WANTED = ("mtSeed", "mtNext", "rndRandom", "rndUniform", "rndRandint", "addPlatform", "spawnPlatformAbove", "resetGame", "stepGame",
          "relXY", "clampRel", "slotIds", "monsterIds", "landFrames", "landPath", "pathSum", "landFramesPathChecked", "landFramesPath",
          "landingFeatures", "observeGame")


def toTransposed(text):
    '''Rewrites the state accesses of a function from [game, ...] to [..., game] (structure of arrays): F[g, X] -> F[X, g], PL[g, i, c] -> PL[c, i, g], MT[g] -> MT[:, g].
    Neighbouring GPU threads play neighbouring games, so with the game index last their reads and writes land next to each other in memory (coalesced).'''
    text = text.replace("MT[g]", "MT[:, g]")
    pattern = re.compile(r"\b(F|I|PL|MO|BU)\[g,")
    out, pos = [], 0
    while True:
        m = pattern.search(text, pos)
        if not m:
            out.append(text[pos:])
            return "".join(out)
        out.append(text[pos:m.start()])
        start = text.index("[", m.start())
        depth, j = 0, start
        while True:
            depth += (text[j] == "[") - (text[j] == "]")
            if depth == 0:
                break
            j += 1
        parts, d, cur = [], 0, ""
        for ch in text[start + 1:j]:
            d += (ch in "[(") - (ch in "])")
            if ch == "," and d == 0:
                parts.append(cur.strip())
                cur = ""
            else:
                cur += ch
        parts.append(cur.strip())
        assert parts[0] == "g", parts
        rest = [toTransposed(p) for p in parts[1:]]
        out.append(m.group(1) + "[" + ", ".join(rest[::-1] + ["g"]) + "]")
        pos = j + 1


def deviceNamespace():
    '''fastEnv.py's constants and the WANTED functions, compiled as CUDA device functions.'''
    source = open(os.path.join(HERE, "fastEnv.py")).read()
    tree = ast.parse(source)
    ns = {"math": math, "np": np, "cuda": cuda}
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == "env":
            exec(ast.get_source_segment(source, node), ns)
        elif isinstance(node, ast.Assign):
            exec(ast.get_source_segment(source, node), ns)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in WANTED:
            text = ast.get_source_segment(source, node)
            text = toTransposed(text)
            text = text.replace("np.floor", "math.floor").replace("np.ceil", "math.ceil").replace("np.sqrt", "math.sqrt").replace("np.inf", "math.inf")
            exec("@cuda.jit(device=True)\n" + text, ns)
    return ns


NS = deviceNamespace()
for _name in WANTED:
    globals()[_name] = NS[_name]
NIN = NS["OBSERVATION_LENGTH"]
NPL = NS["PLATFORMS_BELOW"] + NS["PLATFORMS_ABOVE"]
NMO = NS["MONSTERS_BELOW"] + NS["MONSTERS_ABOVE"]
NH = 33    # hidden width of the network (33-33-33-33-2)
(PX, PY, VY, TOTAL, OFFSET, BEST, MMULT, TOPY, SHOTS, SHOTS_FREE, STEER_VAR, LAST_STEER, DECISIONS, HITS, DETOUR, PATH_SUM, SIGNED_SUM,
 DEV_N, DEV_SQ, DEV_TD, DEVIATION) = (NS[n] for n in ("PX", "PY", "VY", "TOTAL", "OFFSET", "BEST", "MMULT", "TOPY", "SHOTS", "SHOTS_FREE", "STEER_VAR", "LAST_STEER",
                                                     "DECISIONS", "HITS", "DETOUR", "PATH_SUM", "SIGNED_SUM", "DEV_N", "DEV_SQ", "DEV_TD", "DEVIATION"))
(COOL, FWP, NP, NM, NB, DONE, BOUNCED, CAUSE) = (NS[n] for n in ("COOL", "FWP", "NP", "NM", "NB", "DONE", "BOUNCED", "CAUSE"))
MAXP, MAXM, MAXB = NS["MAXP"], NS["MAXM"], NS["MAXB"]
NF, NI = 21, 8    # float / int state columns per game (FastBatch.__init__)


@cuda.jit
def initKernel(F, I, PL, MO, MT, seeds, maxStart, minStart, zeroFraction, monsterFraction, monsterMult):
    g = cuda.grid(1)
    if g >= F.shape[1]:
        return
    for j in range(F.shape[0]):
        F[j, g] = 0.0
    for j in range(I.shape[0]):
        I[j, g] = 0
    F[LAST_STEER, g] = 0.5
    resetGame(F, I, PL, MO, MT, g, seeds[g], maxStart, minStart, zeroFraction, monsterFraction, monsterMult)


MAX_REGISTERS = int(os.environ.get("GPU_MAXREG", "0")) or None    # cap on registers per thread (more resident threads per SM, some values spill to local memory)


@cuda.jit(max_registers=MAX_REGISTERS)
def playKernel(F, I, PL, MO, BU, MT, applied, W0, B0, W1, B1, W2, B2, W3, B3, S, smooth, repeat, maxNoProgress, maxDecisions, maxFrames, origin, n, running):
    '''Thread g plays up to maxDecisions network decisions of the game in slot g (g < n; origin[g] = its game number, child = origin // S); running[g] = 1 if
    that game is still going afterwards (not over, maxFrames not reached).'''
    g = cuda.grid(1)
    if g >= n:
        return
    c = origin[g] // S    # the child that owns this game
    x = cuda.local.array(NIN, float32)
    h = cuda.local.array(NH, float32)
    h2 = cuda.local.array(NH, float32)
    ids = cuda.local.array(NPL, int64)
    mids = cuda.local.array(NMO, int64)
    d = 0
    while d < maxDecisions and I[DONE, g] == 0 and F[DECISIONS, g] * repeat < maxFrames:
        observeGame(F, I, PL, MO, g, x, ids, mids)
        # each layer as "for every input i: add x[i] * row i of W to all outputs": the multiply-adds of one row are independent (the GPU can overlap them)
        # and W is read along its rows; every output still sums its inputs in the order i = 0, 1, 2, ... starting from the bias, as before
        for j in range(NH):
            h[j] = B0[c, j]
        for i in range(NIN):
            xi = x[i]
            for j in range(NH):
                h[j] += xi * W0[c, i, j]
        for j in range(NH):
            h2[j] = B1[c, j]
        for i in range(NH):
            xi = h[i] if h[i] > 0.0 else float32(0.0)
            for j in range(NH):
                h2[j] += xi * W1[c, i, j]
        for j in range(NH):
            h[j] = B2[c, j]
        for i in range(NH):
            xi = h2[i] if h2[i] > 0.0 else float32(0.0)
            for j in range(NH):
                h[j] += xi * W2[c, i, j]
        a0 = B3[c, 0]
        a1 = B3[c, 1]
        for i in range(NH):
            xi = h[i] if h[i] > 0.0 else float32(0.0)
            a0 += xi * W3[c, i, 0]
            a1 += xi * W3[c, i, 1]
        steer = float32(1.0) / (float32(1.0) + math.exp(-a0))
        shoot = float32(1.0) / (float32(1.0) + math.exp(-a1))
        st = float(steer)
        if smooth < 1.0:
            applied[g] += smooth * (st - applied[g])
            st = applied[g]
        F[STEER_VAR, g] += abs(st - F[LAST_STEER, g])
        F[LAST_STEER, g] = st
        F[DECISIONS, g] += 1.0
        sh = float(shoot)
        I[BOUNCED, g] = 0
        for _ in range(repeat):
            if stepGame(F, I, PL, MO, BU, MT, g, st, sh, maxNoProgress):
                I[DONE, g] = 1
                break
        d += 1
    running[g] = 1 if (I[DONE, g] == 0 and F[DECISIONS, g] * repeat < maxFrames) else 0


@cuda.jit
def gatherKernel(F, I, PL, MO, BU, MT, applied, origin, F2, I2, PL2, MO2, BU2, MT2, applied2, origin2, idx, m):
    '''Stream compaction: slot t of the second set of arrays gets the whole state of slot idx[t] of the first (t < m; idx ascending, so the games of one child
    stay next to each other).'''
    t = cuda.grid(1)
    if t >= m:
        return
    s = idx[t]
    for j in range(F.shape[0]):
        F2[j, t] = F[j, s]
    for j in range(I.shape[0]):
        I2[j, t] = I[j, s]
    for k in range(PL.shape[1]):
        for j in range(PL.shape[0]):
            PL2[j, k, t] = PL[j, k, s]
    for k in range(MO.shape[1]):
        for j in range(MO.shape[0]):
            MO2[j, k, t] = MO[j, k, s]
    for k in range(BU.shape[1]):
        for j in range(BU.shape[0]):
            BU2[j, k, t] = BU[j, k, s]
    for j in range(MT.shape[0]):
        MT2[j, t] = MT[j, s]
    applied2[t] = applied[s]
    origin2[t] = origin[s]


class GpuBatch:
    '''children x games games on the GPU (game g belongs to child g // games). Arrays are allocated once and reset for every evaluation.'''

    def __init__(self, children, games):
        self.C, self.S = children, games
        G = children * games
        self.G = G
        # two sets of state arrays (game index last, see toTransposed): the games still running are copied densely from one set into the other (gatherKernel)
        self.sets = [dict(F=cuda.device_array((NF, G), dtype=np.float64), I=cuda.device_array((NI, G), dtype=np.int64),
                          PL=cuda.device_array((4, MAXP, G), dtype=np.float64), MO=cuda.device_array((2, MAXM, G), dtype=np.float64),
                          BU=cuda.device_array((2, MAXB, G), dtype=np.float64), MT=cuda.device_array((625, G), dtype=np.int64),
                          applied=cuda.device_array(G, dtype=np.float64), origin=cuda.device_array(G, dtype=np.int64)) for _ in range(2)]
        self.running = cuda.device_array(G, dtype=np.int8)
        self.threads = 64    # 32-64 measured best (2.58 s vs 2.72 s at 128, 2.94 s at 256; 512 does not fit the registers)
        self.blocks = (G + self.threads - 1) // self.threads

    def evaluate(self, weights, seeds, smooth=0.96875, repeat=5, maxStart=None, minStart=0, zeroFraction=0.0, monsterFraction=0.0, monsterMult=1.0,
                 maxFrames=100_000, chunk=400, maxNoProgress=600, compactBelow=0.95):
        '''weights: (W0, B0, W1, B1, W2, B2, W3, B3), float32, W [C, in, out] (the transposed torch weights), B [C, out]. seeds: `games` ints, every child plays the same.
        Returns a dict of arrays [C, games]: score and the counters (shots, shotsFree, steerChange, hits, detour, deviation, decisions) of fastEnv.
        The games run in chunks of `chunk` decisions; after a chunk, when fewer than compactBelow of the games in the threads are still running, those are copied
        densely into the other set of arrays (a warp otherwise runs until its longest game is over: ~30% of the lanes did work, the mean game has ~3,700
        decisions, the longest 20,000).'''
        dev = [cuda.to_device(np.ascontiguousarray(w, dtype=np.float32)) for w in weights]
        cur, other = self.sets
        G, tpb = self.G, self.threads
        seedsAll = cuda.to_device(np.tile(np.asarray(seeds, dtype=np.int64), self.C))
        initKernel[(G + tpb - 1) // tpb, tpb](cur["F"], cur["I"], cur["PL"], cur["MO"], cur["MT"], seedsAll, -1 if maxStart is None else int(maxStart), int(minStart),
                                              float(zeroFraction), float(monsterFraction), float(monsterMult))
        cur["applied"].copy_to_device(np.full(G, 0.5))
        origin = np.arange(G, dtype=np.int64)
        cur["origin"].copy_to_device(origin)
        results = np.zeros((G, NF))
        n, chunks = G, 0
        while n:
            playKernel[(n + tpb - 1) // tpb, tpb](cur["F"], cur["I"], cur["PL"], cur["MO"], cur["BU"], cur["MT"], cur["applied"], *dev, self.S, float(smooth), int(repeat),
                                                  int(maxNoProgress), int(chunk), int(maxFrames), cur["origin"], n, self.running)
            chunks += 1
            still = self.running[:n].copy_to_host() == 1
            done = ~still
            if done.any():
                Fh = cur["F"].copy_to_host()
                results[origin[:n][done]] = Fh[:, :n][:, done].T
            if not still.any():
                break
            if still.sum() < compactBelow * n:
                idx = np.flatnonzero(still)
                m = idx.size
                gatherKernel[(m + tpb - 1) // tpb, tpb](cur["F"], cur["I"], cur["PL"], cur["MO"], cur["BU"], cur["MT"], cur["applied"], cur["origin"],
                                                        other["F"], other["I"], other["PL"], other["MO"], other["BU"], other["MT"], other["applied"], other["origin"],
                                                        cuda.to_device(idx), m)
                cur, other = other, cur
                origin = origin[:n][idx]
                n = m
        shape = (self.C, self.S)
        col = lambda j: results[:, j].reshape(shape)
        return dict(score=(results[:, TOTAL] - results[:, OFFSET]).reshape(shape), shots=col(SHOTS), shotsFree=col(SHOTS_FREE), steerChange=col(STEER_VAR), hits=col(HITS),
                    detour=col(DETOUR), deviation=col(DEVIATION), decisions=col(DECISIONS), chunks=chunks)
