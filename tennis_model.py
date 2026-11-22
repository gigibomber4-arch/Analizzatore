"""
Motore tennis: Elo per superficie + simulazione di set e game (catena di Markov).
Dati: tennis-data.co.uk (ATP e WTA, gratuiti). Nessuna chiave necessaria.
"""
import io
import os

import numpy as np
import pandas as pd

BASE = "http://www.tennis-data.co.uk"
SURFACES = ["Hard", "Clay", "Grass"]
SET_COLS = [f"{s}{i}" for i in range(1, 6) for s in "WL"]
ODDS_COLS = ["AvgW", "AvgL", "PSW", "PSL", "B365W", "B365L"]


# ---------------------------------------------------------------- dati
def season_years(n=4):
    y = pd.Timestamp.today().year
    return list(range(y - n + 1, y + 1))


def _read(src):
    if str(src).startswith("http"):
        import requests
        r = requests.get(src, headers={"User-Agent": "Mozilla/5.0"}, timeout=60)
        r.raise_for_status()
        return pd.read_excel(io.BytesIO(r.content))
    return pd.read_csv(src) if str(src).endswith(".csv") else pd.read_excel(src)


def load_tennis(tour="ATP", local_dir=None):
    frames = []
    for y in season_years():
        srcs = ([os.path.join(local_dir, f"{tour}_{y}.xlsx"), os.path.join(local_dir, f"{tour}_{y}.csv")]
                if local_dir else [f"{BASE}/{y}{'w' if tour == 'WTA' else ''}/{y}.xlsx"])
        for src in srcs:
            try:
                d = _read(src)
                d.columns = [str(c).strip() for c in d.columns]
                frames.append(d)
                break
            except Exception as e:
                last_err = e
        else:
            print(f"! {tour} {y} non caricato ({last_err})")
    if not frames:
        raise RuntimeError("Nessun dato tennis caricato")
    df = pd.concat(frames, ignore_index=True).rename(columns={"Best of": "BestOf"})
    df = df.reindex(columns=["Date", "Surface", "Winner", "Loser", "Comment", "BestOf"] + SET_COLS + ODDS_COLS)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Surface"] = df["Surface"].replace({"Carpet": "Hard"})
    df["Comment"] = df["Comment"].fillna("").astype(str)
    df = df.dropna(subset=["Date", "Winner", "Loser", "Surface"])
    df = df[df.Surface.isin(SURFACES) & ~df.Comment.str.lower().str.contains("walkover")]
    for c in SET_COLS + ODDS_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["GW"] = df[[f"W{i}" for i in range(1, 6)]].sum(axis=1)
    df["GL"] = df[[f"L{i}" for i in range(1, 6)]].sum(axis=1)
    return df.sort_values("Date", kind="stable").reset_index(drop=True)


# ---------------------------------------------------------------- Elo
def _k(n):
    return 250.0 / ((n + 5) ** 0.4)


def _book_prob(r):
    for w, l in (("AvgW", "AvgL"), ("PSW", "PSL"), ("B365W", "B365L")):
        a, b = getattr(r, w), getattr(r, l)
        if pd.notna(a) and pd.notna(b) and a > 1 and b > 1:
            return (1 / a) / (1 / a + 1 / b)
    return None


def build_ratings(df, record_last=0):
    """Elo generale + per superficie (rating usato = media dei due). Opzionale: registra le
    previsioni sulle ultime `record_last` partite per il backtest."""
    ov, sf, n, ns, hist, preds = {}, {}, {}, {}, {}, []
    start = len(df) - record_last if record_last else len(df) + 1
    for i, r in enumerate(df.itertuples(index=False)):
        w, l, s = r.Winner, r.Loser, r.Surface
        rw = 0.5 * ov.get(w, 1500.0) + 0.5 * sf.get((w, s), 1500.0)
        rl = 0.5 * ov.get(l, 1500.0) + 0.5 * sf.get((l, s), 1500.0)
        p = 1 / (1 + 10 ** (-(rw - rl) / 400))
        if i >= start and n.get(w, 0) >= 10 and n.get(l, 0) >= 10:
            preds.append((p, _book_prob(r)))
        e = 1 - p
        ov[w] = ov.get(w, 1500.0) + _k(n.get(w, 0)) * e
        ov[l] = ov.get(l, 1500.0) - _k(n.get(l, 0)) * e
        sf[(w, s)] = sf.get((w, s), 1500.0) + _k(ns.get((w, s), 0)) * e
        sf[(l, s)] = sf.get((l, s), 1500.0) - _k(ns.get((l, s), 0)) * e
        for pl in (w, l):
            n[pl] = n.get(pl, 0) + 1
            ns[(pl, s)] = ns.get((pl, s), 0) + 1
        g = float(r.GW + r.GL)
        retired = r.Comment.lower() == "retired"
        hist.setdefault(w, []).append((r.Date, g, False))
        hist.setdefault(l, []).append((r.Date, g, retired))
    return dict(ov=ov, sf=sf, n=n, hist=hist), preds


def active_players(model, df, min_matches=15, days=365):
    cut = df.Date.max() - pd.Timedelta(days=days)
    ps = [p for p, h in model["hist"].items() if h[-1][0] >= cut and model["n"][p] >= min_matches]
    return sorted(ps, key=lambda p: -model["ov"][p])


def player_info(model, p, today):
    h = model["hist"].get(p, [])
    if not h:
        return dict(matches=0, days=None, games7=0, retired=False)
    return dict(matches=model["n"][p], days=(today - h[-1][0]).days, retired=h[-1][2],
                games7=int(sum(g for d, g, _ in h if 0 <= (today - d).days <= 7)))


def fatigue_pen(games7):
    """Euristica: -1 Elo per game oltre 45 giocati negli ultimi 7 giorni (max -40)."""
    return float(min(40, max(0, games7 - 45)))


def rating(model, p, surface, pen=0.0):
    return 0.5 * model["ov"].get(p, 1500.0) + 0.5 * model["sf"].get((p, surface), 1500.0) - pen


# ---------------------------------------------------------------- set / match
def set_dist(hA, hB):
    """Distribuzione dei punteggi di un set. hA/hB = prob. di tenere il proprio servizio."""
    tb = 0.5 + (hA - hB) / 2  # prob. che A vinca il tie-break
    res = {}
    for startA in (True, False):
        states = {(0, 0): 0.5}
        for total in range(13):
            for st in [k for k in states if k[0] + k[1] == total]:
                p = states.pop(st)
                a, b = st
                if (a >= 6 and a - b >= 2) or (b >= 6 and b - a >= 2):
                    res[(a, b)] = res.get((a, b), 0) + p
                    continue
                if st == (6, 6):
                    res[(7, 6)] = res.get((7, 6), 0) + p * tb
                    res[(6, 7)] = res.get((6, 7), 0) + p * (1 - tb)
                    continue
                serverA = (total % 2 == 0) == startA
                pw = hA if serverA else 1 - hB
                states[(a + 1, b)] = states.get((a + 1, b), 0) + p * pw
                states[(a, b + 1)] = states.get((a, b + 1), 0) + p * (1 - pw)
    return res


def set_win(hA, hB):
    return sum(p for (a, b), p in set_dist(hA, hB).items() if a > b)


def avg_games(h):
    return sum(p * (a + b) for (a, b), p in set_dist(h, h).items())


def calibrate_hold(target):
    """Prob. di tenere il servizio media tale che i game per set simulati = quelli osservati."""
    lo, hi = 0.5, 0.95
    for _ in range(25):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if avg_games(mid) < target else (lo, mid)
    return (lo + hi) / 2


def games_per_set(df):
    d = df[df.Comment.str.lower() == "completed"]
    out = {}
    for s, g in d.groupby("Surface"):
        tot = cnt = 0
        for i in range(1, 6):
            w, l = g[f"W{i}"], g[f"L{i}"]
            m = w.notna() & l.notna()
            tot += (w[m] + l[m]).sum()
            cnt += m.sum()
        out[s] = tot / cnt if cnt else 10.0
    return out


def _match_to_set_prob(pm, best_of):
    f = (lambda x: x * x * (3 - 2 * x)) if best_of == 3 else (lambda x: x ** 3 * (10 - 15 * x + 6 * x * x))
    lo, hi = 0.0, 1.0
    for _ in range(40):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if f(mid) < pm else (lo, mid)
    return (lo + hi) / 2


def _delta_for_set_prob(ps, h0):
    m = min(h0, 1 - h0) - 0.01
    lo, hi = -m, m
    for _ in range(25):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if set_win(h0 + mid, h0 - mid) < ps else (lo, mid)
    return (lo + hi) / 2


def match_dist(sd, best_of):
    """Distribuzione congiunta (game A, game B) per ogni risultato in set."""
    need, N = best_of // 2 + 1, 7 * best_of + 1
    states = {(0, 0): np.zeros((N, N))}
    states[(0, 0)][0, 0] = 1.0
    term = {}
    for played in range(best_of + 1):
        for a in range(played + 1):
            b = played - a
            if (a, b) not in states:
                continue
            arr = states[(a, b)]
            if a == need or b == need:
                term[(a, b)] = arr
                continue
            for (gA, gB), p in sd.items():
                nxt = (a + 1, b) if gA > gB else (a, b + 1)
                new = states.setdefault(nxt, np.zeros((N, N)))
                new[gA:, gB:] += p * arr[:N - gA, :N - gB]
    return term, N


def analyze(model, h0map, a, b, surface, best_of=3, penA=0.0, penB=0.0):
    ra, rb = rating(model, a, surface, penA), rating(model, b, surface, penB)
    pm = 1 / (1 + 10 ** (-(ra - rb) / 400))
    h0 = h0map.get(surface, 0.78)
    delta = _delta_for_set_prob(_match_to_set_prob(pm, best_of), h0)
    term, N = match_dist(set_dist(h0 + delta, h0 - delta), best_of)
    need = best_of // 2 + 1
    J = sum(term.values())
    I, K = np.indices((N, N))
    tot = np.bincount((I + K).ravel(), weights=J.ravel())
    diff = np.bincount((I - K + N).ravel(), weights=J.ravel())  # indice = diff + N
    return dict(pA=sum(v.sum() for (x, y), v in term.items() if x == need),
                scores={k: float(v.sum()) for k, v in term.items()},
                tot=tot, diff=diff, N=N, need=need,
                exp_games=float((tot * np.arange(len(tot))).sum()), rA=ra, rB=rb)


def market_rows(res, a, b, best_of):
    """Lista (mercato, probabilita') per tutte le giocate analizzate."""
    rows = []
    pA, need, sc = res["pA"], res["need"], res["scores"]
    rows += [(f"Vince {a}", pA), (f"Vince {b}", 1 - pA)]
    for (x, y), p in sorted(sc.items()):
        if x == need:
            rows.append((f"{a} vince {x}-{y}", p))
        else:
            rows.append((f"{b} vince {y}-{x}", p))
    rows.append((f"{a} vince almeno un set", 1 - sc.get((0, need), 0)))
    rows.append((f"{b} vince almeno un set", 1 - sc.get((need, 0), 0)))
    if best_of == 3:
        rows.append(("Il match va al terzo set", sc.get((2, 1), 0) + sc.get((1, 2), 0)))
    tot, base = res["tot"], np.floor(res["exp_games"]) + 0.5
    cdf = np.cumsum(tot)
    for L in np.arange(base - 3, base + 4):
        if L < 12:
            continue
        po = 1 - cdf[int(L)]
        rows += [(f"Over {L:g} game totali", po), (f"Under {L:g} game totali", 1 - po)]
    # handicap game sul favorito
    fav, und, sign = (a, b, 1) if pA >= 0.5 else (b, a, -1)
    d, N = res["diff"], res["N"]
    dcdf = np.cumsum(d)
    md = float(((np.arange(len(d)) - N) * d).sum()) * sign
    for h in np.arange(np.floor(md) + 0.5 - 2, np.floor(md) + 0.5 + 3):
        if h < 0.5:
            continue
        # P(game_fav - game_und > h)
        if sign == 1:
            pf = 1 - dcdf[int(h) + N]
        else:
            pf = dcdf[N - int(h) - 1]
        rows += [(f"{fav} handicap -{h:g} game", pf), (f"{und} handicap +{h:g} game", 1 - pf)]
    return rows


# ---------------------------------------------------------------- backtest
def backtest(df, n=800):
    _, preds = build_ratings(df, record_last=n)
    p = np.array([x[0] for x in preds])
    book = np.array([x[1] if x[1] is not None else np.nan for x in preds])
    fav = np.maximum(p, 1 - p)
    hit = p > 0.5  # il vincitore e' sempre "winner": hit = il favorito ha vinto
    d = pd.DataFrame({"p": fav, "hit": hit})
    d["fascia"] = pd.cut(d.p, [0.5, 0.6, 0.7, 0.8, 1.01], right=False).astype(str)
    calib = d.groupby("fascia").agg(n=("p", "size"), previsto=("p", "mean"), realizzato=("hit", "mean")).reset_index()
    ok = ~np.isnan(book)
    ll = dict(partite=len(p), elo=float(-np.mean(np.log(p))),
              bookmaker=float(-np.mean(np.log(book[ok]))) if ok.any() else None)
    return calib, ll
