"""
Motore NBA: rating di squadra (regressione ridge su margine e punti totali) + proiezioni dei
giocatori dai box score recenti. Dati: feed pubblico ESPN (nessuna chiave).
"""
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from scipy.stats import nbinom, norm, poisson

BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
STATS = {"Punti": "pts", "Rimbalzi": "reb", "Assist": "ast", "Triple": "tpm", "Punti+Rimb+Assist": "pra"}
_BOX = {}  # cache dei box score (le partite finite non cambiano)


# ---------------------------------------------------------------- rete e parsing
def _get(url, params=None, tries=3):
    import requests
    last = None
    for i in range(tries):
        try:
            r = requests.get(url, params=params, headers={"User-Agent": "Mozilla/5.0"}, timeout=40)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last = e
            time.sleep(0.8 * (i + 1))
    raise RuntimeError(f"{url}: {last}")


def _num(x):
    if isinstance(x, dict):
        x = x.get("value", x.get("displayValue"))
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def _ts(s):
    t = pd.Timestamp(s)
    return t.tz_convert("UTC").tz_localize(None) if t.tzinfo else t


def parse_events(js):
    rows = []
    for ev in js.get("events") or []:
        try:
            comp = (ev.get("competitions") or [{}])[0]
            typ = (comp.get("status") or ev.get("status") or {}).get("type") or {}
            cs = {c.get("homeAway"): c for c in comp.get("competitors") or []}
            h, a = cs["home"], cs["away"]
            rows.append(dict(
                id=str(ev["id"]), date=_ts(ev["date"]), stype=(ev.get("season") or {}).get("type"),
                done=bool(typ.get("completed")), state=typ.get("state", ""),
                hid=str(h["team"]["id"]), aid=str(a["team"]["id"]),
                hn=h["team"].get("displayName", ""), an=a["team"].get("displayName", ""),
                hs=_num(h.get("score")), as_=_num(a.get("score"))))
        except Exception:
            continue
    return rows


def _chunks(start, end, days=7):
    s = start
    while s <= end:
        e = min(s + pd.Timedelta(days=days - 1), end)
        yield f"{s:%Y%m%d}-{e:%Y%m%d}"
        s = e + pd.Timedelta(days=1)


def fetch_games():
    """Partite NBA della stagione corrente e della precedente (anche quelle in programma)."""
    today = pd.Timestamp.now("UTC").tz_localize(None).normalize()
    y = today.year if today.month >= 10 else today.year - 1
    rngs = (list(_chunks(pd.Timestamp(f"{y - 1}-10-01"), pd.Timestamp(f"{y}-07-01")))
            + list(_chunks(pd.Timestamp(f"{y}-10-01"), today + pd.Timedelta(days=3))))

    def one(r):
        try:
            return parse_events(_get(BASE + "/scoreboard", {"dates": r, "limit": 1000})), None
        except Exception as e:
            return [], str(e)

    with ThreadPoolExecutor(8) as ex:
        res = list(ex.map(one, rngs))
    rows = [x for r, _ in res for x in r]
    errs = [e for _, e in res if e]
    if not rows:
        raise RuntimeError("Nessuna partita scaricata. " + (errs[0][:200] if errs else "risposta vuota"))
    df = pd.DataFrame(rows).drop_duplicates("id")
    keep = pd.concat([df.hid, df.aid]).value_counts()
    keep = keep[keep >= 10].index  # esclude All-Star e squadre non NBA
    df = df[df.hid.isin(keep) & df.aid.isin(keep) & (df.stype != 1)]
    return df.sort_values("date").reset_index(drop=True)


def fetch_injuries(name_map=()):
    """Infortunati per squadra: {team_id: [{id, name, status, detail}]}."""
    names = dict(name_map)
    js = _get(BASE + "/injuries")
    out = {}
    for t in js.get("injuries") or []:
        tid = t.get("id") or (t.get("team") or {}).get("id") or names.get(t.get("displayName"))
        if tid is None:
            continue
        for i in t.get("injuries") or []:
            at = i.get("athlete") or {}
            out.setdefault(str(tid), []).append(dict(
                id=str(at.get("id")), name=at.get("displayName") or at.get("fullName") or "?",
                status=str(i.get("status") or (i.get("type") or {}).get("description") or ""),
                detail=str((i.get("details") or {}).get("detail") or i.get("shortComment") or "")[:120]))
    return out


# ---------------------------------------------------------------- squadre
def _ridge(X, y, w, lam, unpen):
    pen = np.full(X.shape[1], lam)
    pen[unpen] = 0
    Xw = X * w[:, None]
    return np.linalg.solve(Xw.T @ X + np.diag(pen), Xw.T @ y)


def fit_team(df, ref, halflife=150, lam=15.0):
    """Margine = vantaggio campo + rating casa - rating ospite; totale = media lega + effetto squadre."""
    d = df[df.done].dropna(subset=["hs", "as_"])
    teams = sorted(set(d.hid) | set(d.aid))
    idx = {t: i for i, t in enumerate(teams)}
    n, N = len(teams), len(d)
    h, a, r = d.hid.map(idx).values, d.aid.map(idx).values, np.arange(N)
    w = 0.5 ** ((ref - d.date).dt.days.clip(lower=0).values / halflife)
    Xm = np.zeros((N, n + 1))
    Xm[r, h], Xm[r, a], Xm[:, n] = 1, -1, 1
    Xt = np.zeros((N, n + 1))
    Xt[r, h] += 1
    Xt[r, a] += 1
    Xt[:, n] = 1
    ym, yt = (d.hs - d.as_).values, (d.hs + d.as_).values
    bm, bt = _ridge(Xm, ym, w, lam, [n]), _ridge(Xt, yt, w, lam, [n])
    sm = float(np.sqrt((w * (ym - Xm @ bm) ** 2).sum() / w.sum())) * 1.02
    st = float(np.sqrt((w * (yt - Xt @ bt) ** 2).sum() / w.sum())) * 1.02
    return dict(idx=idx, r=bm[:n], ha=bm[n], t=bt[:n], c=bt[n], sm=sm, st=st, sp=float(np.sqrt((sm ** 2 + st ** 2) / 4)))


def project(m, h, a, lossH=0.0, lossA=0.0):
    """(margine atteso casa-ospite, punti totali attesi). loss = punti persi per assenze/stanchezza."""
    i, j = m["idx"][h], m["idx"][a]
    marg = m["ha"] + m["r"][i] - m["r"][j] - (lossH - lossA)
    tot = m["c"] + m["t"][i] + m["t"][j] - (lossH + lossA)
    return float(marg), float(tot)


def team_rows(m, hn, an, marg, tot):
    rows = []

    def add(fam, label, p):
        rows.append(dict(fam=fam, label=label, p=float(p)))

    ph = norm.cdf(marg / m["sm"])
    add("Vincente", f"Vince {hn}", ph)
    add("Vincente", f"Vince {an}", 1 - ph)
    for nm, mg in ((hn, marg), (an, -marg)):
        for s in np.arange(np.floor(-mg) - 3.5, np.floor(-mg) + 4.5, 1.0):
            add("Handicap", f"Handicap {nm} {s:+g}", norm.cdf((mg + s) / m["sm"]))
    base = np.floor(tot) + 0.5
    for L in np.arange(base - 6, base + 7):
        po = 1 - norm.cdf((L - tot) / m["st"])
        add("Totale", f"Over {L:g} punti totali", po)
        add("Totale", f"Under {L:g} punti totali", 1 - po)
    for nm, mu in ((hn, (tot + marg) / 2), (an, (tot - marg) / 2)):
        b = np.floor(mu) + 0.5
        for L in np.arange(b - 4, b + 5):
            po = 1 - norm.cdf((L - mu) / m["sp"])
            add("Totale squadra", f"{nm} Over {L:g} punti", po)
            add("Totale squadra", f"{nm} Under {L:g} punti", 1 - po)
    return rows


# ---------------------------------------------------------------- giocatori
def parse_box(js):
    """{team_id: [giocatori con minuti, punti, rimbalzi, assist, triple]} da un box score ESPN."""
    out = {}
    for t in (js.get("boxscore") or {}).get("players") or []:
        tid = str((t.get("team") or {}).get("id"))
        lst = []
        for grp in t.get("statistics") or []:
            labels = [str(x).upper() for x in (grp.get("labels") or [])]
            for at in grp.get("athletes") or []:
                st = at.get("stats") or []
                if at.get("didNotPlay") or not st or len(st) < len(labels):
                    continue
                d = dict(zip(labels, st))
                a = at.get("athlete") or {}
                mn = _num(str(d.get("MIN", "0")).split(":")[0])
                pts, reb, ast = _num(d.get("PTS")), _num(d.get("REB")), _num(d.get("AST"))
                tp = _num(str(d.get("3PT", "0-0")).split("-")[0])
                if not mn or np.isnan(mn) or mn <= 0 or np.isnan(pts + reb + ast):
                    continue
                lst.append(dict(id=str(a.get("id")), name=a.get("displayName") or a.get("shortName") or "?",
                                min=mn, pts=pts, reb=reb, ast=ast, tpm=0.0 if np.isnan(tp) else tp,
                                pra=pts + reb + ast))
        out[tid] = lst
    return out


def get_box(eid):
    if eid in _BOX:
        return _BOX[eid]
    try:
        b = parse_box(_get(BASE + "/summary", {"event": eid}))
    except Exception:
        return None
    _BOX[eid] = b
    return b


def player_logs(games, tid, n=15):
    """Ultime n partite finite della squadra: ({id: {name, games[piu' recente prima]}}, punti medi squadra)."""
    d = games[games.done & ((games.hid == tid) | (games.aid == tid))].sort_values("date", ascending=False).head(n)
    with ThreadPoolExecutor(8) as ex:
        boxes = list(ex.map(get_box, list(d.id)))
    logs, pts = {}, []
    for (_, g), b in zip(d.iterrows(), boxes):
        if not b:
            continue
        pts.append(g.hs if g.hid == tid else g.as_)
        for p in b.get(tid, []):
            logs.setdefault(p["id"], {"name": p["name"], "games": []})["games"].append(p)
    return logs, (float(np.mean(pts)) if pts else float("nan"))


def rotation(logs, min_min=12):
    """Giocatori di rotazione con medie delle ultime 10 partite."""
    out = []
    for pid, v in logs.items():
        g = v["games"]
        if len(g) < 5:
            continue
        g10 = g[:10]
        mn = float(np.mean([x["min"] for x in g10]))
        if mn < min_min:
            continue
        out.append(dict(id=pid, name=v["name"], min=mn, n=len(g),
                        **{k: float(np.mean([x[k] for x in g10])) for k in ("pts", "reb", "ast", "tpm")}))
    return sorted(out, key=lambda x: -x["min"])


def _pmf(mu, var):
    K = int(mu + 6 * np.sqrt(max(var, 1)) + 12)
    k = np.arange(K + 1)
    if var <= mu * 1.0001:
        v = poisson.pmf(k, mu)
    else:
        r = mu * mu / (var - mu)
        v = nbinom.pmf(k, r, r / (r + mu))
    return v / v.sum()


def player_rows(logs, team, scale=1.0, boost=1.0, absent=()):
    """Probabilita' Over/Under per giocatore e statistica (linee .5 attorno alla media)."""
    rows = []
    for pid, v in logs.items():
        g = v["games"]
        if len(g) < 5 or pid in absent or np.mean([x["min"] for x in g[:10]]) < 12:
            continue
        w = 0.92 ** np.arange(len(g))
        for lab, key in STATS.items():
            x = np.array([q[key] for q in g], float)
            mu = np.average(x, weights=w)
            var = np.average((x - mu) ** 2, weights=w) * len(g) / (len(g) - 1)
            f = (scale ** 0.5 if key == "reb" else scale) * boost
            mu2, var2 = mu * f, var * f * f
            if mu2 < 0.4:
                continue
            cdf = np.cumsum(_pmf(mu2, max(var2, mu2)))
            base = np.floor(mu2) + 0.5
            for L in np.arange(base - 2, base + 3):
                if L < 0.5:
                    continue
                po = 1 - cdf[int(L)]
                for side, p in (("Over", po), ("Under", 1 - po)):
                    rows.append(dict(team=team, player=v["name"], stat=lab, side=side, line=float(L),
                                     label=f"{v['name']} · {lab} {side} {L:g}", p=float(p), mean=float(mu2)))
    return rows


# ---------------------------------------------------------------- backtest
def backtest(df, n_test=300, step=30):
    d = df[df.done].dropna(subset=["hs", "as_"]).sort_values("date").reset_index(drop=True)
    rows = []
    for s in range(max(60, len(d) - n_test), len(d), step):
        tr, te = d.iloc[:s], d.iloc[s:s + step]
        m = fit_team(tr, tr.date.max() + pd.Timedelta(days=1))
        for g in te.itertuples():
            if g.hid not in m["idx"] or g.aid not in m["idx"]:
                continue
            mg, tot = project(m, g.hid, g.aid)
            am, at = g.hs - g.as_, g.hs + g.as_
            ph = norm.cdf(mg / m["sm"])
            rows += [("Vincente", ph, am > 0), ("Vincente", 1 - ph, am < 0)]
            for L in np.arange(np.floor(tot) + 0.5 - 6, np.floor(tot) + 0.5 + 7, 2):
                po = 1 - norm.cdf((L - tot) / m["st"])
                rows += [("Totale", po, at > L), ("Totale", 1 - po, at < L)]
            for sl in np.arange(np.floor(-mg) - 3.5, np.floor(-mg) + 4.5, 2):
                pc = norm.cdf((mg + sl) / m["sm"])
                rows += [("Handicap", pc, am + sl > 0), ("Handicap", 1 - pc, am + sl < 0)]
    r = pd.DataFrame(rows, columns=["mercato", "p", "hit"])
    r = r[r.p >= 0.5].copy()
    r["fascia"] = pd.cut(r.p, [0.5, 0.6, 0.7, 0.8, 1.01], right=False).astype(str)
    calib = r.groupby("fascia").agg(n=("p", "size"), previsto=("p", "mean"), realizzato=("hit", "mean")).reset_index()
    per = r[r.p >= 0.6].groupby("mercato").agg(n=("p", "size"), previsto=("p", "mean"), realizzato=("hit", "mean")).reset_index()
    return calib, per
