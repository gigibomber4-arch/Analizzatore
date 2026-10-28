#!/usr/bin/env python3
"""
Analizzatore multi-mercato per i 5 grandi campionati: gol, 1T/2T, corner, tiri, tiri in porta,
falli, cartellini (con effetto arbitro) e confronto con le quote.

Uso:
  pip install pandas numpy scipy requests
  python analizzatore_seriea.py                       # Serie A, prossima giornata
  python analizzatore_seriea.py --lega E0             # Premier (I1 Serie A, SP1 La Liga, D1 Bundesliga, F1 Ligue 1)
  python analizzatore_seriea.py --lega all --key LA_TUA_CHIAVE_ODDSPAPI
  python analizzatore_seriea.py --lega SP1 --backtest # verifica l'affidabilita' del modello
  python analizzatore_seriea.py --csv-dir dati        # CSV locali (I1_2627.csv, E0_2627.csv, ...)

Dati: football-data.co.uk (gratuiti). Quote: OddsPapi (opzionale). Modello: Poisson / Binomiale
Negativa con forza attacco-difesa per squadra, vantaggio campo, peso alle partite recenti.
"""
import argparse
import os
import re
import sys
import time
from difflib import SequenceMatcher

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import nbinom, poisson

BASE = "https://www.football-data.co.uk"
SEASONS = ["2627", "2526", "2425"]  # stagione corrente + 2 precedenti

# codice football-data -> (nome, categoria OddsPapi, nomi torneo OddsPapi)
LEAGUES = {
    "I1": ("Serie A", "italy", {"serie a"}),
    "E0": ("Premier League", "england", {"premier league"}),
    "SP1": ("La Liga", "spain", {"laliga", "la liga"}),
    "D1": ("Bundesliga", "germany", {"bundesliga"}),
    "F1": ("Ligue 1", "france", {"ligue 1"}),
}
RAW_COLS = ["FTHG", "FTAG", "HTHG", "HTAG", "HS", "AS", "HST", "AST", "HF", "AF", "HC", "AC", "HY", "AY"]

# statistica -> (colonna casa, colonna ospite)
STATS = {
    "Gol": ("FTHG", "FTAG"),
    "Gol 1T": ("HTHG", "HTAG"),
    "Gol 2T": ("H2G", "A2G"),
    "Corner": ("HC", "AC"),
    "Tiri": ("HS", "AS"),
    "Tiri in porta": ("HST", "AST"),
    "Falli": ("HF", "AF"),
    "Cartellini": ("HY", "AY"),
}

REF_STATS = ("Cartellini", "Falli")  # statistiche su cui conta l'arbitro

# quote disponibili nel file fixtures (altri mercati: confronta tu con la quota equa)
ODDS_COLS = {
    ("Gol", "1"): ["AvgH", "B365H"],
    ("Gol", "X"): ["AvgD", "B365D"],
    ("Gol", "2"): ["AvgA", "B365A"],
    ("Gol", "Over 2.5"): ["Avg>2.5", "B365>2.5"],
    ("Gol", "Under 2.5"): ["Avg<2.5", "B365<2.5"],
}


# ---------------------------------------------------------------- dati
def read_csv_clean(src):
    """Legge un CSV di football-data togliendo il BOM dai nomi colonna e righe/colonne vuote."""
    d = pd.read_csv(src, encoding="latin-1")
    d.columns = [re.sub(r"^(\u00ef\u00bb\u00bf|\ufeff)", "", str(c)).strip() for c in d.columns]
    return d.dropna(how="all")


def load_history(csv_dir=None, divs=None):
    frames = []
    for div in (divs or list(LEAGUES)):
        for se in SEASONS:
            src = os.path.join(csv_dir, f"{div}_{se}.csv") if csv_dir else f"{BASE}/mmz4281/{se}/{div}.csv"
            try:
                d = read_csv_clean(src)
                d = d.reindex(columns=["Date", "HomeTeam", "AwayTeam", "Referee"] + RAW_COLS)
                d["Div"] = div
                frames.append(d)
            except Exception as e:
                print(f"! {div} {se} non caricata ({e})")
    if not frames:
        sys.exit("Nessun dato caricato. Controlla la connessione o --csv-dir.")
    df = pd.concat(frames, ignore_index=True)
    for c in RAW_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, format="mixed", errors="coerce")
    df = df.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"])
    df["H2G"] = df["FTHG"] - df["HTHG"]
    df["A2G"] = df["FTAG"] - df["HTAG"]
    return df.sort_values("Date").reset_index(drop=True)


def load_fixtures(path=None, div="I1"):
    """Prossima giornata del campionato: partite entro 3 giorni dalla prima in programma."""
    fx = read_csv_clean(path or f"{BASE}/fixtures.csv")
    if "Div" not in fx.columns:
        raise RuntimeError(f"File calendario non valido: colonne trovate {list(fx.columns)[:8]}")
    fx = fx[fx["Div"] == div].copy()
    fx["Date"] = pd.to_datetime(fx["Date"], dayfirst=True, format="mixed", errors="coerce")
    fx = fx[fx["Date"] >= pd.Timestamp.today().normalize()].sort_values("Date")
    if fx.empty:
        return fx
    return fx[fx["Date"] <= fx["Date"].min() + pd.Timedelta(days=3)].head(10)


# ---------------------------------------------------------------- modello
def fit_stat(df, hc, ac, ref, xi=0.002, lam=3.0, use_ref=False, lam_ref=4.0):
    """Poisson log-lineare: log mu = c + casa + att[squadra] - dif[avversario].
    Pesi con decadimento temporale (xi) e penalita' ridge (lam) contro l'overfitting."""
    d = df.dropna(subset=[hc, ac])
    teams = sorted(set(d.HomeTeam) | set(d.AwayTeam))
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    h = d.HomeTeam.map(idx).values
    a = d.AwayTeam.map(idx).values
    yh = d[hc].values.astype(float)
    ya = d[ac].values.astype(float)
    w = np.exp(-xi * (ref - d.Date).dt.days.clip(lower=0).values)

    # effetto arbitro (ridge forte: con pochi match l'effetto resta vicino a zero)
    if use_ref and "Referee" in d.columns and d.Referee.notna().any():
        refs = sorted(d.Referee.dropna().unique())
        ridx = {r: i for i, r in enumerate(refs)}
    else:
        refs, ridx = [], {}
    R = len(refs)
    rr = d["Referee"].map(ridx).fillna(R).astype(int).values if R else np.zeros(len(d), int)  # R = "sconosciuto"

    def f(p):
        att, dif, ha, c, rf = p[:n], p[n:2 * n], p[2 * n], p[2 * n + 1], p[2 * n + 2:]
        lr = rf[rr]
        lh = c + ha + att[h] - dif[a] + lr
        la = c + att[a] - dif[h] + lr
        eh, ea = np.exp(lh), np.exp(la)
        nll = -(w * (yh * lh - eh)).sum() - (w * (ya * la - ea)).sum()
        nll += lam * ((att ** 2).sum() + (dif ** 2).sum()) + lam_ref * (rf ** 2).sum()
        gh, ga = w * (yh - eh), w * (ya - ea)
        g = np.zeros_like(p)
        g[:n] = -(np.bincount(h, gh, n) + np.bincount(a, ga, n)) + 2 * lam * att
        g[n:2 * n] = (np.bincount(a, gh, n) + np.bincount(h, ga, n)) + 2 * lam * dif
        g[2 * n] = -gh.sum()
        g[2 * n + 1] = -(gh.sum() + ga.sum())
        g[2 * n + 2:] = -np.bincount(rr, gh + ga, R + 1) + 2 * lam_ref * rf
        g[-1] = 0.0  # il posto 'arbitro sconosciuto' resta a zero
        return nll, g

    p0 = np.zeros(2 * n + 2 + R + 1)
    p0[2 * n + 1] = np.log(max(np.average(np.r_[yh, ya], weights=np.r_[w, w]), 0.1))
    p = minimize(f, p0, jac=True, method="L-BFGS-B").x
    rf = p[2 * n + 2:]
    m = dict(idx=idx, att=p[:n], dif=p[n:2 * n], ha=p[2 * n], c=p[2 * n + 1], ref={r: rf[i] for r, i in ridx.items()})

    # sovradispersione (corner, falli, cartellini variano piu' di una Poisson)
    lr = rf[rr]
    mu = np.r_[np.exp(m["c"] + m["ha"] + m["att"][h] - m["dif"][a] + lr),
               np.exp(m["c"] + m["att"][a] - m["dif"][h] + lr)]
    y, ww = np.r_[yh, ya], np.r_[w, w]
    m["alpha"] = max((ww * ((y - mu) ** 2 - mu)).sum() / (ww * mu ** 2).sum(), 0.0)
    return m


def mus(m, home, away, referee=None):
    i, j = m["idx"][home], m["idx"][away]
    r = m["ref"].get(referee, 0.0) if referee else 0.0
    return (np.exp(m["c"] + m["ha"] + m["att"][i] - m["dif"][j] + r),
            np.exp(m["c"] + m["att"][j] - m["dif"][i] + r))


def referee_effect(models, referee):
    """Fattore moltiplicativo dell'arbitro rispetto alla media (es. 1.12 = +12%)."""
    return {s: float(np.exp(m["ref"][referee])) for s, m in models.items() if referee in m["ref"]}


def pmf(mu, alpha, K):
    k = np.arange(K + 1)
    if alpha < 1e-3:
        v = poisson.pmf(k, mu)
    else:
        r = 1 / alpha
        v = nbinom.pmf(k, r, r / (r + mu))
    return v / v.sum()


def cmp(x, y):
    """P(X>Y), P(X=Y), P(X<Y) per due pmf indipendenti."""
    J = np.outer(x, y)
    return np.tril(J, -1).sum(), np.trace(J), np.triu(J, 1).sum()


# ---------------------------------------------------------------- picks
def build_picks(models, home, away, referee=None):
    picks, exp, dist = [], {}, {}

    def add(stat, scope, label, p):
        picks.append(dict(stat=stat, scope=scope, label=label, p=float(p)))

    for stat, m in models.items():
        if home not in m["idx"] or away not in m["idx"]:
            continue
        mh, ma = mus(m, home, away, referee)
        K = int(4 * (mh + ma) + 15)
        ph, pa = pmf(mh, m["alpha"], K), pmf(ma, m["alpha"], K)
        dist[stat] = (ph, pa)
        exp[stat] = (mh, ma)
        tot = np.convolve(ph, pa)

        # over/under: totale e per squadra, linee attorno al valore atteso
        for scope, d, mu, width in (("Totale", tot, mh + ma, 2), ("Casa", ph, mh, 1), ("Ospite", pa, ma, 1)):
            b = np.floor(mu) + 0.5
            for L in np.arange(b - width, b + width + 1):
                if L < 0.5:
                    continue
                po = 1 - d[:int(L) + 1].sum()
                pre = "" if scope == "Totale" else scope + " "
                add(stat, scope, f"{pre}Over {L:g}", po)
                add(stat, scope, f"{pre}Under {L:g}", 1 - po)

        # chi ne fa di piu'
        pw, px, pl = cmp(ph, pa)
        if stat == "Gol":
            for lab, p in (("1", pw), ("X", px), ("2", pl), ("1X", pw + px), ("X2", px + pl), ("12", pw + pl)):
                add(stat, "Esito", lab, p)
            add(stat, "BTTS", "Gol (entrambe segnano)", (1 - ph[0]) * (1 - pa[0]))
            add(stat, "BTTS", "No Gol", 1 - (1 - ph[0]) * (1 - pa[0]))
        elif stat in ("Gol 1T", "Gol 2T"):
            t = stat[-2:]
            for lab, p in ((f"{t}: 1", pw), (f"{t}: X", px), (f"{t}: 2", pl)):
                add(stat, "Esito", lab, p)
        else:
            add(stat, "Esito", f"Piu' {stat.lower()}: Casa", pw)
            add(stat, "Esito", f"Piu' {stat.lower()}: Ospite", pl)

    # quale tempo avra' piu' gol
    if "Gol 1T" in dist and "Gol 2T" in dist:
        t1, t2 = (np.convolve(*dist[s]) for s in ("Gol 1T", "Gol 2T"))
        p2, _, p1 = cmp(t2, t1)
        add("Gol", "Tempo", "Piu' gol nel 2T", p2)
        add("Gol", "Tempo", "Piu' gol nel 1T", p1)
    return picks, exp


def select(picks, min_prob, min_fair, top):
    """Per ogni famiglia di mercato tiene la linea migliore, poi ordina per probabilita'."""
    best = {}
    for k in picks:
        if k["p"] < min_prob or 1 / k["p"] < min_fair:
            continue
        key = (k["stat"], k["scope"])
        if key not in best or k["p"] > best[key]["p"]:
            best[key] = k
    return sorted(best.values(), key=lambda x: -x["p"])[:top]


# ---------------------------------------------------------------- quote automatiche (OddsPapi)
OP_BASE = "https://api.oddspapi.io/v4"
OP_MARKETS = {  # nome mercato OddsPapi (normalizzato) -> (statistica, ambito)
    "over under full time": ("Gol", "Totale"),
    "over under team 1": ("Gol", "Casa"),
    "over under team 2": ("Gol", "Ospite"),
    "over under first half": ("Gol 1T", "Totale"),
    "over under second half": ("Gol 2T", "Totale"),
    "corners - over under full time": ("Corner", "Totale"),
    "corners - over under team 1": ("Corner", "Casa"),
    "corners - over under team 2": ("Corner", "Ospite"),
    "bookings - over under full time": ("Cartellini", "Totale"),
    "bookings - over under team 1": ("Cartellini", "Casa"),
    "bookings - over under team 2": ("Cartellini", "Ospite"),
}


def _norm_market(n):
    return re.sub(r"\s+", " ", re.sub(r"[\u2013\u2014]", "-", n)).strip().lower()


def op_get(path, key, **params):
    import requests
    params["apiKey"] = key
    for _ in range(5):
        r = requests.get(f"{OP_BASE}{path}", params=params, timeout=120)
        if r.status_code == 429:
            try:
                wait = r.json().get("error", {}).get("retryMs", 1500) / 1000
            except Exception:
                wait = 1.5
            time.sleep(wait + 0.3)
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError("OddsPapi: troppe richieste, riprova tra poco")


def parse_odds(payload, cat, bookmaker):
    """Trasforma la risposta /odds di una partita in {(statistica, etichetta): quota}."""
    out = {}
    book = (payload.get("bookmakerOdds") or {}).get(bookmaker) or {}
    for mid, mk in (book.get("markets") or {}).items():
        info = cat.get(int(mid))
        if not info or mk.get("marketActive") is False:
            continue
        name, line = _norm_market(info["marketName"]), info.get("handicap")
        names = {o["outcomeId"]: o["outcomeName"] for o in info.get("outcomes", [])}
        for oid, oc in (mk.get("outcomes") or {}).items():
            pr = (oc.get("players") or {}).get("0")
            if not isinstance(pr, dict) or not pr.get("price") or pr.get("active") is False:
                continue
            side, price = names.get(int(oid)), float(pr["price"])
            if name == "full time result" and side in ("1", "X", "2"):
                out[("Gol", side)] = price
            elif name == "both teams to score" and side in ("Yes", "No"):
                out[("Gol", "Gol (entrambe segnano)" if side == "Yes" else "No Gol")] = price
            elif name in OP_MARKETS and side in ("Over", "Under") and line is not None:
                stat, scope = OP_MARKETS[name]
                line = float(line)
                if abs(line % 1 - 0.5) > 1e-6 or (stat == "Cartellini" and line > 12.5):
                    continue  # solo linee .5; oltre 12.5 sono "punti ammonizione", non cartellini
                pre = "" if scope == "Totale" else scope + " "
                out[(stat, f"{pre}{side} {line:g}")] = price
    return out


_CAT = {}  # catalogo mercati (pesante): scaricato una sola volta


def fetch_odds(key, bookmaker="pinnacle", div="I1", n=10):
    """Scarica le quote della prossima giornata del campionato. Ritorna lista di dict."""
    _, cat_slug, names = LEAGUES[div]
    tours = op_get("/tournaments", key, sportId=10)
    t = [x for x in tours if x.get("categorySlug") == cat_slug and str(x.get("tournamentName", "")).lower() in names]
    if not t:
        raise RuntimeError(f"{LEAGUES[div][0]} non trovato su OddsPapi")
    time.sleep(2.1)
    fx = op_get("/fixtures", key, tournamentId=t[0]["tournamentId"], statusId=0, hasOdds="true", bookmakers=bookmaker)
    fx = sorted(fx, key=lambda x: x["startTime"])
    if fx:
        first = pd.Timestamp(fx[0]["startTime"])
        fx = [f for f in fx if pd.Timestamp(f["startTime"]) <= first + pd.Timedelta(days=3.5)][:n]
    if not _CAT:
        time.sleep(1.1)
        _CAT.update({m["marketId"]: m for m in op_get("/markets", key)})
    out = []
    for f in fx:
        time.sleep(0.6)
        payload = op_get("/odds", key, fixtureId=f["fixtureId"], bookmakers=bookmaker)
        out.append(dict(start=pd.Timestamp(f["startTime"]).tz_localize(None), home=f["participant1Name"],
                        away=f["participant2Name"], odds=parse_odds(payload, _CAT, bookmaker)))
    return out


_DROP = {"fc", "ac", "as", "ss", "ssc", "us", "calcio", "acf", "bc", "1913", "1907", "1928"}


def _tokens(n):
    return {t for t in re.sub(r"[^a-z0-9 ]", " ", n.lower()).split() if t not in _DROP}


def _team_score(a, b):
    ta, tb = _tokens(a), _tokens(b)
    if ta == tb:
        return 1.0
    if ta and tb and (ta <= tb or tb <= ta):
        return 0.85
    if any(x.startswith(y) or y.startswith(x) for x in ta for y in tb if min(len(x), len(y)) >= 4):
        return 0.8  # es. 'inter' / 'internazionale'
    return SequenceMatcher(None, " ".join(sorted(ta)), " ".join(sorted(tb))).ratio()


def match_odds(fx, op_list):
    """Abbina ogni partita del fixtures alle quote OddsPapi (nome squadre + data). {indice: quote}."""
    res = {}
    for i, r in fx.iterrows():
        best, bs = None, 0
        for o in op_list:
            if abs((o["start"].normalize() - r.Date).days) > 1:
                continue
            sc = _team_score(r.HomeTeam, o["home"]) + _team_score(r.AwayTeam, o["away"])
            if sc > bs:
                best, bs = o, sc
        if best and bs >= 1.5:
            res[i] = best["odds"]
    return res


def get_odds(row, cols):
    for c in cols:
        if c in row.index and pd.notna(row[c]):
            return float(row[c])
    return None


def csv_odds(row):
    """Quote 1X2 e Over/Under 2.5 dal file fixtures di football-data."""
    out = {}
    for k, cols in ODDS_COLS.items():
        v = get_odds(row, cols)
        if v:
            out[k] = v
    return out


def analyze(models, home, away, referee=None, odds=None):
    """Tutte le probabilita' della partita, con quota e edge dove disponibili."""
    picks, exp = build_picks(models, home, away, referee)
    odds = odds or {}
    for k in picks:
        k["odds"] = odds.get((k["stat"], k["label"]))
        k["edge"] = k["p"] * k["odds"] - 1 if k["odds"] else None
    return picks, exp


# ---------------------------------------------------------------- backtest
def backtest_tables(df, n_test=150, step=10):
    """Walk-forward: addestra solo sul passato, prevede le partite successive e confronta."""
    rows = []
    for s in range(len(df) - n_test, len(df), step):
        tr, te = df.iloc[:s], df.iloc[s:s + step]
        ref = tr.Date.max() + pd.Timedelta(days=1)
        for stat, (hc, ac) in STATS.items():
            m = fit_stat(tr, hc, ac, ref, use_ref=stat in REF_STATS)
            for _, r in te.iterrows():
                if pd.isna(r[hc]) or pd.isna(r[ac]) or r.HomeTeam not in m["idx"] or r.AwayTeam not in m["idx"]:
                    continue
                rf = r.get("Referee")
                mh, ma = mus(m, r.HomeTeam, r.AwayTeam, None if pd.isna(rf) else rf)
                K = int(4 * (mh + ma) + 15)
                tot = np.convolve(pmf(mh, m["alpha"], K), pmf(ma, m["alpha"], K))
                y = r[hc] + r[ac]
                b = np.floor(mh + ma) + 0.5
                for L in np.arange(b - 2, b + 3):
                    if L < 0.5:
                        continue
                    po = 1 - tot[:int(L) + 1].sum()
                    rows.append((stat, po, y > L))
                    rows.append((stat, 1 - po, y < L))
    r = pd.DataFrame(rows, columns=["stat", "p", "hit"])
    r = r[r.p >= 0.5].copy()
    r["fascia"] = pd.cut(r.p, [0.5, 0.6, 0.7, 0.8, 1.01], right=False).astype(str)
    calib = r.groupby("fascia").agg(n=("p", "size"), previsto=("p", "mean"), realizzato=("hit", "mean")).reset_index()
    per = r[r.p >= 0.6].groupby("stat").agg(n=("p", "size"), previsto=("p", "mean"), realizzato=("hit", "mean")).reset_index()
    return calib, per


def backtest(df):
    calib, per = backtest_tables(df)
    print("\nBACKTEST (over/under sul totale, tutte le statistiche)\n")
    print("Fascia prob.      n   previsto  realizzato")
    for _, x in calib.iterrows():
        print(f"{x.fascia:14s} {x.n:5d}   {x.previsto:.1%}     {x.realizzato:.1%}")
    print("\nPer statistica (solo picks con prob >= 60%):")
    for _, x in per.iterrows():
        print(f"{x.stat:15s} {x.n:5d}   {x.previsto:.1%}     {x.realizzato:.1%}")
    print("\nSe 'realizzato' e' molto piu' basso di 'previsto', quel mercato non e' affidabile: ignoralo.")


# ---------------------------------------------------------------- dati/modelli condivisi
def fit_all(df, div="I1"):
    d = df[df["Div"] == div]
    ref = pd.Timestamp.today().normalize()
    return {k: fit_stat(d, *v, ref, use_ref=k in REF_STATS) for k, v in STATS.items()}


# ---------------------------------------------------------------- CLI
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lega", default="I1", help="I1 Serie A, E0 Premier, SP1 La Liga, D1 Bundesliga, F1 Ligue 1, oppure 'all'")
    ap.add_argument("--min-prob", type=float, default=0.60, help="probabilita' minima (default 0.60)")
    ap.add_argument("--min-fair", type=float, default=1.25, help="quota equa minima, scarta picks banali (default 1.25)")
    ap.add_argument("--top", type=int, default=6, help="picks per partita")
    ap.add_argument("--backtest", action="store_true")
    ap.add_argument("--csv-dir")
    ap.add_argument("--fixtures", help="percorso a un fixtures.csv locale")
    ap.add_argument("--arbitri", help="CSV con colonne HomeTeam,AwayTeam,Referee")
    ap.add_argument("--key", default=os.environ.get("ODDSPAPI_KEY"), help="chiave OddsPapi (o variabile ODDSPAPI_KEY)")
    ap.add_argument("--bookmaker", default="pinnacle")
    args = ap.parse_args()

    divs = list(LEAGUES) if args.lega.lower() == "all" else [x.strip() for x in args.lega.split(",")]
    for d in divs:
        if d not in LEAGUES:
            sys.exit(f"Lega sconosciuta: {d}. Usa: {', '.join(LEAGUES)}")
    df = load_history(args.csv_dir, divs)
    print(f"Caricate {len(df)} partite ({df.Date.min():%d/%m/%Y} - {df.Date.max():%d/%m/%Y})")
    refs = {}
    if args.arbitri:
        a = pd.read_csv(args.arbitri)
        refs = {(x.HomeTeam, x.AwayTeam): x.Referee for x in a.itertuples()}

    all_rows = []
    for div in divs:
        lname = LEAGUES[div][0]
        print(f"\n################ {lname} ################")
        if (df.Div == div).sum() < 100:
            print("Dati insufficienti per questo campionato (controlla il download o --csv-dir): saltato.")
            continue
        if args.backtest:
            backtest(df[df.Div == div].reset_index(drop=True))
            continue
        models = fit_all(df, div)
        fx = load_fixtures(args.fixtures, div)
        if fx.empty:
            print("Nessuna partita in programma trovata.")
            continue
        op = {}
        if args.key:
            try:
                op = match_odds(fx, fetch_odds(args.key, args.bookmaker, div))
                print(f"Quote OddsPapi ({args.bookmaker}) trovate per {len(op)}/{len(fx)} partite")
            except Exception as e:
                print(f"! quote OddsPapi non disponibili: {e}")
        for i, r in fx.iterrows():
            referee = refs.get((r.HomeTeam, r.AwayTeam))
            odds = {**csv_odds(r), **op.get(i, {})}
            picks, exp = analyze(models, r.HomeTeam, r.AwayTeam, referee, odds)
            if not picks:
                print(f"\n! {r.HomeTeam}-{r.AwayTeam}: squadra senza storico, saltata")
                continue
            for k in picks:
                all_rows.append(dict(lega=lname, data=f"{r.Date:%d/%m/%Y}", partita=f"{r.HomeTeam}-{r.AwayTeam}",
                                     arbitro=referee, statistica=k["stat"], mercato=k["label"], prob=round(k["p"], 4),
                                     quota_equa=round(1 / k["p"], 2), quota_book=k["odds"]))
            print(f"\n=== {r.HomeTeam} - {r.AwayTeam}  ({r.Date:%d/%m}){'  arbitro: ' + referee if referee else ''} ===")
            e = " | ".join(f"{s.lower()} {exp[s][0]:.1f}-{exp[s][1]:.1f}"
                           for s in ("Gol", "Corner", "Tiri in porta", "Cartellini") if s in exp)
            print(f"attesi: {e}")
            for k in select(picks, args.min_prob, args.min_fair, args.top):
                line = f"  - {k['stat']}: {k['label']} | {k['p']:.0%} (equa {1 / k['p']:.2f})"
                if k["odds"]:
                    line += f" | book {k['odds']:.2f} ({k['edge']:+.0%})" + ("  VALUE" if k["edge"] > 0.05 else "")
                print(line)

    if all_rows:
        pd.DataFrame(all_rows).to_csv("picks_completi.csv", index=False)
        print("\nTutte le probabilita' salvate in picks_completi.csv.")
        print("Dove manca la quota, gioca solo se il book paga >= quota equa x 1.05.")


if __name__ == "__main__":
    main()
