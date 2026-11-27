"""Sezione NBA dell'app (Streamlit)."""
import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import norm

import nba_model as N

LOSS_PER_PT = 0.25  # punti di squadra persi per ogni punto di media del giocatore assente (euristica)


@st.cache_data(ttl=3 * 3600, show_spinner="Scarico le partite NBA (puo' richiedere un minuto)...")
def load_games():
    return N.fetch_games()


@st.cache_data(ttl=3 * 3600, show_spinner="Calcolo i rating delle squadre...")
def get_model(_games, sig):
    return N.fit_team(_games, pd.Timestamp.now("UTC").tz_localize(None).normalize())


@st.cache_data(ttl=3600, show_spinner="Scarico i box score recenti...")
def get_logs(_games, tid, sig):
    return N.player_logs(_games, tid)


@st.cache_data(ttl=1800)
def get_inj(name_map):
    return N.fetch_injuries(name_map)


@st.cache_data(ttl=6 * 3600, show_spinner="Backtest in corso...")
def get_backtest(_games, sig):
    return N.backtest(_games)


def market_table(rows, key, min_prob, min_fair):
    data = [dict(Mercato=r["label"], Prob=f"{r['p']:.0%}", **{"Quota equa": round(1 / r["p"], 2),
                                                               "Minima (+5%)": round(1.05 / r["p"], 2)},
                 _p=r["p"], **{"Quota Eplay": None})
            for r in rows if r["p"] >= min_prob and 1 / r["p"] >= min_fair]
    if not data:
        st.write("Nessun mercato sopra le soglie.")
        return
    tbl = pd.DataFrame(data).sort_values("_p", ascending=False).reset_index(drop=True)
    st.write("Scrivi nella colonna **Quota Eplay** la quota che trovi per lo stesso mercato: sotto vedi se c'e' valore.")
    ed = st.data_editor(
        tbl, hide_index=True, use_container_width=True,
        column_order=["Mercato", "Prob", "Quota equa", "Minima (+5%)", "Quota Eplay"],
        disabled=["Mercato", "Prob", "Quota equa", "Minima (+5%)"],
        column_config={"Quota Eplay": st.column_config.NumberColumn(min_value=1.01, step=0.01, format="%.2f")},
        key=key)
    out = []
    for _, r in ed.iterrows():
        q = r["Quota Eplay"]
        if pd.notna(q):
            edge = r["_p"] * q - 1
            out.append({"Mercato": r["Mercato"], "Quota": q, "Edge": f"{edge:+.1%}",
                        "Esito": "✅ valore" if edge >= 0.05 else "salta"})
    if out:
        st.dataframe(pd.DataFrame(out), hide_index=True, use_container_width=True)
        st.caption("Edge = probabilita' x quota - 1. Dal +5% in su c'e' valore; oltre +15% controlla infortuni "
                   "e formazioni: di solito il modello sta sbagliando.")


def render():
    try:
        games = load_games()
    except Exception as e:
        st.error(f"Dati NBA non disponibili: {str(e)[:300]}")
        st.caption("Se il problema persiste mandami questo messaggio.")
        return
    sig = f"{len(games)}-{games[games.done].date.max()}"
    model = get_model(games, sig)
    now = pd.Timestamp.now("UTC").tz_localize(None)
    up = games[(~games.done) & (games.date >= now - pd.Timedelta(hours=4))].sort_values("date").head(15)
    st.caption(f"Dati fino al {games[games.done].date.max():%d/%m/%Y} · {int(games.done.sum())} partite")
    if up.empty:
        st.info("Nessuna partita NBA in programma nei prossimi giorni.")
        return

    def lab(i):
        g = up.iloc[i]
        t = g.date.tz_localize("UTC").tz_convert("Europe/Rome")
        return f"{g.an} @ {g.hn} · {t:%d/%m %H:%M}"

    i = st.selectbox("Partita (orari italiani)", range(len(up)), format_func=lab)
    g = up.iloc[i]
    names = tuple({**dict(zip(games.hn, games.hid)), **dict(zip(games.an, games.aid))}.items())
    try:
        inj = get_inj(names)
    except Exception:
        inj = {}
        st.warning("Infortuni non disponibili: scegli tu gli assenti qui sotto.")

    logs, tpts, rot, auto = {}, {}, {}, {}
    for side, tid in (("h", g.hid), ("a", g.aid)):
        logs[side], tpts[side] = get_logs(games, tid, sig)
        rot[side] = N.rotation(logs[side])
        ids = {x["id"]: x for x in inj.get(tid, []) if "out" in x["status"].lower()}
        auto[side] = [r["name"] for r in rot[side] if r["id"] in ids]

    st.markdown("**Assenze e condizioni**")
    outs, loss = {}, {}
    for side, nm, tid in (("h", g.hn, g.hid), ("a", g.an, g.aid)):
        opts = [r["name"] for r in rot[side]]
        outs[side] = st.multiselect(f"Assenti {nm}", opts, default=auto[side], key=f"out_{g.id}_{side}")
        dub = [f"{x['name']} ({x['status']})" for x in inj.get(tid, []) if "out" not in x["status"].lower()]
        if dub:
            st.caption(f"In dubbio per {nm}: " + ", ".join(dub[:6]))
        loss[side] = sum(LOSS_PER_PT * r["pts"] for r in rot[side] if r["name"] in outs[side])
    use_loss = st.checkbox("Applica le assenze alle proiezioni (euristica: -0,25 punti di squadra per ogni punto di media)", True)
    b2b = {}
    for side, tid in (("h", g.hid), ("a", g.aid)):
        prev = games[games.done & ((games.hid == tid) | (games.aid == tid)) & (games.date < g.date)].date.max()
        b2b[side] = pd.notna(prev) and (g.date - prev).total_seconds() < 36 * 3600
        if b2b[side]:
            st.caption(f"⚠️ {g.hn if side == 'h' else g.an} gioca in back-to-back (partita ieri).")
    use_b2b = st.checkbox("Applica il back-to-back (-1 punto alla squadra)", True)
    lh = (loss["h"] if use_loss else 0) + (1.0 if use_b2b and b2b["h"] else 0)
    la = (loss["a"] if use_loss else 0) + (1.0 if use_b2b and b2b["a"] else 0)

    if g.hid not in model["idx"] or g.aid not in model["idx"]:
        st.warning("Una delle due squadre non ha abbastanza storico.")
        return
    marg, tot = N.project(model, g.hid, g.aid, lh, la)
    ph = norm.cdf(marg / model["sm"])
    st.markdown(f"**{g.hn}** {ph:.0%} · **{g.an}** {1 - ph:.0%}")
    st.caption(f"Proiezione: {g.hn} {(tot + marg) / 2:.0f} - {g.an} {(tot - marg) / 2:.0f} · totale {tot:.1f} · "
               f"margine {marg:+.1f} · assenze: {lh:.1f} / {la:.1f} punti stimati persi")

    min_prob = st.slider("Probabilita' minima", 0.50, 0.90, 0.60, 0.01)
    min_fair = st.slider("Quota equa minima (scarta picks banali)", 1.10, 2.00, 1.25, 0.01)
    t1, t2, t3 = st.tabs(["Squadre", "Giocatori", "Affidabilita'"])

    with t1:
        rows = N.team_rows(model, g.hn, g.an, marg, tot)
        fam = st.multiselect("Mercati", ["Vincente", "Handicap", "Totale", "Totale squadra"],
                             default=["Vincente", "Handicap", "Totale"])
        market_table([r for r in rows if r["fam"] in fam], f"nba_t_{g.id}_{min_prob}_{min_fair}_{fam}_{lh:.1f}_{la:.1f}",
                     min_prob, min_fair)

    with t2:
        all_rows = []
        for side, nm in (("h", g.hn), ("a", g.an)):
            miss = sum(r["pts"] for r in rot[side] if r["name"] in outs[side]) if use_loss else 0
            ok = bool(tpts[side]) and tpts[side] == tpts[side]
            share = min(0.4, miss / tpts[side]) if ok else 0
            proj_pts = (tot + marg) / 2 if side == "h" else (tot - marg) / 2
            scale = float(np.clip(proj_pts / tpts[side], 0.92, 1.08)) if ok else 1.0
            ids_out = {r["id"] for r in rot[side] if r["name"] in outs[side]}
            all_rows += N.player_rows(logs[side], nm, scale, 1 + 0.5 * share, ids_out)
        if not all_rows:
            st.info("Dati giocatori non disponibili per questa partita.")
        else:
            ps = sorted({r["player"] for r in all_rows})
            who = st.selectbox("Giocatore", ["Tutti (migliori linee)"] + ps)
            stats = st.multiselect("Statistiche", list(N.STATS), default=["Punti", "Rimbalzi", "Assist"])
            sel = [r for r in all_rows if r["stat"] in stats]
            if who.startswith("Tutti"):
                best = {}
                for r in sel:
                    if r["p"] < min_prob or 1 / r["p"] < min_fair:
                        continue
                    k = (r["player"], r["stat"], r["side"])
                    if k not in best or r["p"] > best[k]["p"]:
                        best[k] = r
                sel = sorted(best.values(), key=lambda x: -x["p"])[:25]
            else:
                sel = [r for r in sel if r["player"] == who]
            market_table(sel, f"nba_p_{g.id}_{who}_{stats}_{min_prob}_{min_fair}", min_prob, min_fair)
            with st.expander("Medie recenti (ultime 10 partite)"):
                for side, nm in (("h", g.hn), ("a", g.an)):
                    st.write(f"**{nm}**")
                    st.dataframe(pd.DataFrame([dict(Giocatore=r["name"], Min=round(r["min"]), Pt=round(r["pts"], 1),
                                                    Rimb=round(r["reb"], 1), Ast=round(r["ast"], 1),
                                                    Triple=round(r["tpm"], 1), Assente="si" if r["name"] in outs[side] else "")
                                               for r in rot[side]]), hide_index=True, use_container_width=True)

    with t3:
        st.write("Controlla se le probabilita' di squadra (vincente, handicap, totali) si avverano davvero "
                 "sulle ultime 300 partite.")
        if st.button("Avvia backtest"):
            calib, per = get_backtest(games, sig)
            for t in (calib, per):
                for c in ("previsto", "realizzato"):
                    t[c] = (t[c] * 100).round(1).astype(str) + "%"
            st.dataframe(calib, hide_index=True, use_container_width=True)
            st.write("Per mercato (picks >= 60%):")
            st.dataframe(per, hide_index=True, use_container_width=True)
            st.caption("Il backtest non copre i giocatori. Se 'realizzato' e' molto piu' basso di 'previsto', "
                       "quel mercato non e' affidabile.")

    st.caption("Il modello non conosce rotazioni decise all'ultimo, minuti limitati, riposi dei titolari e motivazione. "
               "Controlla gli infortuni ufficiali prima di giocare. Nessun esito e' sicuro. Gioca responsabilmente (18+).")
