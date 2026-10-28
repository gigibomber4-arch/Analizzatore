"""App web (Streamlit) per l'analizzatore dei 5 grandi campionati: si apre dal browser del telefono."""
import os

import pandas as pd
import streamlit as st

import analizzatore_seriea as A

st.set_page_config(page_title="Football Analyzer", page_icon="⚽", layout="centered")


@st.cache_data(ttl=6 * 3600, show_spinner="Scarico i dati storici...")
def load_df():
    return A.load_history()


@st.cache_data(ttl=6 * 3600, show_spinner="Addestro il modello...")
def load_models(_df, div):
    return A.fit_all(_df, div)


@st.cache_data(ttl=1800, show_spinner="Scarico il calendario...")
def load_fx(div):
    return A.load_fixtures(div=div)


@st.cache_data(ttl=1800, show_spinner="Scarico le quote (puo' richiedere ~30 secondi)...")
def load_op(key, bookmaker, div):
    return A.fetch_odds(key, bookmaker, div)


@st.cache_data(ttl=6 * 3600, show_spinner="Backtest in corso (1-2 minuti)...")
def load_backtest(_df, div):
    return A.backtest_tables(_df[_df.Div == div].reset_index(drop=True))


def secret_key():
    try:
        return st.secrets["ODDSPAPI_KEY"]
    except Exception:
        return os.environ.get("ODDSPAPI_KEY", "")


st.title("⚽ Football Analyzer")

div = st.selectbox("Campionato", list(A.LEAGUES), format_func=lambda d: A.LEAGUES[d][0])

with st.expander("Impostazioni"):
    min_prob = st.slider("Probabilita' minima", 0.50, 0.90, 0.60, 0.01)
    min_fair = st.slider("Quota equa minima (scarta picks banali)", 1.10, 2.00, 1.25, 0.01)
    top = st.slider("Picks per partita", 3, 12, 6)
    value_thr = st.slider("Soglia value (edge)", 0.02, 0.30, 0.05, 0.01)
    bookmaker = st.text_input("Bookmaker (slug OddsPapi)", "pinnacle")
    key = secret_key() or st.text_input("Chiave OddsPapi (opzionale)", type="password")

df = load_df()
models = load_models(df, div)
fx = load_fx(div).reset_index(drop=True)
if fx.empty:
    st.info("Nessuna partita in programma trovata.")
    st.stop()

op = {}
if key:
    try:
        op = A.match_odds(fx, load_op(key, bookmaker, div))
        st.caption(f"Quote {bookmaker}: trovate per {len(op)}/{len(fx)} partite. "
                   "I corner si aprono ~3 giorni prima del match, i cartellini ~1 giorno prima.")
    except Exception as e:
        st.warning(f"Quote OddsPapi non disponibili: {e}")
else:
    st.caption("Senza chiave OddsPapi: quote solo per 1X2 e Over/Under 2.5 (football-data).")

refs = []
if "Referee" in df.columns:
    d = df[df.Div == div]
    recent = d[d.Date >= d.Date.max() - pd.Timedelta(days=400)]
    refs = recent.Referee.dropna().value_counts().index.tolist()

# analisi di tutte le partite (usa l'arbitro scelto, se presente)
res = {}
for i, r in fx.iterrows():
    ref = st.session_state.get(f"ref{div}_{i}", "—")
    referee = None if ref == "—" else ref
    odds = {**A.csv_odds(r), **op.get(i, {})}
    res[i] = A.analyze(models, r.HomeTeam, r.AwayTeam, referee, odds)


def fmt(k):
    s = f"**{k['stat']}**: {k['label']} · {k['p']:.0%} (eq. {1 / k['p']:.2f})"
    if k["odds"]:
        s += f" · book {k['odds']:.2f} ({k['edge']:+.0%})" + (" ✅" if k["edge"] > value_thr else "")
    return s


tab1, tab2, tab3, tab4 = st.tabs(["Giornata", "Value", "Tutti i mercati", "Affidabilita'"])

with tab1:
    for i, r in fx.iterrows():
        picks, exp = res[i]
        with st.expander(f"{r.HomeTeam} - {r.AwayTeam} · {r.Date:%d/%m}"):
            if not picks:
                st.write("Squadra senza storico, partita saltata.")
                continue
            if refs:
                st.selectbox("Arbitro", ["—"] + refs, key=f"ref{div}_{i}")
                sel = st.session_state.get(f"ref{div}_{i}", "—")
                if sel != "—":
                    eff = A.referee_effect(models, sel)
                    st.caption("Effetto arbitro: " + ", ".join(f"{s.lower()} x{v:.2f}" for s, v in eff.items()))
            st.caption(" | ".join(f"{s.lower()} {exp[s][0]:.1f}-{exp[s][1]:.1f}"
                                  for s in ("Gol", "Corner", "Tiri in porta", "Cartellini") if s in exp))
            chosen = A.select(picks, min_prob, min_fair, top)
            st.markdown("\n".join("- " + fmt(k) for k in chosen) if chosen else "Nessun pick sopra le soglie.")

with tab2:
    rows = []
    for i, r in fx.iterrows():
        for k in res[i][0]:
            if k["odds"] and k["edge"] >= value_thr and k["p"] >= 0.5:
                rows.append({"Partita": f"{r.HomeTeam}-{r.AwayTeam}", "Mercato": f"{k['stat']}: {k['label']}",
                             "Prob": f"{k['p']:.0%}", "Quota": k["odds"], "Edge": f"{k['edge']:+.0%}", "_e": k["edge"]})
    if rows:
        out = pd.DataFrame(rows).sort_values("_e", ascending=False).drop(columns="_e")
        st.dataframe(out, hide_index=True, use_container_width=True)
        st.caption("Un edge molto alto (>15%) di solito indica un errore del modello (infortuni, formazioni), non un affare.")
    else:
        st.write("Nessun value trovato (servono le quote: aggiungi la chiave OddsPapi nelle impostazioni).")

with tab3:
    names = {i: f"{r.HomeTeam} - {r.AwayTeam}" for i, r in fx.iterrows()}
    i = st.selectbox("Partita", list(names), format_func=names.get)
    stats = st.multiselect("Statistiche", list(A.STATS), default=["Corner", "Cartellini"])
    rows = [{"Mercato": f"{k['stat']}: {k['label']}", "Prob": k["p"], "Quota equa": round(1 / k["p"], 2),
             "Book": k["odds"]} for k in res[i][0] if k["stat"] in stats and k["p"] >= 0.5]
    if rows:
        st.dataframe(pd.DataFrame(rows).sort_values("Prob", ascending=False).style.format({"Prob": "{:.0%}"}),
                     hide_index=True, use_container_width=True)

with tab4:
    st.write("Controlla se le probabilita' del modello si avverano davvero sulle ultime 150 partite del campionato scelto.")
    if st.button("Avvia backtest"):
        calib, per = load_backtest(df, div)
        for t in (calib, per):
            for c in ("previsto", "realizzato"):
                t[c] = (t[c] * 100).round(1).astype(str) + "%"
        st.dataframe(calib, hide_index=True, use_container_width=True)
        st.write("Per statistica (picks >= 60%):")
        st.dataframe(per, hide_index=True, use_container_width=True)
        st.caption("Se 'realizzato' e' molto piu' basso di 'previsto', quel mercato non e' affidabile.")

st.caption("Strumento statistico: nessun esito e' sicuro. Gioca responsabilmente (18+).")
