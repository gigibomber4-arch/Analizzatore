"""Sezione tennis dell'app (Streamlit)."""
import os

import pandas as pd
import streamlit as st

import tennis_model as T

INSTR = (
    "1. Apri **tennis-data.co.uk/data.php** dal browser del telefono.\n"
    "2. Scarica i file **xlsx** del circuito (almeno l'anno in corso e i due precedenti).\n"
    "3. Caricali qui (apri 'Dati tennis'). Il circuito si riconosce dal contenuto, il nome non conta.\n"
    "4. Per averli sempre pronti, caricali anche su GitHub nella pagina principale del repository "
    "(Add file, poi Upload files)."
)

STATUS = {"In forma": 0.0, "Dubbio fisico (-40 Elo)": 40.0, "Non al meglio / infortunio (-100 Elo)": 100.0}


@st.cache_data(ttl=6 * 3600, show_spinner="Leggo i dati tennis...")
def load(items, tour):
    return T.load_tennis(list(items), tour)


@st.cache_data(ttl=6 * 3600, show_spinner="Calcolo i rating...")
def get_model(_df, sig):
    return T.build_ratings(_df)[0]


@st.cache_data(ttl=6 * 3600, show_spinner="Calibro il modello...")
def get_hold(_df, sig):
    return {s: T.calibrate_hold(g) for s, g in T.games_per_set(_df).items()}


@st.cache_data(ttl=6 * 3600, show_spinner="Backtest in corso...")
def get_backtest(_df, sig):
    return T.backtest(_df)


def info_box(model, name, today):
    i = T.player_info(model, name, today)
    st.caption(f"**{name}** · {i['matches']} match · ultimo {i['days']} giorni fa · "
               f"{i['games7']} game negli ultimi 7 giorni")
    if i["retired"]:
        st.warning(f"⚠️ {name} si e' ritirato nell'ultimo match: possibile problema fisico.")
    if i["days"] is not None and i["days"] > 45:
        st.warning(f"⚠️ {name} non gioca da {i['days']} giorni (rientro? ritmo partita?).")
    return i


def render():
    tour = st.radio("Circuito", ["ATP", "WTA"], horizontal=True)
    base = os.path.dirname(os.path.abspath(__file__))
    with st.expander("Dati tennis: carica o aggiorna i file"):
        st.write(INSTR)
        ups = st.file_uploader("File xlsx di tennis-data.co.uk", type=["xlsx", "csv"], accept_multiple_files=True)
    items = tuple(T.local_files(base)) + tuple((f.name, f.getvalue()) for f in (ups or []))
    try:
        df = load(items, tour)
    except Exception:
        st.info(f"Per usare il tennis servono i dati {tour}.\n\n" + INSTR)
        return
    st.caption(f"Dati disponibili fino al {df.Date.max():%d/%m/%Y} ({len(df)} partite)")
    sig = f"{tour}-{len(df)}-{df.Date.max()}"
    model = get_model(df, sig)
    h0map = get_hold(df, sig)
    players = T.active_players(model, df)
    if len(players) < 2:
        st.info("Dati insufficienti.")
        return
    today = pd.Timestamp.today().normalize()

    tab1, tab2 = st.tabs(["Analisi partita", "Affidabilita'"])
    with tab1:
        a = st.selectbox("Giocatore 1", players, index=0)
        b = st.selectbox("Giocatore 2", players, index=1)
        if a == b:
            st.warning("Scegli due giocatori diversi.")
            return
        last_surface = df.Surface.iloc[-1]
        surface = st.selectbox("Superficie", T.SURFACES,
                               index=T.SURFACES.index(last_surface) if last_surface in T.SURFACES else 0)
        best_of = 3
        if tour == "ATP":
            best_of = 5 if st.radio("Formato", ["Al meglio dei 3", "Al meglio dei 5"], horizontal=True).endswith("5") else 3

        st.markdown("**Informazioni aggiuntive**")
        ia, ib = info_box(model, a, today), info_box(model, b, today)
        sa = st.selectbox(f"Condizione di {a}", list(STATUS))
        sb = st.selectbox(f"Condizione di {b}", list(STATUS))
        auto_fat = st.checkbox("Applica stanchezza automatica (euristica: -1 Elo per game oltre 45 in 7 giorni)")
        penA = STATUS[sa] + (T.fatigue_pen(ia["games7"]) if auto_fat else 0)
        penB = STATUS[sb] + (T.fatigue_pen(ib["games7"]) if auto_fat else 0)

        res = T.analyze(model, h0map, a, b, surface, best_of, penA, penB)
        st.markdown(f"**{a}** {res['pA']:.0%} · **{b}** {1 - res['pA']:.0%} · "
                    f"game attesi {res['exp_games']:.1f}")
        st.caption(f"Rating {surface}: {a} {res['rA']:.0f} · {b} {res['rB']:.0f}")

        min_prob = st.slider("Probabilita' minima", 0.50, 0.90, 0.60, 0.01)
        rows = [dict(Mercato=m, Prob=f"{p:.0%}", **{"Quota equa": round(1 / p, 2), "Minima (+5%)": round(1.05 / p, 2)},
                     _p=p, **{"Quota Eplay": None})
                for m, p in T.market_rows(res, a, b, best_of) if p >= min_prob and 1 / p >= 1.15]
        if not rows:
            st.write("Nessun mercato sopra le soglie.")
        else:
            tbl = pd.DataFrame(rows).sort_values("_p", ascending=False).reset_index(drop=True)
            st.write("Scrivi nella colonna **Quota Eplay** le quote che trovi, e sotto vedi se c'e' valore.")
            edited = st.data_editor(
                tbl, hide_index=True, use_container_width=True,
                column_order=["Mercato", "Prob", "Quota equa", "Minima (+5%)", "Quota Eplay"],
                disabled=["Mercato", "Prob", "Quota equa", "Minima (+5%)"],
                column_config={"Quota Eplay": st.column_config.NumberColumn(min_value=1.01, step=0.01, format="%.2f")},
                key=f"ed_{tour}_{a}_{b}_{surface}_{best_of}_{min_prob}_{penA}_{penB}")
            out = []
            for _, r in edited.iterrows():
                q = r["Quota Eplay"]
                if pd.notna(q):
                    edge = r["_p"] * q - 1
                    out.append({"Mercato": r["Mercato"], "Quota": q, "Edge": f"{edge:+.1%}",
                                "Esito": "✅ valore" if edge >= 0.05 else "salta"})
            if out:
                st.dataframe(pd.DataFrame(out), hide_index=True, use_container_width=True)
            st.caption("Edge = probabilita' x quota - 1. Dal +5% in su c'e' valore; oltre +15% controlla "
                       "infortuni e condizioni: di solito il modello sta sbagliando.")
        st.caption("Il modello non conosce condizioni di gioco, fatica reale, motivazione. "
                   "Nessun esito e' sicuro. Gioca responsabilmente (18+).")

    with tab2:
        st.write("Confronta il modello con le ultime 800 partite e, se disponibili, con le quote dei bookmaker.")
        if st.button("Avvia backtest"):
            calib, ll = get_backtest(df, sig)
            c = calib.copy()
            for col in ("previsto", "realizzato"):
                c[col] = (c[col] * 100).round(1).astype(str) + "%"
            st.dataframe(c, hide_index=True, use_container_width=True)
            msg = f"Errore (log-loss, piu' basso = meglio): modello {ll['elo']:.3f}"
            if ll["bookmaker"]:
                msg += f" · bookmaker {ll['bookmaker']:.3f}"
            st.write(msg)
            st.caption("Se il modello e' peggiore dei bookmaker (di solito lo e'), le loro quote riflettono "
                       "informazioni che il modello non ha: cerca valore con cautela.")
