# alejandria_thoth_final.py
"""
Los Factores de Alejandría — Versión Final con Thoth fijo, ayudas mejoradas y visuales
Guardar como alejandria_thoth_final.py
Ejecutar:
  pip install streamlit plotly scikit-learn scipy pandas numpy
  streamlit run alejandria_thoth_final.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.decomposition import FactorAnalysis
from sklearn.preprocessing import StandardScaler
from scipy.linalg import svd, eigh
from scipy.stats import chi2
import random
import os

# ---------------------------
# Page config & CSS (tema)
# ---------------------------
st.set_page_config(page_title="Los Factores de Alejandría — Thoth", layout="wide", page_icon="⚱️")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Merriweather:wght@300;400;700&family=Inter:wght@300;400;600&display=swap');

body {
  background: linear-gradient(180deg,#fbf7ef 0%, #efe3c7 100%);
  color: #2f2a24;
}
.header-title {
  font-family: 'Merriweather', serif;
  color: #3d2f1b;
  font-size: 30px;
  margin: 6px 0;
}
.header-sub {
  font-family: 'Inter', sans-serif;
  color: #6b5a3a;
  margin-bottom: 12px;
}
.card {
  background: rgba(255,255,255,0.85);
  padding: 12px;
  border-radius: 10px;
  box-shadow: 0 6px 18px rgba(0,0,0,0.06);
  border: 1px solid rgba(120,90,40,0.06);
}
.thoth-img img {
  border-radius: 10px;
  border: 3px solid #c9a65a;
  box-shadow: 0 8px 20px rgba(0,0,0,0.14);
}
.tooltip {
  background: linear-gradient(180deg,#2b2b2b,#161616);
  color: #fff8dc;
  padding: 10px;
  border-radius: 8px;
  font-size: 14px;
}
.small-muted { color:#6b5a3a; font-size:13px; }
.level-title { font-family: 'Merriweather', serif; font-size:20px; color:#2f2a24; }
</style>
""", unsafe_allow_html=True)

# ---------------------------
# Utility functions: stats & rotations
# ---------------------------
def varimax(Phi, gamma=1.0, q=50, tol=1e-6):
    p, k = Phi.shape
    R = np.eye(k)
    d = 0
    for i in range(q):
        d_old = d
        Lambda = Phi @ R
        u, s, vh = svd(Phi.T @ (Lambda**3 - (gamma / p) * (Lambda @ np.diag(np.diag(Lambda.T @ Lambda)))))
        R = u @ vh
        d = np.sum(s)
        if d_old != 0 and d / d_old < 1 + tol:
            break
    return Phi @ R

def promax(Phi, power=1.5):
    Phi_v = varimax(Phi)
    return np.sign(Phi_v) * (np.abs(Phi_v) ** power)

def kmo_test(df):
    corr = df.corr().values
    try:
        inv_corr = np.linalg.inv(corr)
    except Exception:
        return np.nan, pd.Series(np.nan, index=df.columns)
    partial = -inv_corr.copy()
    d = np.sqrt(np.diag(inv_corr))
    partial = (partial / d).T / d
    np.fill_diagonal(partial, 0)
    a = np.abs(corr)
    b = np.abs(partial)
    kmo_num = a.sum() - np.trace(a)
    kmo_denom = kmo_num + b.sum()
    overall = kmo_num / kmo_denom if kmo_denom != 0 else np.nan
    per_item = (a.sum(axis=0) - np.diag(a)) / ((a.sum(axis=0) - np.diag(a)) + (b.sum(axis=0)))
    per_item = pd.Series(np.nan_to_num(per_item), index=df.columns)
    return overall, per_item

def bartlett_sphericity(df):
    n, p = df.shape
    R = df.corr().values
    try:
        detR = np.linalg.det(R)
    except Exception:
        return np.nan, np.nan, p*(p-1)//2
    if detR <= 0:
        return np.nan, np.nan, p*(p-1)//2
    chi2_val = -(n - 1 - (2*p + 5)/6.0) * np.log(detR)
    dfree = p*(p-1)//2
    pval = 1 - chi2.cdf(chi2_val, dfree)
    return chi2_val, pval, dfree

# ---------------------------
# Scenario generator (data)
# ---------------------------
def generar_escenario(n_samples=160, n_vars=8, min_f=2, max_f=4, noise=0.30):
    np.random.seed(np.random.randint(0, 999999))
    n_f = np.random.randint(min_f, max_f + 1)
    F = np.random.normal(size=(n_samples, n_f))
    L = np.random.uniform(0.45, 0.95, size=(n_vars, n_f))
    # sparsify
    mask = np.random.rand(n_vars, n_f) < 0.25
    L[mask] *= np.random.uniform(0.0, 0.5, size=mask.sum())
    X = F @ L.T + np.random.normal(scale=noise, size=(n_samples, n_vars))
    cols = [f"Var{i+1}" for i in range(n_vars)]
    df = pd.DataFrame(X, columns=cols)
    return df, L, n_f

# ---------------------------
# Session init
# ---------------------------
if "started" not in st.session_state:
    st.session_state.started = False
if "scenario" not in st.session_state:
    st.session_state.scenario = None
if "level" not in st.session_state:
    st.session_state.level = 0
for key in ["user_answers","correct_answers","explanations","points_per_level"]:
    if key not in st.session_state:
        st.session_state[key] = []
if "player" not in st.session_state:
    st.session_state.player = "Helios"
if "xp" not in st.session_state:
    st.session_state.xp = 0

# ---------------------------
# Start new game: prepare dynamic correct answers safely
# ---------------------------
def start_new_game():
    df, L_true, n_f = generar_escenario()
    st.session_state.scenario = {"df": df, "L": L_true, "n_f": n_f}
    # derive correct answers dynamically
    corr = df.corr().abs()
    np.fill_diagonal(corr.values, 0)
    max_idx = np.unravel_index(np.nanargmax(corr.values), corr.shape)
    pair = (corr.index[max_idx[0]], corr.columns[max_idx[1]])
    true_nf = n_f
    var_top_f1 = f"Var{int(np.argmax(np.abs(L_true[:,0]))+1)}"
    # rotations on true loadings
    try:
        v = varimax(L_true.copy())
    except Exception:
        v = L_true.copy()
    try:
        p = promax(L_true.copy())
    except Exception:
        p = L_true.copy()
    simp_v = np.sum(np.max(np.abs(v), axis=1))
    simp_p = np.sum(np.max(np.abs(p), axis=1))
    rotation_best = "Varimax" if simp_v >= simp_p else "Promax"
    rotated = v if rotation_best == "Varimax" else p
    communalities = np.sum(rotated**2, axis=1)
    var_top_comm = f"Var{int(np.argmax(communalities)+1)}"
    overall_kmo, per_item = kmo_test(df)
    chi2_stat, pval, dfree = bartlett_sphericity(df)
    bartlett_significant = (not np.isnan(pval)) and (pval < 0.05)
    top3_f1 = list(pd.Series(np.abs(rotated[:,0]), index=[f"Var{i+1}" for i in range(rotated.shape[0])]).sort_values(ascending=False).head(3).index)
    summary_phrase = f"Modelo: {true_nf} factores, rotación {rotation_best} — patrones interpretables en este pergamino."

    st.session_state.correct_dynamic = {
        1: {"answer": f"{pair[0]} & {pair[1]}", "explanation": "Pareja con mayor correlación absoluta en este pergamino."},
        2: {"answer": str(true_nf), "explanation": "Número latente de factores usado para generar los datos."},
        3: {"answer": var_top_f1, "explanation": "Variable con mayor carga en el Factor 1 de la matriz verdadera."},
        4: {"answer": rotation_best, "explanation": f"Rotación ({rotation_best}) que produjo la estructura más interpretable."},
        5: {"answer": var_top_comm, "explanation": "Variable con mayor comunalidad tras la rotación elegida."},
        6: {"answer": "Sí" if bartlett_significant else "No", "explanation": f"Bartlett p-value ≈ {pval if not np.isnan(pval) else 'N/A'}."},
        7: {"answer": ", ".join(top3_f1), "explanation": "Top-3 variables representativas del Factor 1 en esta rotación."},
        8: {"answer": summary_phrase, "explanation": "Síntesis interpretativa esperada para esta partida."}
    }

    # reset trackers
    st.session_state.user_answers = []
    st.session_state.correct_answers = []
    st.session_state.explanations = []
    st.session_state.points_per_level = []
    st.session_state.level = 1
    st.session_state.xp = 0
    st.session_state.started = True

# ---------------------------
# Sidebar: Thoth fixed + controls + help mode
# ---------------------------
with st.sidebar:
    st.markdown("<div class='card' style='text-align:center'>", unsafe_allow_html=True)
    # Thoth image (public domain / wikipedia). Replace with local file if desired.
    current_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(current_dir, "Toth.jpg")

    st.image(image_path, caption="Thoth — Dios de la Sabiduría", use_container_width=True)

    st.markdown("<h3 style='margin:6px 0 2px 0;'>Thoth — tu guía</h3>", unsafe_allow_html=True)
    st.markdown("<div class='small-muted'>Thoth aparece siempre en su imagen: calma, sabiduría y guía pedagógica.</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.text_input("Nombre del erudito:", key="player_name", value=st.session_state.player)
    st.session_state.player = st.session_state.player if st.session_state.player else st.session_state.player
    st.markdown("---")
    help_mode = st.selectbox("Modo de ayuda de Thoth:", ["Breve", "Teórica", "Pista"], index=0)
    st.markdown("---")
    if st.button("🔄 Nueva partida (escenario aleatorio)"):
        start_new_game()
        st.rerun()
    if st.button("⏮ Reiniciar todo"):
        keys = ["scenario","started","level","user_answers","correct_answers","explanations","points_per_level","xp","fa_player","loadings_player","rotated_player","rot_choice","correct_dynamic"]
        for k in keys:
            if k in st.session_state:
                del st.session_state[k]
        st.session_state.player = "Helios"
        st.rerun()
    st.markdown("---")
    st.write("Nivel:", st.session_state.level if "level" in st.session_state else "—")
    st.write("XP:", st.session_state.xp)

# ---------------------------
# Ensure scenario exists
# ---------------------------
if not st.session_state.started:
    start_new_game()

df = st.session_state.scenario["df"]
L_true = st.session_state.scenario["L"]
n_f_true = st.session_state.scenario["n_f"]

# ---------------------------
# Thoth messages per level (three modes)
# ---------------------------
msgs_brief = {
    1: "Observa las correlaciones; busca parejas que se muevan juntas.",
    2: "Decide cuántos factores retener: observa el scree y eigenvalues.",
    3: "Prueba Varimax para simplificar interpretaciones.",
    4: "Promax permite correlaciones entre factores: útil en datos reales.",
    5: "Calcula comunalidades para ver cuánto explica cada variable.",
    6: "KMO y Bartlett ayudan a validar si AF es apropiado.",
    7: "Selecciona las variables con mayor carga en el factor elegido.",
    8: "Revisa todo: interpretación + validación."
}
msgs_theory = {
    1: "Teoría: la correlación mide relación lineal. Matriz de correlaciones muestra grupos de variables afines.",
    2: "Teoría: los factores son variables latentes; autovalores indican cuánto varianza explican.",
    3: "Teoría: Varimax es una rotación ortogonal que facilita identificar variables dominantes por factor.",
    4: "Teoría: rotaciones oblicuas (Promax) permiten que factores se correlacionen — realista en ciencias sociales.",
    5: "Teoría: comunalidad = suma de squared loadings; indica varianza explicada por todos los factores.",
    6: "Teoría: KMO mide adecuación muestral; Bartlett prueba si la correlación no es identidad.",
    7: "Teoría: las cargas>0.4 suelen ser consideradas relevantes; atención a doble cargas.",
    8: "Teoría: valida estabilidad, interpretabilidad y utilidad práctica del modelo factorial."
}
msgs_hint = {
    1: "Pista: mira el valor absoluto más alto fuera de la diagonal en la matriz de correlaciones.",
    2: "Pista: el 'codo' en el scree plot indica número razonable de factores.",
    3: "Pista: busca la rotación que haga las cargas más esparcidas (unos altos, otros bajos).",
    4: "Pista: compara la matriz de correlaciones entre factores si usas Promax.",
    5: "Pista: comunalidades = (rotadas**2).sum(axis=1).",
    6: "Pista: KMO > 0.6 y Bartlett p < 0.05 es una buena señal.",
    7: "Pista: elige variables top-3 por carga absoluta en el factor.",
    8: "Pista: revisa cada explicación para ver cómo se construyó la respuesta correcta."
}

# Helper to show Thoth help
def thoth_help(level, mode):
    if mode == "Breve":
        txt = msgs_brief.get(level, "")
    elif mode == "Teórica":
        txt = msgs_theory.get(level, "")
    else:
        txt = msgs_hint.get(level, "")
    st.sidebar.info(f"Thoth ({mode}): {txt}")

thoth_help(st.session_state.level, help_mode)

# ---------------------------
# Record answer helper
# ---------------------------
def record_answer(level, user_ans, pts):
    st.session_state.user_answers.append(user_ans)
    correct = st.session_state.correct_dynamic[level]["answer"]
    st.session_state.correct_answers.append(correct)
    st.session_state.explanations.append(st.session_state.correct_dynamic[level].get("explanation", ""))
    st.session_state.points_per_level.append(pts)
    st.session_state.xp += pts

# ---------------------------
# Main area header
# ---------------------------
colA, colB = st.columns([3,1])
with colA:
    st.markdown("<div class='header-title'>⚱️ Los Factores de Alejandría</div>", unsafe_allow_html=True)
    st.markdown("<div class='header-sub'>Edición con Thoth como guía — 8 misiones educativas</div>", unsafe_allow_html=True)
    st.write(f"Erudito: **{st.session_state.player}** — XP: **{st.session_state.xp}**")
with colB:
    st.markdown("<div class='card small-muted' style='text-align:center'>Progreso</div>", unsafe_allow_html=True)
    st.progress(min(100, st.session_state.xp))

# ---------------------------
# Levels logic (1..8)
# ---------------------------
level = st.session_state.level

# Utility to safely sample distractors
def safe_sample(population, k):
    k_valid = min(k, max(0, len(population)))
    if k_valid == 0:
        return []
    return random.sample(population, k=k_valid)

# LEVEL 1 - Correlations
if level == 1:
    st.markdown("<div class='level-title'>Misión 1 — Biblioteca: Observa correlaciones</div>", unsafe_allow_html=True)
    st.markdown("<div class='card'>Busca la pareja de variables con mayor correlación (valor absoluto).</div>", unsafe_allow_html=True)
    corr = df.corr().abs()
    np.fill_diagonal(corr.values, 0)
    fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r", title="Matriz de correlaciones (abs)")
    st.plotly_chart(fig, use_container_width=True)
    correct_pair = st.session_state.correct_dynamic[1]["answer"]
    vars_list = list(df.columns)
    all_pairs = [f"{a} & {b}" for i,a in enumerate(vars_list) for b in vars_list[i+1:]]
    distractors = safe_sample([p for p in all_pairs if p != correct_pair], k=2)
    options = [correct_pair] + distractors
    random.shuffle(options)
    choice = st.radio("¿Cuál pareja muestra mayor correlación en este pergamino?", options)
    if st.button("Confirmar Nivel 1"):
        pts = 12 if choice == correct_pair else 0
        record_answer(1, choice, pts)
        if pts > 0:
            st.success("Thoth: Excelente observación.")
        else:
            st.error(f"Thoth: No exactamente. La pareja correcta era: {correct_pair}")
        st.session_state.level = 2
        st.rerun()

# LEVEL 2 - Extraction (scree)
elif level == 2:
    st.markdown("<div class='level-title'>Misión 2 — Taller: Extracción de factores</div>", unsafe_allow_html=True)
    st.markdown("<div class='card'>Usa el scree plot para estimar cuántos factores retener.</div>", unsafe_allow_html=True)
    corr = df.corr().values
    eigvals, _ = eigh(corr)
    eig = np.sort(eigvals)[::-1]
    scree_df = pd.DataFrame({"Index": np.arange(1, len(eig)+1), "Eigen": eig})
    fig_scree = px.line(scree_df, x="Index", y="Eigen", markers=True, title="Scree plot (autovalores)")
    st.plotly_chart(fig_scree, use_container_width=True)
    opt_min = 1
    opt_max = min(6, df.shape[1])
    guess = st.number_input("¿Cuántos factores retendrías?", min_value=opt_min, max_value=opt_max, value=min(3,opt_max))
    if st.button("Confirmar Nivel 2"):
        correct = st.session_state.correct_dynamic[2]["answer"]
        pts = 12 if str(guess) == correct else 0
        record_answer(2, str(guess), pts)
        if pts > 0:
            st.success("Thoth: Buena intuición — coincide con la estructura latente.")
        else:
            st.error(f"Thoth: No coincide con el número real ({correct}). Observa el codo.")
        # build player's FA for later demonstration
        n_extract = int(guess)
        Xs = StandardScaler().fit_transform(df)
        fa = FactorAnalysis(n_components=n_extract, random_state=42)
        _ = fa.fit_transform(Xs)
        st.session_state.fa_player = fa
        st.session_state.loadings_player = pd.DataFrame(fa.components_.T, index=df.columns, columns=[f"F{i+1}" for i in range(n_extract)])
        st.session_state.level = 3
        st.rerun()

# LEVEL 3 - Rotations
elif level == 3:
    st.markdown("<div class='level-title'>Misión 3 — Templo: Rotaciones</div>", unsafe_allow_html=True)
    st.markdown("<div class='card'>Compara Varimax y Promax; elige la rotación más interpretable.</div>", unsafe_allow_html=True)
    if "loadings_player" not in st.session_state:
        st.warning("Completa la extracción en el Nivel 2 primero.")
    else:
        Lp = st.session_state.loadings_player.values
        try:
            v = varimax(Lp)
        except Exception:
            v = Lp
        try:
            p = promax(Lp)
        except Exception:
            p = Lp
        st.markdown("Cargas Varimax (vista):")
        st.dataframe(pd.DataFrame(np.round(v,3), index=st.session_state.loadings_player.index, columns=st.session_state.loadings_player.columns).style.background_gradient(cmap="RdBu_r"))
        st.markdown("Cargas Promax (vista):")
        st.dataframe(pd.DataFrame(np.round(p,3), index=st.session_state.loadings_player.index, columns=st.session_state.loadings_player.columns).style.background_gradient(cmap="RdBu_r"))
        choice = st.radio("¿Cuál rotación hace las cargas más simples?", ["Varimax","Promax"])
        if st.button("Confirmar Nivel 3"):
            correct = st.session_state.correct_dynamic[4]["answer"]
            pts = 10 if choice == correct else 0
            record_answer(3, choice, pts)
            if pts > 0:
                st.success("Thoth: Correcto — la rotación facilita la interpretación.")
            else:
                st.error(f"Thoth: No era la rotación esperada en este escenario (correcta: {correct}).")
            st.session_state.rot_choice = choice
            st.session_state.rotated_player = v if choice == "Varimax" else p
            st.session_state.level = 4
            st.rerun()

# LEVEL 4 - Interpretation
elif level == 4:
    st.markdown("<div class='level-title'>Misión 4 — Faro: Interpretación de Factor 1</div>", unsafe_allow_html=True)
    st.markdown("<div class='card'>Identifica la variable que contribuye más al Factor 1 (según tu rotación).</div>", unsafe_allow_html=True)
    if "rotated_player" not in st.session_state:
        st.warning("Aplica rotación primero (Nivel 3).")
    else:
        rot = st.session_state.rotated_player
        top_var = pd.Series(np.abs(rot[:,0]), index=st.session_state.loadings_player.index).idxmax()
        others = [v for v in st.session_state.loadings_player.index if v != top_var]
        opts = [top_var] + safe_sample(others, k=2)
        random.shuffle(opts)
        choice = st.radio("¿Qué variable domina el Factor 1?", opts)
        if st.button("Confirmar Nivel 4"):
            pts = 10 if choice == top_var else 0
            record_answer(4, choice, pts)
            if pts > 0:
                st.success("Thoth: Bien interpretado.")
            else:
                st.error(f"Thoth: La correcta era {top_var}.")
            st.session_state.level = 5
            st.rerun()

# LEVEL 5 - Communalities
elif level == 5:
    st.markdown("<div class='level-title'>Misión 5 — Concilio: Comunalidades</div>", unsafe_allow_html=True)
    st.markdown("<div class='card'>Calcula la comunalidad (suma de cuadrados de cargas) y elige la más alta.</div>", unsafe_allow_html=True)
    if "rotated_player" not in st.session_state:
        st.warning("Finaliza rotación primero.")
    else:
        rot = st.session_state.rotated_player
        communal = pd.Series(np.sum(rot**2, axis=1), index=st.session_state.loadings_player.index)
        correct_var = communal.idxmax()
        opts = [correct_var] + safe_sample([v for v in communal.index if v != correct_var], k=2)
        random.shuffle(opts)
        st.dataframe(pd.DataFrame({"Comunalidad": np.round(communal,3)}))
        choice = st.radio("¿Cuál variable tiene la comunalidad más alta?", opts)
        if st.button("Confirmar Nivel 5"):
            pts = 10 if choice == correct_var else 0
            record_answer(5, choice, pts)
            if pts > 0:
                st.success("Thoth: Correcto — explica mucha varianza por los factores.")
            else:
                st.error(f"Thoth: No. La correcta era {correct_var}.")
            st.session_state.level = 6
            st.rerun()

# LEVEL 6 - Validation
elif level == 6:
    st.markdown("<div class='level-title'>Misión 6 — Cripta: KMO & Bartlett</div>", unsafe_allow_html=True)
    st.markdown("<div class='card'>Evalúa si la matriz es apta para AF: KMO y Bartlett.</div>", unsafe_allow_html=True)
    overall_kmo, per_item = kmo_test(df)
    chi2_stat, pval, dfree = bartlett_sphericity(df)
    st.metric("KMO global", round(overall_kmo,3) if not np.isnan(overall_kmo) else "N/A")
    st.write("KMO por variable (vista):")
    st.dataframe(per_item)
    st.write(f"Bartlett p-valor aproximado: {np.round(pval,4) if not np.isnan(pval) else 'N/A'}")
    choice = st.radio("¿Bartlett es significativo (p < 0.05)?", ["Sí","No"])
    if st.button("Confirmar Nivel 6"):
        correct = st.session_state.correct_dynamic[6]["answer"]
        pts = 10 if choice == correct else 0
        record_answer(6, choice, pts)
        if pts > 0:
            st.success("Thoth: La prueba respalda la idoneidad para AF.")
        else:
            st.error(f"Thoth: No es lo esperado en este pergamino (respuesta: {correct}).")
        st.session_state.level = 7
        st.rerun()

# LEVEL 7 - Mini-case selection
elif level == 7:
    st.markdown("<div class='level-title'>Misión 7 — Mini-caso: Selección Top-3 Factor 1</div>", unsafe_allow_html=True)
    st.markdown("<div class='card'>Escoge el conjunto de variables que mejor representa el Factor 1 (top-3).</div>", unsafe_allow_html=True)
    if "rotated_player" not in st.session_state:
        st.warning("Necesitas rotación previa.")
    else:
        top3_true = st.session_state.correct_dynamic[7]["answer"].split(", ")
        all_vars = list(st.session_state.loadings_player.index)
        set_A = top3_true
        set_B = safe_sample([v for v in all_vars if v not in set_A], k=3)
        set_C = safe_sample([v for v in all_vars if v not in set_A and v not in set_B], k=3)
        c1,c2,c3 = st.columns(3)
        with c1:
            st.write("A:", ", ".join(set_A))
        with c2:
            st.write("B:", ", ".join(set_B))
        with c3:
            st.write("C:", ", ".join(set_C))
        choice = st.radio("¿Qué conjunto eliges?", ["A","B","C"])
        if st.button("Confirmar Nivel 7"):
            correct_label = "A"
            pts = 12 if choice == correct_label else 0
            record_answer(7, choice, pts)
            if pts > 0:
                st.success("Thoth: Excelente selección.")
            else:
                st.error(f"Thoth: No era la mejor opción. La correcta era A: {', '.join(set_A)}")
            st.session_state.level = 8
            st.rerun()

# LEVEL 8 - Final summary
elif level == 8:
    st.markdown("<div class='level-title'>Misión 8 — Concilio: Resumen y retroalimentación</div>", unsafe_allow_html=True)
    st.markdown("<div class='card'>Revisa tus respuestas frente a las respuestas correctas generadas para esta partida.</div>", unsafe_allow_html=True)
    n_levels = 7
    df_results = pd.DataFrame({
        "Nivel": list(range(1, n_levels+1)),
        "Tu respuesta": st.session_state.user_answers,
        "Respuesta correcta (esta partida)": st.session_state.correct_answers,
        "Explicación (correcta)": st.session_state.explanations,
        "Puntos": st.session_state.points_per_level
    })
    st.table(df_results)
    total = sum(st.session_state.points_per_level)
    st.success(f"Puntuación total en esta partida: {total} puntos — XP acumulada: {st.session_state.xp}")
    st.markdown("### Síntesis pedagógica por nivel (repasa las explicaciones):")
    for i in range(1, n_levels+1):
        st.markdown(f"**Nivel {i}** — {st.session_state.correct_dynamic[i]['explanation']}")
    if st.button("🔁 Reiniciar partida (nuevo escenario y respuestas)"):
        player = st.session_state.player
        keys = ["scenario","started","level","user_answers","correct_answers","explanations","points_per_level","xp","fa_player","loadings_player","rotated_player","rot_choice","correct_dynamic"]
        for k in keys:
            if k in st.session_state:
                del st.session_state[k]
        st.session_state.player = player
        st.rerun()

# Fallback
else:
    st.warning("Estado inesperado — reiniciando.")
    start_new_game()
    st.rerun()
