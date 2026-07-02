import streamlit as st
import numpy as np
import pandas as pd
import re
import time
import random
from simple_model import load_model, Tokenizer, pad_sequences, SimpleTokenizer

# ─────────────────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="WordBloom · Next Word Predictor",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────
#  GLOBAL CSS  ·  Soft Pastel Dreamy Theme
# ─────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&family=Caveat:wght@500;600&display=swap');

:root {
    --bg:          #fcefff;   /* softer pinkish */
    --bg2:         #f7e6ff;   /* deeper pastel */
    --lavender:    #c9b8f5;
    --lavender-d:  #a78bfa;
    --peach:       #ffcba8;
    --peach-d:     #fb923c;
    --rose:        #fbbbd6;
    --rose-d:      #f472b6;
    --mint:        #b8f0e0;
    --mint-d:      #34d399;
    --sky:         #bae6fd;
    --sky-d:       #38bdf8;
    --text:        #3d2c5e;
    --text-soft:   #7c6899;
    --text-muted:  #b0a0c8;
    --card:        #fff6fb;   /* 🔥 key change (no more pure white) */
    --card-border: rgba(169,140,237,0.25);
    --shadow:      rgba(169,140,237,0.15);
}

html, body, .stApp {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif;
}

.stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    background:
        radial-gradient(ellipse 70% 55% at 15% -5%,  rgba(201,184,245,0.45) 0%, transparent 55%),
        radial-gradient(ellipse 55% 45% at 90% 100%, rgba(255,203,168,0.40) 0%, transparent 55%),
        radial-gradient(ellipse 45% 35% at 50% 50%,  rgba(251,187,214,0.20) 0%, transparent 65%),
        radial-gradient(ellipse 35% 30% at 80% 10%,  rgba(184,240,224,0.30) 0%, transparent 55%);
    pointer-events: none;
    z-index: 0;
    animation: dreamShift 10s ease-in-out infinite alternate;
}
@keyframes dreamShift {
    0%   { opacity: 0.75; transform: scale(1);    }
    100% { opacity: 1;    transform: scale(1.03); }
}

.petal {
    position: fixed;
    pointer-events: none;
    animation: floatPetal linear infinite;
    opacity: 0;
    z-index: 0;
}
@keyframes floatPetal {
    0%   { transform: translateY(110vh) rotate(0deg);   opacity: 0; }
    10%  { opacity: 0.55; }
    90%  { opacity: 0.35; }
    100% { transform: translateY(-10vh) rotate(360deg); opacity: 0; }
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #f3ebff 0%, #ffeef8 60%, #fff6ee 100%) !important;
    border-right: 1.5px solid rgba(201,184,245,0.5) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

#MainMenu, footer {
    visibility: hidden;
}

/* Keep header visible for sidebar toggle */
header {
    background: transparent !important;
}
.block-container { padding-top: 0.5rem !important; }

.hero { text-align: center; padding: 2.5rem 1rem 1.2rem; }
.hero-icon {
    font-size: 3.2rem;
    animation: iconFloat 3s ease-in-out infinite;
    display: block;
}
@keyframes iconFloat {
    0%, 100% { transform: translateY(0px);  }
    50%       { transform: translateY(-8px); }
}
.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: clamp(2rem, 5vw, 3.4rem);
    letter-spacing: -0.5px;
    margin: 0.3rem 0 0;
    background: linear-gradient(135deg, #8b5cf6 0%, #ec4899 50%, #f97316 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero-sub {
    font-family: 'Caveat', cursive;
    font-size: 1.25rem;
    color: var(--text-soft);
    margin-top: 0.3rem;
}

.petal-divider {
    display: flex; align-items: center; gap: 0.5rem;
    margin: 1.2rem 0; justify-content: center;
}
.petal-divider-line {
    flex: 1; height: 1.5px;
    background: linear-gradient(90deg, transparent, var(--lavender), var(--rose), var(--peach), transparent);
    border-radius: 4px;
}

.petal-card {
    background: var(--card);
    border: 1.5px solid var(--card-border);
    border-radius: 20px;
    padding: 1.6rem 1.8rem;
    box-shadow: 0 4px 24px var(--shadow);
    transition: transform 0.25s, box-shadow 0.25s;
    position: relative;
    overflow: hidden;
}
.petal-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, var(--lavender-d), var(--rose-d), var(--peach-d));
    border-radius: 20px 20px 0 0;
}
.petal-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 10px 36px rgba(167,139,250,0.22);
}

.info-card {
    background: linear-gradient(135deg, rgba(201,184,245,0.15), rgba(251,187,214,0.12));
    border: 1.5px solid rgba(201,184,245,0.35);
    border-radius: 16px;
    padding: 1.3rem 1.6rem;
}

.sec-label {
    font-family: 'Caveat', cursive;
    font-size: 1.1rem;
    color: var(--lavender-d);
    margin-bottom: 0.5rem;
    font-weight: 600;
}

.stTextInput > div > div > input {
    background: #fdf9ff !important;
    border: 1.5px solid rgba(167,139,250,0.4) !important;
    border-radius: 14px !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 1rem !important;
    padding: 0.7rem 1rem !important;
    box-shadow: 0 2px 8px rgba(167,139,250,0.1) !important;
    transition: border-color 0.25s, box-shadow 0.25s !important;
}
.stTextInput > div > div > input:focus {
    border-color: var(--lavender-d) !important;
    box-shadow: 0 0 0 3px rgba(167,139,250,0.2) !important;
}
.stTextInput > div > div > input::placeholder { color: var(--text-muted) !important; }

.stButton > button {
    background: linear-gradient(135deg, #c9b8f5, #fbbbd6) !important;
    border: none !important;
    border-radius: 14px !important;
    color: #4c1d95 !important;
    font-weight: 600 !important;
    font-size: 0.92rem !important;
    padding: 0.6rem 1.5rem !important;
    box-shadow: 0 4px 14px rgba(201,184,245,0.45) !important;
    transition: transform 0.2s, box-shadow 0.2s !important;
}
.stButton > button:hover {
    transform: translateY(-2px) scale(1.02) !important;
    box-shadow: 0 8px 22px rgba(201,184,245,0.6) !important;
    background: linear-gradient(135deg, #b5a1f3, #f9a8d4) !important;
}
.stButton > button:active { transform: scale(0.98) !important; }

[data-testid="metric-container"] {
    background: var(--card) !important;
    border: 1.5px solid rgba(201,184,245,0.4) !important;
    border-radius: 16px !important;
    padding: 1rem 1.3rem !important;
    box-shadow: 0 3px 14px var(--shadow) !important;
}

[data-testid="metric-container"] label {
    color: var(--text-soft) !important;
    font-size: 0.73rem !important;
    text-transform: uppercase;
    letter-spacing: 0.09em;
}

[data-testid="stMetricValue"] {
    color: var(--text) !important;
    font-size: 1.5rem !important;
    font-weight: 700 !important;
}

[data-baseweb="tab-list"] {
    background: rgba(201,184,245,0.15) !important;
    border-radius: 14px !important;
    padding: 4px !important;
    gap: 4px !important;
    border-bottom: none !important;
}
[data-baseweb="tab"] {
    border-radius: 10px !important;
    color: var(--text-soft) !important;
    font-weight: 500 !important;
    padding: 0.45rem 1.1rem !important;
    font-family: 'DM Sans', sans-serif !important;
}
[aria-selected="true"][data-baseweb="tab"] {
    background: var(--card) !important;
    color: var(--text) !important;
    box-shadow: 0 2px 10px rgba(167,139,250,0.25) !important;
}

.bloom-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    padding: 0.4rem 1rem;
    border-radius: 999px;
    font-size: 0.88rem;
    font-weight: 500;
    margin: 0.3rem 0.2rem;
    transition: transform 0.2s, box-shadow 0.2s;
    animation: pillBloom 0.5s cubic-bezier(0.34,1.56,0.64,1) both;
    border: 1.5px solid transparent;
    font-family: 'DM Sans', sans-serif;
    cursor: default;
}
.bloom-pill:hover {
    transform: translateY(-3px) scale(1.06);
    box-shadow: 0 6px 18px rgba(0,0,0,0.12);
}
@keyframes pillBloom {
    0%   { opacity: 0; transform: scale(0.5) rotate(-5deg); }
    100% { opacity: 1; transform: scale(1) rotate(0deg); }
}
.pill-lav   { background:rgba(201,184,245,0.35); border-color:#c9b8f5; color:#5b21b6; animation-delay:0.05s; }
.pill-rose  { background:rgba(251,187,214,0.35); border-color:#fbbbd6; color:#9d174d; animation-delay:0.10s; }
.pill-peach { background:rgba(255,203,168,0.40); border-color:#ffcba8; color:#9a3412; animation-delay:0.15s; }
.pill-mint  { background:rgba(184,240,224,0.40); border-color:#b8f0e0; color:#065f46; animation-delay:0.20s; }
.pill-sky   { background:rgba(186,230,253,0.40); border-color:#bae6fd; color:#0369a1; animation-delay:0.25s; }

.bloom-bar-wrap { margin: 0.45rem 0; }
.bloom-bar-label {
    display: flex; justify-content: space-between;
    font-size: 0.82rem; color: var(--text-soft);
    margin-bottom: 5px; font-weight: 500;
}
.bloom-bar-track {
    background: rgba(201,184,245,0.18);
    border-radius: 99px; height: 8px; overflow: hidden;
}
.bloom-bar-fill {
    height: 100%; border-radius: 99px;
    animation: barBloom 0.8s cubic-bezier(0.22,1,0.36,1) both;
}
@keyframes barBloom { from { width: 0 !important; } }

.bloom-text-box {
    background: linear-gradient(135deg, rgba(201,184,245,0.12), rgba(251,187,214,0.10));
    border: 1.5px solid rgba(201,184,245,0.45);
    border-radius: 18px;
    padding: 1.8rem 2rem 1.6rem 2.6rem;
    font-family: 'DM Serif Display', serif;
    font-style: italic;
    font-size: 1.25rem;
    color: var(--text);
    line-height: 1.85;
    animation: fadeUp 0.5s ease both;
    position: relative;
}
.bloom-text-box::before {
    content: '\201C';
    position: absolute;
    top: -0.2rem; left: 1rem;
    font-size: 4rem;
    color: rgba(201,184,245,0.55);
    font-family: 'DM Serif Display', serif;
    line-height: 1;
}
.blink {
    display: inline-block;
    width: 2.5px; height: 1.1em;
    background: var(--lavender-d);
    margin-left: 3px;
    vertical-align: text-bottom;
    border-radius: 2px;
    animation: blink 0.95s step-end infinite;
}
@keyframes blink { 0%,100%{opacity:1;} 50%{opacity:0;} }

.bloom-quote {
    background: var(--card);
    border-left: 3.5px solid var(--lavender);
    border-radius: 0 14px 14px 0;
    padding: 1rem 1.4rem;
    margin: 0.5rem 0;
    box-shadow: 0 2px 12px var(--shadow);
    animation: fadeUp 0.35s ease both;
    font-style: italic;
    color: var(--text);
    font-size: 0.93rem;
    line-height: 1.7;
}
.bloom-quote .q-author {
    font-style: normal;
    font-family: 'Caveat', cursive;
    font-size: 1rem;
    color: var(--peach-d);
    margin-top: 0.5rem;
}

.hist-chip {
    display: inline-block;
    background: rgba(201,184,245,0.2);
    border: 1px solid rgba(201,184,245,0.45);
    border-radius: 99px;
    padding: 0.2rem 0.75rem;
    font-size: 0.78rem;
    color: var(--text-soft);
    margin: 0.2rem;
    font-style: italic;
}

.step-row { display:flex; align-items:flex-start; gap:0.9rem; margin-bottom:1rem; }
.step-num {
    width:28px; height:28px; border-radius:50%;
    display:flex; align-items:center; justify-content:center;
    font-size:0.8rem; font-weight:700; flex-shrink:0; margin-top:1px;
}
.step-text { font-size:0.86rem; color:var(--text-soft); line-height:1.6; }
.step-title { font-weight:600; color:var(--text); font-size:0.9rem; }

@keyframes fadeUp {
    from { opacity:0; transform:translateY(10px); }
    to   { opacity:1; transform:translateY(0); }
}
</style>

<div class="petal" style="left:8%;animation-duration:14s;animation-delay:0s;font-size:1.1rem;">🌸</div>
<div class="petal" style="left:22%;animation-duration:18s;animation-delay:3s;font-size:0.9rem;">🌷</div>
<div class="petal" style="left:40%;animation-duration:12s;animation-delay:6s;font-size:1.3rem;">✿</div>
<div class="petal" style="left:60%;animation-duration:16s;animation-delay:1s;font-size:1rem;">🌺</div>
<div class="petal" style="left:76%;animation-duration:20s;animation-delay:4s;font-size:0.85rem;">🌸</div>
<div class="petal" style="left:91%;animation-duration:13s;animation-delay:8s;font-size:1.1rem;">🌷</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────
PILL_CLS   = ["pill-lav","pill-rose","pill-peach","pill-mint","pill-sky"]
BAR_COLORS = ["#a78bfa","#f472b6","#fb923c","#34d399","#38bdf8"]
W_EMOJI    = {"love":"💜","life":"🌿","time":"⏳","world":"🌍","mind":"🧠",
              "heart":"💗","people":"🫂","great":"⭐","never":"🚫",
              "always":"🔁","truth":"✨","beauty":"🌸","dream":"💭","light":"☀️"}

def clean(text):
    return re.sub(r'[^a-z\s]', '', text.lower()).strip()

@st.cache_resource(show_spinner=False)
def load_assets(csv_path, model_path, vocab_size=10_000):
    df  = pd.read_csv(csv_path)
    qs  = df["quote"].astype(str).str.lower().str.replace(r'[^a-z\s]','',regex=True)
    tok = Tokenizer(num_words=vocab_size)
    tok.fit_on_texts(qs)
    mdl = load_model(model_path, compile=False)
    try:    ml = mdl.input_shape[1] + 1
    except: ml = 20
    return tok, mdl, ml, df

def predict_topk(seed, tok, mdl, ml, k=5, temp=1.0):
    tokens = tok.texts_to_sequences([clean(seed)])[0][-(ml-1):]
    padded = pad_sequences([tokens], maxlen=ml-1, padding='pre')
    p      = mdl.predict(padded, verbose=0)[0]
    p      = np.log(p + 1e-10) / temp
    p      = np.exp(p) / np.sum(np.exp(p))
    top    = p.argsort()[-k:][::-1]
    i2w    = {v:k for k,v in tok.word_index.items()}
    return [(i2w.get(i,"<unk>"), float(p[i])) for i in top]

def generate(seed, tok, mdl, ml, n=10, temp=1.0):
    text = seed
    for _ in range(n):
        res = predict_topk(text, tok, mdl, ml, k=5, temp=temp)
        if not res: break
        ws, ps = zip(*res)
        text  += " " + random.choices(ws, weights=ps, k=1)[0]
    return text

def bar_html(word, prob, color, rank):
    emoji = W_EMOJI.get(word, "")
    pct   = round(prob * 100, 1)
    w     = max(5, min(100, pct * 2))
    return f"""
<div class='bloom-bar-wrap'>
  <div class='bloom-bar-label'>
    <span>{rank}. <strong>{word}</strong> {emoji}</span>
    <span style='color:{color};font-weight:700;'>{pct}%</span>
  </div>
  <div class='bloom-bar-track'>
    <div class='bloom-bar-fill' style='width:{w}%;background:linear-gradient(90deg,{color}88,{color});'></div>
  </div>
</div>"""


# ─────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:1.4rem 0 0.8rem;'>
      <span style='font-size:3rem;display:inline-block;animation:iconFloat 3s ease-in-out infinite;'>🌸</span>
      <div style='font-family:"DM Serif Display",serif;font-size:1.3rem;
                  background:linear-gradient(135deg,#8b5cf6,#ec4899);
                  -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                  background-clip:text;margin-top:0.4rem;'>
        WordBloom
      </div>
      <div style='font-family:"Caveat",cursive;font-size:0.95rem;color:#7c6899;'>
        next word · dreaming ahead 🌷
      </div>
    </div>
    <div style='height:1.5px;background:linear-gradient(90deg,transparent,#c9b8f5,#fbbbd6,transparent);
                margin:0.5rem 0 1rem;border-radius:4px;'></div>
    """, unsafe_allow_html=True)

    st.markdown("**🎛️ Prediction Settings**")
    top_k       = st.slider("🌸 Top-K Words",      1, 10, 5)
    temperature = st.slider("🌡️ Temperature",      0.1, 2.0, 1.0, 0.1,
                             help="Low = confident · High = creative 🎨")
    n_words     = st.slider("✍️ Generate Length",  5, 30, 10)

    st.markdown("""<div style='height:1.5px;background:linear-gradient(90deg,transparent,#c9b8f5,#fbbbd6,transparent);
                margin:0.8rem 0;border-radius:4px;'></div>""", unsafe_allow_html=True)

    st.markdown("**📂 File Paths**")
    csv_path   = st.text_input("CSV",   "qoute_dataset.csv")
    
    model_path = st.text_input("Model", "lstm_model.h5")

    st.markdown("""<div style='height:1.5px;background:linear-gradient(90deg,transparent,#c9b8f5,#fbbbd6,transparent);
                margin:0.8rem 0;border-radius:4px;'></div>""", unsafe_allow_html=True)

    st.markdown("""
    <div style='background:rgba(201,184,245,0.15);border-radius:14px;
                padding:1rem 1.1rem;font-size:0.8rem;line-height:1.9;color:#7c6899;'>
      <div style='font-weight:700;color:#5b21b6;'>🏗️ Architecture</div>
      Embedding → LSTM → Softmax Dense<br>
      <div style='font-weight:700;color:#9d174d;margin-top:0.5rem;'>📊 Dataset</div>
      3,038 quotes · 1,005 authors<br>~8,978 unique words<br>
      <div style='font-weight:700;color:#9a3412;margin-top:0.5rem;'>🔬 Training</div>
      100 epochs · Adam · CCE Loss
    </div>
    <div style='text-align:center;margin-top:0.9rem;font-family:"Caveat",cursive;
                font-size:0.95rem;color:#b0a0c8;'>
      made with 💜 · TensorFlow & Streamlit
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
#  LOAD MODEL
# ─────────────────────────────────────────────────────────
load_ok = False
try:
    with st.spinner("🌸 Waking up the model…"):
        tok, mdl, max_len, df = load_assets(csv_path, model_path)
    load_ok = True
except Exception as e:
    st.error(f"❌ Could not load files: `{e}`\n\nPlace **{csv_path}** and **{model_path}** next to **app.py** then run:\n```\nstreamlit run app.py\n```")


# ─────────────────────────────────────────────────────────
#  HERO
# ─────────────────────────────────────────────────────────
st.markdown("""
<div class='hero'>
  <span class='hero-icon'>🌸</span>
  <div class='hero-title'>WordBloom</div>
  <div class='hero-sub'>✨ LSTM-powered · next word prediction · trained on wisdom &amp; wonder ✨</div>
</div>
<div class='petal-divider'>
  <div class='petal-divider-line'></div>
  <span>🌷</span>
  <div class='petal-divider-line'></div>
</div>
""", unsafe_allow_html=True)


if load_ok:
    # ── METRICS ─────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    with m1: st.metric("🗂️ Quotes",      f"{len(df):,}")
    with m2: st.metric("📖 Vocabulary",  f"{len(tok.word_index):,}")
    with m3: st.metric("🔢 Seq Length",  f"{max_len - 1}")
    with m4: st.metric("🌡️ Temperature", f"{temperature}×")

    st.markdown("<br>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs([
        "🔮 Next Word Predictor",
        "✍️ Text Generator",
        "📚 Dataset Explorer",
    ])

    # ════════════════════════════════════════════
    #  TAB 1 — NEXT WORD PREDICTOR
    # ════════════════════════════════════════════
    with tab1:
        left, right = st.columns([3, 2], gap="large")

        with left:
            st.markdown("<div class='sec-label'>🌱 Type your seed phrase</div>", unsafe_allow_html=True)
            seed = st.text_input("seed", placeholder="e.g.  the greatest glory in …",
                                  label_visibility="collapsed", key="seed_input")

            b1, b2 = st.columns(2)
            with b1: predict_btn = st.button("🔮 Predict Next Words", use_container_width=True)
            with b2: rand_btn    = st.button("🎲 Random Seed",        use_container_width=True)

            if rand_btn:
                row  = df["quote"].dropna().sample(1).iloc[0]
                wds  = str(row).split()
                take = min(len(wds), random.randint(3, 6))
                st.session_state["_rseed"] = " ".join(wds[:take])
                st.rerun()

            if "_rseed" in st.session_state:
                seed = st.session_state.pop("_rseed")
                st.info(f"🎲 Random seed loaded: **{seed}**")

            if predict_btn and seed.strip():
                with st.spinner("🌸 Thinking…"):
                    time.sleep(0.25)
                    results = predict_topk(seed, tok, mdl, max_len, k=top_k, temp=temperature)

                st.markdown("<br>", unsafe_allow_html=True)

                # Confidence bars
                bars = "".join(bar_html(w, p, BAR_COLORS[i % 5], i+1)
                               for i, (w, p) in enumerate(results))
                st.markdown(f"""
                <div class='petal-card'>
                  <div class='sec-label' style='margin-bottom:1rem;'>📊 Prediction Confidence</div>
                  {bars}
                </div>""", unsafe_allow_html=True)

                # Pill tags
                st.markdown("<br><div class='sec-label'>💡 Top predictions</div>", unsafe_allow_html=True)
                pills = "".join(
                    f"<span class='bloom-pill {PILL_CLS[i%5]}'>{W_EMOJI.get(w,'🌸')} {w}</span>"
                    for i, (w, _) in enumerate(results)
                )
                st.markdown(f"<div>{pills}</div>", unsafe_allow_html=True)

                # Preview
                best = results[0][0] if results else ""
                st.markdown(f"""<br>
                <div class='bloom-text-box'>
                  {seed.strip()} <strong style='color:#8b5cf6;'>{best}</strong>
                  <span class='blink'></span>
                </div>""", unsafe_allow_html=True)

                if "pred_hist" not in st.session_state:
                    st.session_state.pred_hist = []
                st.session_state.pred_hist.insert(0, (seed, best))
                st.session_state.pred_hist = st.session_state.pred_hist[:8]

            elif predict_btn:
                st.warning("🌷 Please type a seed phrase first!")

        with right:
            st.markdown("""
            <div class='petal-card'>
              <div class='sec-label' style='margin-bottom:1rem;'>✨ How It Works</div>

              <div class='step-row'>
                <div class='step-num' style='background:rgba(201,184,245,0.4);color:#5b21b6;'>①</div>
                <div class='step-text'>
                  <div class='step-title'>Tokenization</div>
                  Your input is cleaned, lowercased and mapped to integer token IDs using the fitted Keras Tokenizer.
                </div>
              </div>
              <div class='step-row'>
                <div class='step-num' style='background:rgba(251,187,214,0.4);color:#9d174d;'>②</div>
                <div class='step-text'>
                  <div class='step-title'>Sequence Padding</div>
                  Tokens are pre-padded to match the model's expected input length.
                </div>
              </div>
              <div class='step-row'>
                <div class='step-num' style='background:rgba(255,203,168,0.4);color:#9a3412;'>③</div>
                <div class='step-text'>
                  <div class='step-title'>LSTM Inference</div>
                  Embedding → LSTM → Dense(Softmax) outputs a probability over all ~8,978 vocabulary words.
                </div>
              </div>
              <div class='step-row'>
                <div class='step-num' style='background:rgba(184,240,224,0.4);color:#065f46;'>④</div>
                <div class='step-text'>
                  <div class='step-title'>Temperature Scaling</div>
                  Low temperature = confident picks. High = creative diversity.
                </div>
              </div>
              <div class='step-row' style='margin-bottom:0;'>
                <div class='step-num' style='background:rgba(186,230,253,0.4);color:#0369a1;'>⑤</div>
                <div class='step-text'>
                  <div class='step-title'>Top-K Output</div>
                  The K most probable words are ranked with confidence scores and shown as blooms 🌸
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            if "pred_hist" in st.session_state and st.session_state.pred_hist:
                st.markdown("<br><div class='sec-label'>🕓 Recent Predictions</div>", unsafe_allow_html=True)
                chips = "".join(
                    f"<span class='hist-chip'>{s[:22]}… → <strong>{w}</strong></span>"
                    for s, w in st.session_state.pred_hist[:6]
                )
                st.markdown(chips, unsafe_allow_html=True)

    # ════════════════════════════════════════════
    #  TAB 2 — TEXT GENERATOR
    # ════════════════════════════════════════════
    with tab2:
        st.markdown("<div class='sec-label'>🌱 Start your sentence — the model will bloom the rest</div>",
                    unsafe_allow_html=True)

        g1, g2 = st.columns([4, 1])
        with g1:
            gen_seed = st.text_input("GenSeed",
                                      placeholder="e.g.  the only way to live is …",
                                      label_visibility="collapsed")
        with g2:
            gen_btn = st.button("🌸 Bloom!", use_container_width=True)

        if gen_btn and gen_seed.strip():
            with st.spinner("🌷 Growing your sentence…"):
                time.sleep(0.35)
                generated = generate(gen_seed.strip(), tok, mdl, max_len,
                                     n=n_words, temp=temperature)

            st.markdown(f"""<br>
            <div class='bloom-text-box'>
              {generated}<span class='blink'></span>
            </div>""", unsafe_allow_html=True)

            st.markdown("<br><div class='sec-label'>🔬 Word-by-word confidence</div>", unsafe_allow_html=True)
            new_words = generated[len(gen_seed):].strip().split()
            step_text = gen_seed.strip()
            bars_html = ""
            for i, w in enumerate(new_words[:n_words]):
                preds = predict_topk(step_text, tok, mdl, max_len, k=5, temp=temperature)
                if preds:
                    color = BAR_COLORS[i % 5]
                    prob  = next((p for ww, p in preds if ww == w), preds[0][1])
                    bars_html += bar_html(w, prob, color, i + 1)
                step_text += " " + w
            st.markdown(f"<div class='petal-card'>{bars_html}</div>", unsafe_allow_html=True)

            if "gen_hist" not in st.session_state:
                st.session_state.gen_hist = []
            st.session_state.gen_hist.insert(0, generated)
            st.session_state.gen_hist = st.session_state.gen_hist[:5]

        elif gen_btn:
            st.warning("🌷 Enter a seed phrase to bloom your text!")

        if "gen_hist" in st.session_state and st.session_state.gen_hist:
            st.markdown("<br><div class='sec-label'>📖 Previously Bloomed</div>", unsafe_allow_html=True)
            for txt in st.session_state.gen_hist:
                st.markdown(f"<div class='bloom-quote'>{txt}</div>", unsafe_allow_html=True)

    # ════════════════════════════════════════════
    #  TAB 3 — DATASET EXPLORER
    # ════════════════════════════════════════════
    with tab3:
        e1, e2 = st.columns([1, 2], gap="large")

        with e1:
            st.markdown("<div class='petal-card'>", unsafe_allow_html=True)
            st.markdown("<div class='sec-label'>🔍 Filter Quotes</div>", unsafe_allow_html=True)
            authors  = sorted(df["Author"].dropna().unique().tolist())
            sel_auth = st.selectbox("Author", ["🌸 All Authors"] + authors,
                                     label_visibility="collapsed")
            kw     = st.text_input("Keyword", placeholder="Search quotes…",
                                    label_visibility="collapsed")
            n_show = st.slider("Show rows", 3, 20, 6)
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown(f"""<br>
            <div class='info-card' style='font-size:0.82rem;color:#7c6899;line-height:1.9;'>
              🌸 <strong>{len(df):,}</strong> total quotes<br>
              🌷 <strong>{df['Author'].nunique()}</strong> unique authors<br>
              ✨ Vocab: ~8,978 words<br>
              💜 Trained end-to-end on quotes
            </div>""", unsafe_allow_html=True)

        with e2:
            filtered = df.copy()
            if sel_auth != "🌸 All Authors":
                filtered = filtered[filtered["Author"] == sel_auth]
            if kw.strip():
                filtered = filtered[filtered["quote"].str.contains(kw, case=False, na=False)]

            st.markdown(f"<div class='sec-label'>📋 {min(n_show, len(filtered))} of {len(filtered)} quotes</div>",
                        unsafe_allow_html=True)
            for _, row in filtered.head(n_show).iterrows():
                q = str(row["quote"])[:300]
                a = str(row.get("Author","Unknown"))
                st.markdown(f"""
                <div class='bloom-quote'>
                  {q}
                  <div class='q-author'>— {a}</div>
                </div>""", unsafe_allow_html=True)

            if filtered.empty:
                st.info("🌸 No quotes match — try a different filter.")

else:
    st.markdown("""
    <div class='petal-card' style='text-align:center;padding:3rem;max-width:520px;margin:2rem auto;'>
      <div style='font-size:3rem;'>🌸</div>
      <div style='font-family:"DM Serif Display",serif;font-size:1.3rem;color:#5b21b6;margin-top:0.8rem;'>
        Model files not found
      </div>
      <div style='color:#7c6899;margin-top:0.5rem;font-size:0.9rem;line-height:1.7;'>
        Place <code>qoute_dataset.csv</code> and <code>lstm_model.h5</code>
        in the same folder as <code>app.py</code>, then run:<br><br>
        <code style='color:#8b5cf6;'>streamlit run app.py</code>
      </div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────────────────
st.markdown("""
<div class='petal-divider' style='margin-top:2rem;'>
  <div class='petal-divider-line'></div>
  <span>🌸</span>
  <div class='petal-divider-line'></div>
</div>
<div style='text-align:center;font-family:"Caveat",cursive;font-size:1rem;
            color:#b0a0c8;padding-bottom:1.5rem;'>
  WordBloom 🌸 · LSTM Next Word Predictor · made with 💜 &amp; Streamlit
</div>
""", unsafe_allow_html=True)
