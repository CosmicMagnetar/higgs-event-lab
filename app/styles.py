"""Scientific theme styling, color tokens, and custom CSS for HiggsLens dashboard.

Uses a light canvas, navy navigation and restrained teal, blue and amber accents.
Material Icons are loaded from Google Fonts CDN and used via <span class="material-icons">...</span>.
"""

from __future__ import annotations

# Shared UI and chart palette.
SURFACE_BASE = "#f6f8fa"
SURFACE_CARD = "#ffffff"
SURFACE_CARD_HOVER = "#eef5f6"
SURFACE_BORDER = "#dce4eb"
TEXT_PRIMARY = "#15283f"
TEXT_SECONDARY = "#475b70"
TEXT_MUTED = "#5f7082"
ACCENT_CYAN = "#127c80"
ACCENT_BLUE = "#4c79a4"
ACCENT_TEAL = "#16858a"
ACCENT_AMBER = "#b57b30"
ACCENT_RED = "#b33b43"
ACCENT_GREEN = "#19745c"
ACCENT_VIOLET = "#637687"
MPL_COLORS = ["#16858a", "#628bb5", "#b98645", "#537568", "#b66b66", "#6c8196", "#706d59"]

MATERIAL_ICONS_CSS = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
    '<link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@24,400,0,0" rel="stylesheet">'
    '<style>'
    '.material-icons { font-family: "Material Symbols Outlined"; font-weight: 400; font-size: 1.1rem; vertical-align: middle; }'
    '.icon-sm { font-size: 0.95rem; }'
    '.icon-md { font-size: 1.25rem; }'
    '</style>'
)

CUSTOM_CSS = """
<style>
:root { color-scheme: light; }
.stApp {
    background: #f6f8fa;
    color: #15283f;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
[data-testid="stMainBlockContainer"] { max-width: 1600px; padding: 2rem 2.2rem 3rem; }
[data-testid="stHeader"] { background: #f6f8fa; }
h1, h2, h3, h4 { color: #15283f; font-weight: 650; letter-spacing: -0.025em; }
h1 { font-size: 2rem; line-height: 1.2; }
h2 { font-size: 1.45rem; }
h3 { font-size: 1.15rem; margin-top: .35rem; }
p, li { line-height: 1.6; }
[data-testid="stCaptionContainer"] { color: #5f7082; }
section[data-testid="stSidebar"] { background: #142b43; border-right: 1px solid #233f57; }
section[data-testid="stSidebar"] [data-testid="stSidebarContent"] { padding-top: 1rem; }
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] { color: #d9e6ee; }
section[data-testid="stSidebar"] hr { border-color: #355065; }
section[data-testid="stSidebar"] { min-width: 290px; max-width: 340px; }
section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] { padding: 1.25rem 1rem 1.5rem; }
.sidebar-nav-label { color: #9db7ca; font-size: .68rem; font-weight: 650; letter-spacing: .12em; margin: .7rem .7rem .3rem; }
section[data-testid="stSidebar"] [role="radiogroup"] { gap: .35rem; }
section[data-testid="stSidebar"] [role="radiogroup"] label {
    box-sizing: border-box; min-height: 48px; width: 100%; margin: 0;
    padding: .65rem .75rem; gap: .75rem;
    border: 1px solid transparent; border-radius: 8px; color: #ccdae5;
    transition: background 150ms ease, border-color 150ms ease;
}
section[data-testid="stSidebar"] [role="radiogroup"] label:hover { background: #203e56; color: #fff; }
section[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
    background: #127c80; color: #fff; border-color: #279297; box-shadow: inset 3px 0 #8cdbd5;
}
section[data-testid="stSidebar"] [role="radiogroup"] label:has(input:focus-visible) { outline: 2px solid #b3e8e1; outline-offset: 2px; }
/* Keep the native radio input for keyboard navigation; replace its visual dot with an icon. */
section[data-testid="stSidebar"] [role="radiogroup"] label > div:first-of-type { position: absolute; width: 1px; height: 1px; overflow: hidden; clip-path: inset(50%); }
section[data-testid="stSidebar"] [role="radiogroup"] label::before {
    font-family: "Material Symbols Outlined"; font-size: 21px; line-height: 1;
    flex: 0 0 22px; font-weight: 400; color: #9fc6d9;
}
section[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked)::before { color: #fff; }
section[data-testid="stSidebar"] [role="radiogroup"] label:nth-of-type(1)::before { content: "home"; }
section[data-testid="stSidebar"] [role="radiogroup"] label:nth-of-type(2)::before { content: "science"; }
section[data-testid="stSidebar"] [role="radiogroup"] label:nth-of-type(3)::before { content: "database"; }
section[data-testid="stSidebar"] [role="radiogroup"] label:nth-of-type(4)::before { content: "bar_chart"; }
section[data-testid="stSidebar"] [role="radiogroup"] label:nth-of-type(5)::before { content: "verified_user"; }
section[data-testid="stSidebar"] [role="radiogroup"] label:nth-of-type(6)::before { content: "article"; }
section[data-testid="stSidebar"] [role="radiogroup"] label:nth-of-type(7)::before { content: "play_circle"; }
section[data-testid="stSidebar"] [role="radiogroup"] label p { color: inherit; font-size: .88rem; line-height: 1.4; font-weight: 500; }
section[data-testid="stSidebar"] [data-testid="stExpander"] { border: 1px solid #355065; border-radius: 8px; margin-top: .6rem; background: #18334b; }
section[data-testid="stSidebar"] [data-testid="stExpander"] summary,
section[data-testid="stSidebar"] [data-testid="stExpander"] summary p { color: #d9e6ee; }
section[data-testid="stSidebar"] [data-testid="stExpander"] [data-testid="stTooltipIcon"] { color: #bdcedc; }
section[data-testid="stSidebar"] .scientific-badge { background: #1c3e4a; border-color: #315a61; color: #c6eae2; text-align: left !important; font-weight: 500; font-size: .8rem; margin-bottom: .25rem !important; }
section[data-testid="stSidebar"] .scientific-badge-amber { background: #453c2c; color: #f3dbaf; border-color: #6b5736; }
section[data-testid="stSidebar"] .scientific-badge-cyan { background: #213e56; color: #d1e4f0; border-color: #3a5b74; }
section[data-testid="stSidebar"] [data-baseweb="input"] { background: #fff; color: #15283f; }
section[data-testid="stSidebar"] input { color: #15283f; caret-color: #127c80; }
section[data-testid="stSidebar"] .stButton button { background: #e7f4f3; color: #125e64; border-color: #b8dedc; }
section[data-testid="stSidebar"] .stButton button:hover { background: #fff; }
.sidebar-brand { padding: 0 .65rem 1rem; border-bottom: 1px solid #355065; margin-bottom: .8rem; }
.sidebar-brand h2 { color: #fff !important; font-size: 1.7rem; margin: 0; }
.sidebar-brand p { color: #c1d2df; font-size: .82rem; margin: .2rem 0 0; }
.sidebar-details { color: #bdcedc; font-size: .72rem; line-height: 1.9; padding: 0 .5rem; }
[data-testid="stMetric"] {
    background: white; border: 1px solid #dce4eb; border-radius: 10px;
    padding: 1rem; box-shadow: 0 2px 4px #142b4306; height: 100%;
}
[data-testid="stMetricLabel"] { color: #475b70; font-size: .85rem; }
[data-testid="stMetricValue"] { color: #15283f; font-size: 1.9rem; font-weight: 700; font-variant-numeric: tabular-nums; }
[data-testid="stMetric"]::after { content: ""; display: block; width: 28px; height: 3px; margin-top: .7rem; background: #16858a; border-radius: 2px; }
.scientific-panel, .chart-container, .evidence-card {
    background: #fff; border: 1px solid #dce4eb; border-radius: 10px;
    padding: 1.2rem; margin-bottom: .8rem; box-shadow: 0 2px 4px #142b4305;
}
.scientific-panel-cyan { background: #edf7f6; border: 1px solid #c8e4e1; border-left: 3px solid #16858a; border-radius: 8px; padding: 1rem 1.2rem; margin: 1rem 0; }
.evidence-card { min-height: 140px; height: 100%; margin-bottom: 0; }
.evidence-card .eyebrow { color: #127c80; font-size: .75rem; font-weight: 650; }
.evidence-card .value { color: #15283f; font-size: 1.2rem; font-weight: 650; margin: .4rem 0 .25rem; }
.evidence-card .detail { color: #475b70; font-size: .85rem; line-height: 1.5; }
.section-header { display: flex; align-items: center; gap: .6rem; margin-bottom: .6rem; }
.section-header h2 { margin: 0; }
.section-rule { border-top: 1px solid #dce4eb; margin: 1.3rem 0 .7rem; }
.hero-shell { margin-bottom: 1.2rem; }
.hero-title { font-size: clamp(1.65rem, 2.8vw, 2.25rem); font-weight: 700; line-height: 1.2; letter-spacing: -.035em; color: #15283f; margin-bottom: .45rem; }
.hero-copy { color: #475b70; font-size: 1rem; max-width: 850px; }
.hero-kicker { color: #127c80; font-size: .72rem; font-weight: 650; margin-bottom: .35rem; }
.proof-strip { display: flex; flex-wrap: wrap; gap: .65rem 1.2rem; border: 1px solid #c8e4e1; background: #eaf6f4; border-radius: 8px; padding: .7rem .9rem; margin: 1rem 0 .5rem; }
.proof-pill { color: #285762; font-size: .8rem; }
.proof-pill strong { color: #125e64; margin-right: .25rem; }
.scientific-badge { display: inline-block; border-radius: 7px; padding: .5rem .65rem; font-size: .8rem; font-weight: 600; border: 1px solid; }
.scientific-badge-green { background: #e7f4ef; color: #17634f; border-color: #c3dfd3; }
.scientific-badge-cyan { background: #e9f3f8; color: #285b76; border-color: #c9dfe9; }
.scientific-badge-amber { background: #fff4e4; color: #80551f; border-color: #ebd2ad; }
[data-testid="stDataFrame"], [data-testid="stImage"] { border: 1px solid #dce4eb; border-radius: 9px; background: #fff; overflow: hidden; }
[data-testid="stVerticalBlockBorderWrapper"] > div { border-color: #dce4eb; border-radius: 10px; }
.stButton button, .stDownloadButton button {
    border: 1px solid #b7c7d5; border-radius: 8px; background: #fff;
    color: #15283f; min-height: 44px; font-weight: 600;
    transition: background 150ms ease, border-color 150ms ease, transform 150ms ease;
}
.stButton button:hover, .stDownloadButton button:hover { background: #edf7f6; color: #125e64; border-color: #16858a; }
.stButton button:active, .stDownloadButton button:active { transform: translateY(1px); }
button:focus-visible, a:focus-visible, input:focus-visible { outline: 3px solid #16858a; outline-offset: 3px; }
button[data-baseweb="tab"] { color: #475b70; min-height: 44px; }
button[data-baseweb="tab"][aria-selected="true"] { color: #127c80; font-weight: 650; }
[data-baseweb="tab-highlight"] { background: #127c80; }
[data-testid="stFileUploader"] { border: 1px dashed #b7c7d5; border-radius: 8px; padding: .7rem; }
@media (max-width: 1100px) {
    [data-testid="stMainBlockContainer"] { padding: 1.5rem; }
    [data-testid="stMetricValue"] { font-size: 1.45rem; }
}
@media (max-width: 640px) {
    [data-testid="stMainBlockContainer"] { padding: 1rem; }
    [data-testid="stHorizontalBlock"] { flex-wrap: wrap; }
    [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] { width: 100% !important; flex: 1 1 100% !important; min-width: 0; }
    .proof-strip { flex-direction: column; }
    .evidence-card { min-height: auto; }
}
@media (prefers-reduced-motion: reduce) {
    .stButton button, .stDownloadButton button, section[data-testid="stSidebar"] label { transition: none; }
}
</style>
"""

def apply_theme() -> None:
    """Inject custom CSS and Material Icons into Streamlit application."""
    import streamlit as st
    st.markdown(MATERIAL_ICONS_CSS, unsafe_allow_html=True)
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
