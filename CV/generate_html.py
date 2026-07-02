#!/usr/bin/env python3
"""
generate_html.py
Convert CV_Notes.md → CV_Notes.html

A self-contained, modern HTML build:
  • CommonMark + GFM rendering via markdown-it-py (matches an MD preview exactly —
    nested lists, tables, raw <img> HTML)
  • MathJax for $…$ / $$…$$  (consecutive "= …" lines auto-aligned)
  • GitHub-style heading IDs (so the hand-written TOC anchors resolve)
  • Per-section accent colours, collapsible sidebar, scroll-spy, dark mode,
    reading-progress bar
  • Images: filename shown on hover, fade-in size control, size persisted per image
"""

import re
import sys
import html as _html
import subprocess
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
MD_FILE    = SCRIPT_DIR / "CV_Notes.md"
OUT_FILE   = SCRIPT_DIR.parent / "CV_Notes.html"   # written to the repo root

# Site version — shown in the top bar. Bump on every push (see CHANGELOG.md);
# kept in sync with ML DL/generate_html.py's SITE_VERSION.
SITE_VERSION = "v1.11"

# Images live in this folder relative to the repo root (space → %20 for URLs).
IMG_BASE   = "CV"
# Cross-link to the sibling notes page (also at the repo root).
OTHER_HREF  = "ML_DL_Notes.html"
OTHER_LABEL = "ML/DL Notes ↗"

# ── Section palette  (num, accent, light-bg, dark-accent) ────────────────────
SECTIONS = [
    ( 0, "#64748B", "#F8FAFC", "#94A3B8"),  # TOC / misc
    ( 1, "#2563EB", "#EFF6FF", "#60A5FA"),  # Mathematical Foundations
    ( 2, "#059669", "#ECFDF5", "#34D399"),  # Statistical Inference
    ( 3, "#7C3AED", "#F5F3FF", "#A78BFA"),  # Core ML Concepts
    ( 4, "#16A34A", "#F0FDF4", "#4ADE80"),  # Feature Engineering
    ( 5, "#EA580C", "#FFF7ED", "#FB923C"),  # Classical ML Models
    ( 6, "#DB2777", "#FDF2F8", "#F472B6"),  # Ensemble Methods
    ( 7, "#DC2626", "#FEF2F2", "#F87171"),  # Regularization & Loss
    ( 8, "#9333EA", "#FAF5FF", "#C084FC"),  # Optimization
    ( 9, "#4F46E5", "#EEF2FF", "#818CF8"),  # Neural Networks
    (10, "#0891B2", "#ECFEFF", "#22D3EE"),  # Dimensionality Reduction
    (11, "#0D9488", "#F0FDFA", "#2DD4BF"),  # Sequential Models
    (12, "#15803D", "#F0FDF4", "#4ADE80"),  # NLP & Transformers
    (13, "#475569", "#F8FAFC", "#94A3B8"),  # Advanced Topics
    (14, "#B45309", "#FFFBEB", "#FBBF24"),  # Amazon Interview
    (15, "#64748B", "#F8FAFC", "#94A3B8"),  # References
]


# ── Equation alignment ───────────────────────────────────────────────────────
# Merge runs of single-line  $$ … $$  blocks (separated only by blank lines)
# where the continuation blocks start with a relational operator, so MathJax
# lines them up on the "=".

_BLOCK_RE  = re.compile(r'^\s*\$\$(.+?)\$\$\s*$')
_REL_START = re.compile(
    r'^\s*(=|\\leq|\\geq|\\approx|\\Rightarrow|\\rightarrow|\\to|\\equiv|\\propto|\\le|\\ge)'
)
_REL_OPS = ['=', r'\leq', r'\geq', r'\approx', r'\Rightarrow', r'\equiv', r'\propto']


def _amp_first(expr: str) -> str:
    """Insert an alignment '&' before the first relational operator."""
    best = None
    for op in _REL_OPS:
        idx = expr.find(op)
        if idx != -1 and (best is None or idx < best):
            best = idx
    if best is None:
        return expr
    return expr[:best] + '&' + expr[best:]


def merge_aligned_blocks(text: str) -> str:
    lines = text.split('\n')
    out, i, n = [], 0, len(text.split('\n'))
    while i < n:
        m = _BLOCK_RE.match(lines[i])
        if m:
            run = [m.group(1).strip()]
            j = i + 1
            while j < n:
                k = j
                while k < n and lines[k].strip() == '':
                    k += 1
                if k >= n:
                    break
                m2 = _BLOCK_RE.match(lines[k])
                if m2 and _REL_START.match(m2.group(1)):
                    run.append(m2.group(1).strip())
                    j = k + 1
                else:
                    break
            if len(run) > 1:
                body = [(_amp_first(e) if idx == 0 else '&' + e)
                        for idx, e in enumerate(run)]
                out.append('$$\\begin{aligned}\n'
                           + ' \\\\\n'.join(body)
                           + '\n\\end{aligned}$$')
                i = j
                continue
        out.append(lines[i])
        i += 1
    return '\n'.join(out)


# ── Math protection ───────────────────────────────────────────────────────────

def protect_math(text):
    """Replace $…$ and $$…$$ with plain-text placeholders markdown won't alter."""
    store = []

    def save(m):
        idx = len(store)
        store.append(m.group(0))
        return f"MLMATH{idx}MLEND"

    text = re.sub(r'\$\$[\s\S]*?\$\$', save, text)   # display first
    text = re.sub(r'\$[^\$\n]+?\$', save, text)       # then inline
    return text, store


def restore_math(html, store):
    def repl_block(m):
        return f'<div class="math-display">{store[int(m.group(1))]}</div>'

    html = re.sub(r'<p>\s*MLMATH(\d+)MLEND\s*</p>', repl_block, html)
    html = re.sub(r'MLMATH(\d+)MLEND', lambda m: store[int(m.group(1))], html)
    return html


# ── Heading IDs (GitHub-compatible slugify) ───────────────────────────────────

def _slug(text, used):
    s = text.strip().lower()
    s = re.sub(r'[^\w\s-]', '', s)   # drop punctuation, keep spaces/hyphens/word chars
    s = s.replace(' ', '-')          # each space -> one hyphen (no collapsing)
    base, c = s, 1
    while s in used:
        s, c = f'{base}-{c}', c + 1
    used.add(s)
    return s


def add_heading_ids(html):
    used = set()

    def repl(m):
        tag, attrs, inner = m.group(1), m.group(2), m.group(3)
        if re.search(r'\bid=', attrs):
            return m.group(0)
        text = _html.unescape(re.sub(r'<[^>]+>', '', inner))
        return f'<{tag}{attrs} id="{_slug(text, used)}">{inner}</{tag}>'

    return re.sub(r'<(h[123])((?:\s[^>]*)?)>(.*?)</\1>', repl, html, flags=re.S)


# ── Markdown conversion ───────────────────────────────────────────────────────

def md_to_html(md_file: Path) -> str:
    try:
        from markdown_it import MarkdownIt
    except ImportError:
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'markdown-it-py'],
                       check=True)
        from markdown_it import MarkdownIt

    text = md_file.read_text(encoding='utf-8')
    text = merge_aligned_blocks(text)
    text, store = protect_math(text)

    md = MarkdownIt('commonmark', {'html': True})
    for rule in ('table', 'strikethrough'):
        try:
            md.enable(rule)
        except Exception:
            pass
    html = md.render(text)

    html = restore_math(html, store)
    html = add_heading_ids(html)
    return html


# ── Section wrapping ──────────────────────────────────────────────────────────

def wrap_sections(body: str) -> str:
    """Split body HTML at <h2> boundaries and wrap each chunk in a coloured card.

    The leading H1 title block and the "Table of Contents" section are dropped
    from the rendered page — the top bar already shows the title and the sidebar
    is the table of contents. (The source MD keeps both.)
    """
    parts = re.split(r'(<h2[^>]*>[\s\S]*?</h2>)', body)

    result, open_sec = [], False

    for part in parts:
        m = re.match(r'<h2[^>]*>([\s\S]*?)</h2>', part)
        if m:
            if open_sec:
                result.append('</div>\n')
                open_sec = False

            heading_txt = re.sub(r'<[^>]+>', '', m.group(1)).strip()

            # Skip the Table of Contents section (the sidebar replaces it)
            if re.search(r'table of contents', heading_txt, re.I):
                continue

            num_m = re.match(r'^(\d+)\.', heading_txt)
            if num_m:
                sec_n = int(num_m.group(1))
            elif re.search(r'reference', heading_txt, re.I):
                sec_n = 15
            else:
                sec_n = 0

            open_sec = True
            result.append(f'<div class="content-section sec-{sec_n}">\n')
            result.append(part + '\n')
        else:
            # Keep only content inside an open section; the pre-H1 title block
            # and the TOC list (no section open) are dropped.
            if open_sec:
                result.append(part)

    if open_sec:
        result.append('</div>\n')

    return ''.join(result)


# ── CSS ───────────────────────────────────────────────────────────────────────

def gen_section_css() -> str:
    return "\n".join(
        f".sec-{n}{{--sa:{a};--sl:{l};--sd:{d};}}"
        for (n, a, l, d) in SECTIONS
    )


def get_css() -> str:
    return """
/* ─ Reset ───────────────────────────────────────────────── */
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}

/* ─ Variables ───────────────────────────────────────────── */
:root{
  --bg:#F1F5F9;--bg-card:#FFFFFF;--bg-sub:#F8FAFC;
  --text:#0F172A;--muted:#64748B;--border:#E6EAF0;
  --shadow:0 1px 2px rgba(16,24,40,.04),0 4px 16px rgba(16,24,40,.06);
  --shadow-lg:0 10px 30px rgba(16,24,40,.12);
  --sw:280px;--th:54px;--r:14px;
  --sa:#2563EB;--sl:#EFF6FF;--sd:#60A5FA;
  --font:'Inter',-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",sans-serif;
  --mono:'JetBrains Mono',Consolas,Monaco,"Courier New",monospace;
}
[data-theme=dark]{
  --bg:#0B1120;--bg-card:#131C2E;--bg-sub:#0F1830;
  --text:#E5ECF5;--muted:#93A4BD;--border:#24314A;
  --shadow:0 1px 2px rgba(0,0,0,.3),0 6px 20px rgba(0,0,0,.35);
  --shadow-lg:0 14px 40px rgba(0,0,0,.5);
}

/* ─ Base ────────────────────────────────────────────────── */
html{scroll-behavior:smooth;scroll-padding-top:66px}
body{font-family:var(--font);background:var(--bg);color:var(--text);
  line-height:1.72;font-size:15.5px;-webkit-font-smoothing:antialiased;
  transition:background .2s,color .2s}
a{color:var(--sa);text-decoration:none}
a:hover{text-decoration:underline}
p{margin:10px 0}
ul,ol{padding-left:24px;margin:10px 0}
li{margin:4px 0}
li>ul,li>ol{margin:4px 0}
strong{font-weight:700}
em{font-style:italic}
hr{border:none;border-top:1px solid var(--border);margin:20px 0}
h1{font-size:2rem;font-weight:800;letter-spacing:-.03em}

/* ─ Reading progress ────────────────────────────────────── */
#progress{position:fixed;top:var(--th);left:0;height:3px;width:0;z-index:199;
  background:linear-gradient(90deg,#2563EB,#7C3AED,#DB2777,#15803D);
  transition:width .08s linear}

/* ─ Top bar ─────────────────────────────────────────────── */
#topbar{position:fixed;top:0;left:0;right:0;height:var(--th);
  background:var(--bg-card);
  background:color-mix(in srgb,var(--bg-card) 84%,transparent);
  backdrop-filter:saturate(180%) blur(12px);
  border-bottom:1px solid var(--border);display:flex;align-items:center;gap:12px;
  padding:0 18px;z-index:200}
#tog-btn{font-size:1rem;background:none;border:1px solid var(--border);
  cursor:pointer;color:var(--muted);padding:6px 10px;border-radius:9px;line-height:1}
#tog-btn:hover{background:var(--bg-sub)}
#bar-title{font-weight:700;font-size:.95rem;color:var(--text);flex:1;
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis;letter-spacing:-.01em}
#bar-version{font-family:var(--mono);font-size:.68rem;color:var(--muted);
  background:var(--bg-sub);border:1px solid var(--border);border-radius:20px;
  padding:2px 9px;white-space:nowrap}
#dark-btn{font-size:.8rem;font-weight:600;background:none;border:1px solid var(--border);
  cursor:pointer;color:var(--muted);padding:6px 14px;border-radius:20px;white-space:nowrap}
#dark-btn:hover{background:var(--bg-sub)}
#cross-link{font-size:.8rem;font-weight:600;background:none;border:1px solid var(--border);
  cursor:pointer;color:var(--muted);padding:6px 14px;border-radius:20px;white-space:nowrap;
  text-decoration:none}
#cross-link:hover{background:var(--bg-sub);text-decoration:none}

/* ─ Sidebar ─────────────────────────────────────────────── */
#sidebar{position:fixed;top:var(--th);left:0;width:var(--sw);
  height:calc(100vh - var(--th));background:var(--bg-card);
  border-right:1px solid var(--border);overflow-y:auto;overflow-x:hidden;
  z-index:150;transition:transform .25s ease}
#sidebar.hidden{transform:translateX(-100%)}
#sidebar::-webkit-scrollbar{width:5px}
#sidebar::-webkit-scrollbar-thumb{background:var(--border);border-radius:3px}
#sb-inner{padding:14px 0 60px}

.sb-h2-btn{display:flex;align-items:center;gap:10px;width:100%;position:relative;
  padding:8px 16px;border:none;background:none;text-align:left;cursor:pointer;
  color:var(--text);font-weight:600;font-size:.82rem;transition:background .12s}
.sb-h2-btn:hover,.sb-h2-btn.active{background:var(--bg-sub)}
.sb-h2-btn.active::before{content:'';position:absolute;left:0;top:6px;bottom:6px;
  width:3px;border-radius:0 3px 3px 0;background:var(--ac,var(--sa))}
.sb-dot{width:22px;height:22px;border-radius:7px;display:flex;align-items:center;
  justify-content:center;color:#fff;font-size:.66rem;font-weight:800;flex-shrink:0}
.sb-label{flex:1;line-height:1.3}
.sb-chev{font-size:.6rem;color:var(--muted);transition:transform .2s;flex-shrink:0;
  padding:2px 4px;border-radius:5px}
.sb-chev:hover{background:var(--border)}
.sb-h2-btn.open .sb-chev{transform:rotate(90deg)}

.sb-subs{display:none;padding:2px 0 8px 42px}
.sb-subs.open{display:block}
.sb-h3-link{display:block;padding:5px 14px 5px 10px;color:var(--muted);
  font-size:.76rem;line-height:1.4;border-left:2px solid transparent;
  transition:color .12s,border-color .12s}
.sb-h3-link:hover{color:var(--text);text-decoration:none}
.sb-h3-link.active{color:var(--text);font-weight:600;border-left-color:var(--ac,var(--sa))}

/* ─ Main layout ─────────────────────────────────────────── */
#main-wrap{margin-left:var(--sw);padding-top:var(--th);transition:margin-left .25s ease}
#main-wrap.wide{margin-left:0}
#content{max-width:900px;margin:0 auto;padding:30px 26px 120px}

/* ─ Section cards ───────────────────────────────────────── */
.content-section{background:var(--bg-card);border:1px solid var(--border);
  border-radius:var(--r);margin-bottom:24px;box-shadow:var(--shadow);
  overflow:hidden;padding:0 24px 24px}
.content-section h2{font-size:1.4rem;font-weight:800;letter-spacing:-.02em;color:#fff;
  padding:15px 24px;margin:0 -24px 18px;line-height:1.3;display:flex;align-items:center;
  background:var(--sa);
  background:linear-gradient(120deg,var(--sa),color-mix(in srgb,var(--sa) 66%,#0B1020))}
.sec-num{display:inline-flex;align-items:center;justify-content:center;
  min-width:1.8em;height:1.8em;padding:0 .45em;margin-right:.6em;
  background:rgba(255,255,255,.22);border-radius:9px;font-size:.82em;font-weight:800}
.content-section h3{font-size:1.08rem;font-weight:700;letter-spacing:-.01em;
  color:var(--sa);margin:26px 0 12px;padding-bottom:6px;
  border-bottom:2px solid var(--sl);
  border-bottom:2px solid color-mix(in srgb,var(--sa) 24%,transparent)}
[data-theme=dark] .content-section h3{color:var(--sd);border-bottom-color:#24314A}

/* title card (h1 + preamble before first section) */
.title-card{padding:0 0 14px;border:none}
.title-card h1{padding:32px 28px;margin:0;color:#F8FAFC;font-size:2.1rem;
  font-weight:800;letter-spacing:-.035em;
  background:linear-gradient(135deg,#0F172A,#1E293B 55%,#334155)}
.title-card>:not(h1){margin-left:28px;margin-right:28px}
.title-card hr{display:none}

/* ─ Tables ──────────────────────────────────────────────── */
.content-section table{width:100%;border-collapse:separate;border-spacing:0;
  margin:16px 0;font-size:.9rem;border:1px solid var(--border);
  border-radius:10px;overflow:hidden;box-shadow:var(--shadow)}
table thead{background:var(--sa)}
table thead th{color:#fff;padding:11px 14px;text-align:left;font-weight:600;
  font-size:.78rem;letter-spacing:.04em;text-transform:uppercase}
table td,table th{padding:10px 14px;vertical-align:top;
  border-bottom:1px solid var(--border)}
table tbody tr:last-child td{border-bottom:none}
table tbody tr:nth-child(even){background:var(--bg-sub)}
table tbody tr:hover{background:color-mix(in srgb,var(--sa) 7%,var(--bg-card))}

/* ─ Code ────────────────────────────────────────────────── */
pre{background:#0D1426;color:#CBD5E1;padding:16px 18px;border-radius:12px;
  overflow-x:auto;font-family:var(--mono);font-size:.84rem;line-height:1.6;
  margin:16px 0;border:1px solid #1E293B}
[data-theme=dark] pre{background:#070D1A}
code{font-family:var(--mono);font-size:.86em;background:var(--bg-sub);
  color:var(--text);padding:1.5px 6px;border-radius:6px;border:1px solid var(--border)}
pre code{background:none;color:inherit;padding:0;border:none}

/* ─ Blockquotes ─────────────────────────────────────────── */
blockquote{border-left:4px solid var(--sa);
  background:var(--sl);
  background:color-mix(in srgb,var(--sa) 8%,var(--bg-card));
  padding:12px 16px;margin:14px 0;border-radius:0 10px 10px 0;font-size:.95rem}
[data-theme=dark] blockquote{background:rgba(148,163,184,.08)}

/* ─ Math ────────────────────────────────────────────────── */
.math-display{overflow-x:auto;margin:14px 0;padding:12px 18px;text-align:center;
  background:var(--bg-sub);border:1px solid var(--border);
  border-left:3px solid var(--sa);border-radius:10px}

/* ─ Images ──────────────────────────────────────────────── */
.img-fig{position:relative;margin:20px 0;text-align:center}
.img-fig img{display:block;margin:0 auto;height:auto;max-width:100%;
  border-radius:12px;border:1px solid var(--border);box-shadow:var(--shadow);
  transition:width .15s ease}
.img-bar{position:absolute;left:50%;bottom:10px;
  transform:translateX(-50%) translateY(6px);
  display:flex;align-items:center;gap:8px;flex-wrap:wrap;justify-content:center;
  background:rgba(8,12,22,.86);backdrop-filter:blur(8px);
  padding:6px 10px;border-radius:12px;border:1px solid rgba(255,255,255,.12);
  opacity:0;pointer-events:none;transition:opacity .18s,transform .18s;
  box-shadow:0 8px 24px rgba(0,0,0,.32);z-index:5}
.img-fig:hover .img-bar{opacity:1;pointer-events:auto;
  transform:translateX(-50%) translateY(0)}
.img-name{font-family:var(--mono);font-size:.7rem;color:#CBD5E1;
  background:rgba(255,255,255,.08);padding:2px 8px;border-radius:20px}
.img-slider{width:150px;cursor:pointer;accent-color:var(--sa)}
.img-pct{font-family:var(--mono);font-size:.7rem;color:#CBD5E1;min-width:38px;text-align:right}

/* ─ Back to top ─────────────────────────────────────────── */
#back-top{position:fixed;bottom:30px;right:24px;width:44px;height:44px;
  border:1px solid var(--border);border-radius:14px;background:var(--bg-card);
  color:var(--text);box-shadow:var(--shadow-lg);font-size:1.1rem;cursor:pointer;
  display:none;align-items:center;justify-content:center;z-index:300;
  transition:transform .15s}
#back-top:hover{transform:translateY(-2px)}
#back-top.show{display:flex}

/* ─ Responsive ──────────────────────────────────────────── */
@media(max-width:860px){
  #sidebar{transform:translateX(-100%)}
  #sidebar.mobile-open{transform:translateX(0);box-shadow:var(--shadow-lg)}
  #main-wrap{margin-left:0!important}
  #content{padding:20px 16px 100px}
}
""" + gen_section_css()


# ── JavaScript ────────────────────────────────────────────────────────────────

def get_js() -> str:
    colors_json = "{" + ",".join(
        f'"{n}":{{"accent":"{a}","light":"{l}","dark":"{d}"}}'
        for (n, a, l, d) in SECTIONS
    ) + "}"

    js = r'''
const SEC = __SEC_JSON__;

/* ── Dark mode ───────────────────────────────────────────── */
const darkBtn = document.getElementById('dark-btn');
const saved   = localStorage.getItem('theme') || 'light';
if (saved === 'dark') document.documentElement.setAttribute('data-theme','dark');
darkBtn.textContent = saved === 'dark' ? '☀  Light' : '🌙  Dark';
darkBtn.addEventListener('click', () => {
  const dark = document.documentElement.getAttribute('data-theme') === 'dark';
  document.documentElement.setAttribute('data-theme', dark ? 'light' : 'dark');
  darkBtn.textContent = dark ? '🌙  Dark' : '☀  Light';
  localStorage.setItem('theme', dark ? 'light' : 'dark');
});

/* ── Sidebar toggle ──────────────────────────────────────── */
const sidebar  = document.getElementById('sidebar');
const mainWrap = document.getElementById('main-wrap');
const togBtn   = document.getElementById('tog-btn');
togBtn.addEventListener('click', () => {
  if (window.matchMedia('(max-width:860px)').matches) {
    sidebar.classList.toggle('mobile-open');
  } else {
    sidebar.classList.toggle('hidden');
    mainWrap.classList.toggle('wide');
  }
});

/* ── Build sidebar ───────────────────────────────────────── */
function buildSidebar() {
  const inner = document.getElementById('sb-inner');
  inner.innerHTML = '';
  document.querySelectorAll('.content-section').forEach(sec => {
    const cls   = sec.className.match(/sec-(\d+)/);
    const secN  = cls ? +cls[1] : 0;
    const color = (SEC[secN] || SEC[0]).accent;
    const h2 = sec.querySelector('h2');
    if (!h2) return;

    const numM = h2.textContent.match(/^(\d+)\./);
    const chip = numM ? numM[1] : '';
    const h3s  = sec.querySelectorAll('h3');

    const secDiv = document.createElement('div');
    secDiv.className = 'sb-sec';
    secDiv.style.setProperty('--ac', color);

    const btn = document.createElement('button');
    btn.className = 'sb-h2-btn';
    btn.dataset.id = h2.id || '';
    btn.innerHTML = `
      <span class="sb-dot" style="background:${color}">${chip}</span>
      <span class="sb-label">${h2.textContent}</span>
      ${h3s.length ? '<span class="sb-chev">▶</span>' : ''}`;

    const subs = document.createElement('div');
    subs.className = 'sb-subs';
    h3s.forEach(h3 => {
      const a = document.createElement('a');
      a.className   = 'sb-h3-link';
      a.href        = '#' + (h3.id || '');
      a.textContent = h3.textContent;
      subs.appendChild(a);
    });

    const chev = btn.querySelector('.sb-chev');
    if (chev) {
      chev.addEventListener('click', e => {
        e.stopPropagation();
        subs.classList.toggle('open');
        btn.classList.toggle('open');
      });
    }
    btn.addEventListener('click', e => {
      if (e.target === chev) return;
      subs.classList.add('open');
      btn.classList.add('open');
      if (h2.id) document.getElementById(h2.id)
        ?.scrollIntoView({behavior:'smooth',block:'start'});
    });

    secDiv.appendChild(btn);
    secDiv.appendChild(subs);
    inner.appendChild(secDiv);
  });
}

/* ── Section-number chip in each h2 ──────────────────────── */
function decorateHeadings() {
  document.querySelectorAll('.content-section h2').forEach(h2 => {
    const m = h2.textContent.match(/^(\d+)\.\s*(.+)$/);
    if (m) h2.innerHTML = `<span class="sec-num">${m[1]}</span>${m[2]}`;
  });
}

/* ── Scroll spy ──────────────────────────────────────────── */
function initScrollSpy() {
  const heads = Array.from(document.querySelectorAll('h2[id],h3[id]'));
  let raf = false;
  function update() {
    const cut = window.scrollY + window.innerHeight * 0.22;
    let active = null;
    for (const h of heads) {
      if (h.getBoundingClientRect().top + window.scrollY <= cut) active = h;
      else break;
    }
    if (!active) return;
    document.querySelectorAll('.sb-h2-btn.active,.sb-h3-link.active')
      .forEach(el => el.classList.remove('active'));
    const id = active.id;
    if (active.tagName === 'H2') {
      document.querySelector(`.sb-h2-btn[data-id="${id}"]`)?.classList.add('active');
    } else {
      const link = document.querySelector(`.sb-h3-link[href="#${id}"]`);
      if (link) {
        link.classList.add('active');
        const par = link.closest('.sb-subs');
        if (par) {
          par.classList.add('open');
          par.previousElementSibling?.classList.add('open');
        }
      }
    }
  }
  window.addEventListener('scroll', () => {
    if (!raf) { raf = true; requestAnimationFrame(() => { update(); raf = false; }); }
  });
  update();
}

/* ── Images: hover filename + size control + persistence ─── */
function initImages() {
  /* Bump this when the default sizes change → clears stale saved sizes so the
     new defaults take effect (manual resizes made afterwards still persist). */
  const IMG_DEFAULTS_VERSION = '9';
  if (localStorage.getItem('imgDefaultsVersion') !== IMG_DEFAULTS_VERSION) {
    Object.keys(localStorage)
      .filter(k => k.indexOf('imgsize:') === 0)
      .forEach(k => localStorage.removeItem(k));
    localStorage.setItem('imgDefaultsVersion', IMG_DEFAULTS_VERSION);
  }

  /* Map each image to its subsection (nearest preceding h3). */
  const subOf = new Map();
  let curH3 = '';
  document.querySelectorAll('#content h3, #content img').forEach(el => {
    if (el.tagName === 'H3') curH3 = el.textContent.trim();
    else subOf.set(el, curH3);
  });

  document.querySelectorAll('#content img').forEach(img => {
    const src  = img.getAttribute('src') || '';
    const name = src.split('/').pop();
    const origW = parseInt(img.getAttribute('width')) || 700;
    img.removeAttribute('width');
    img.removeAttribute('height');
    img.title = name;
    img.style.maxWidth = '100%';   // never overflow the content column

    /* drop the filename badge that precedes the image */
    let prev = img.previousSibling;
    while (prev && prev.nodeType === 3 && !prev.textContent.trim())
      prev = prev.previousSibling;
    if (prev && prev.nodeType === 1 && prev.tagName === 'CODE'
        && /^image\d+\.(png|jpe?g|gif)$/i.test(prev.textContent.trim()))
      prev.remove();

    const fig = document.createElement('div');
    fig.className = 'img-fig';
    img.parentNode.insertBefore(fig, img);
    fig.appendChild(img);

    /* Default size = % of the image's natural width. Full (100%) by default;
       a few figures get fixed sizes, and §11.3 (Diagonal BiLSTM) → S (30%). */
    const stem     = name.replace(/\.(png|jpe?g|gif)$/i, '');
    const OVERRIDE = { image9: 60, image16: 140, image18: 140, image26: 130,
                       image40: 120, image50: 120, image52: 150, image55: 120,
                       image62: 75, image64: 75, image65: 120, image69: 120,
                       image70: 90, image81: 140 };
    const defW     = (stem in OVERRIDE) ? OVERRIDE[stem]
                   : (/^11\.3\b/.test(subOf.get(img) || '') ? 30 : 100);
    const KEY      = 'imgsize:' + name;
    const savedW   = parseInt(localStorage.getItem(KEY));
    const initW    = (savedW >= 15 && savedW <= 150) ? savedW : defW;
    const setWidth = w => { img.style.width = Math.round(origW * w / 100) + 'px'; };
    setWidth(initW);

    const bar = document.createElement('div');
    bar.className = 'img-bar';
    bar.innerHTML =
      `<span class="img-name">${name}</span>` +
      `<input type="range" class="img-slider" min="15" max="150" value="${initW}">` +
      `<span class="img-pct">${initW}%</span>`;
    fig.appendChild(bar);

    const slider = bar.querySelector('.img-slider');
    const pctEl  = bar.querySelector('.img-pct');
    function setW(w) {
      w = Math.max(15, Math.min(150, Math.round(w)));
      setWidth(w);
      slider.value = w;
      pctEl.textContent = w + '%';
      localStorage.setItem(KEY, w);
    }
    slider.addEventListener('input', () => setW(+slider.value));
  });
}

/* ── Reading progress ────────────────────────────────────── */
function initProgress() {
  const bar = document.getElementById('progress');
  function upd() {
    const h = document.documentElement;
    const max = h.scrollHeight - h.clientHeight;
    bar.style.width = (max > 0 ? (h.scrollTop / max) * 100 : 0) + '%';
  }
  window.addEventListener('scroll', upd);
  upd();
}

/* ── Back to top ─────────────────────────────────────────── */
const bt = document.getElementById('back-top');
window.addEventListener('scroll', () => bt.classList.toggle('show', window.scrollY > 350));
bt.addEventListener('click', () => window.scrollTo({top:0,behavior:'smooth'}));

/* ── Init ────────────────────────────────────────────────── */
document.addEventListener('DOMContentLoaded', () => {
  buildSidebar();
  decorateHeadings();
  initScrollSpy();
  initImages();
  initProgress();
});
'''
    return js.replace('__SEC_JSON__', colors_json)


# ── Assemble full HTML ────────────────────────────────────────────────────────

_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Computer Vision — Interview Notes</title>

<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">

<!-- MathJax: render $...$ and $$...$$ -->
<script>
MathJax = {
  tex: {
    inlineMath: [['$','$'],['\\\\(','\\\\)']],
    displayMath: [['$$','$$'],['\\\\[','\\\\]']],
    processEscapes: true,
    processEnvironments: true
  },
  options: { skipHtmlTags: ['script','noscript','style','textarea','pre'] }
};
</script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js" async></script>

<style>
__CSS__
</style>
</head>
<body>

<div id="topbar">
  <button id="tog-btn">☰</button>
  <span id="bar-title">Computer Vision — Interview Notes</span>
  <span id="bar-version">__VERSION__</span>
  __CROSSLINK__
  <button id="dark-btn">🌙  Dark</button>
</div>
<div id="progress"></div>

<nav id="sidebar"><div id="sb-inner"></div></nav>

<div id="main-wrap">
  <main id="content">
__BODY__
  </main>
</div>

<button id="back-top" title="Back to top">↑</button>

<script>
__JS__
</script>
</body>
</html>"""


def assemble(body: str) -> str:
    body = wrap_sections(body)
    html = _HTML_TEMPLATE
    html = html.replace('__CSS__', get_css())
    html = html.replace('__BODY__', body)
    html = html.replace('__JS__', get_js())
    html = html.replace('__CROSSLINK__',
                        f'<a id="cross-link" href="{OTHER_HREF}">{OTHER_LABEL}</a>')
    html = html.replace('__VERSION__', SITE_VERSION)
    # Page sits at the repo root; point images at the subfolder that holds them.
    html = html.replace('src="images/', f'src="{IMG_BASE}/images/')
    return html


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    print(f"Source : {MD_FILE}")
    print("Converting markdown (markdown-it-py) …")
    body = md_to_html(MD_FILE)

    print("Assembling HTML …")
    html = assemble(body)

    OUT_FILE.write_text(html, encoding='utf-8')
    kb = OUT_FILE.stat().st_size // 1024
    print(f"Output : {OUT_FILE}  ({kb} KB)")
    print("Done — open CV_Notes.html in your browser.")


if __name__ == '__main__':
    main()
