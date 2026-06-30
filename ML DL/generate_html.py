#!/usr/bin/env python3
"""
generate_html.py
Convert ML_DL_Notes.md → ML_DL_Notes.html
A beautiful, self-contained HTML with colored sections, MathJax, image sliders,
collapsible sidebar, dark mode, and scroll spy.
"""

import re
import sys
import subprocess
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
MD_FILE    = SCRIPT_DIR / "ML_DL_Notes.md"
OUT_FILE   = SCRIPT_DIR / "ML_DL_Notes.html"

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


# ── Math protection ───────────────────────────────────────────────────────────

def protect_math(text):
    """Replace $...$ and $$...$$ with plain-text placeholders markdown won't alter."""
    store = []

    def save(m):
        idx = len(store)
        store.append(m.group(0))
        return f"MLMATH{idx}MLEND"

    # Display math first (allow newlines inside)
    text = re.sub(r'\$\$[\s\S]*?\$\$', save, text)
    # Inline math (no newlines)
    text = re.sub(r'\$[^\$\n]+?\$', save, text)
    return text, store


def restore_math(html, store):
    # Display math: markdown wraps the lone placeholder in <p>…</p>; replace whole <p>
    def repl_block(m):
        expr = store[int(m.group(1))]
        return f'<div class="math-display">{expr}</div>'

    html = re.sub(r'<p>\s*MLMATH(\d+)MLEND\s*</p>', repl_block, html)

    # Inline math (placeholder still inside other tags)
    html = re.sub(r'MLMATH(\d+)MLEND', lambda m: store[int(m.group(1))], html)
    return html


# ── Markdown conversion ───────────────────────────────────────────────────────

def md_to_html(md_file: Path) -> str:
    try:
        import markdown
    except ImportError:
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'markdown'], check=True)
        import markdown

    text = md_file.read_text(encoding='utf-8')
    text, store = protect_math(text)

    html = markdown.markdown(
        text,
        extensions=['extra', 'toc'],
        extension_configs={
            'toc': {'permalink': False}
        }
    )

    html = restore_math(html, store)
    return html


# ── Section wrapping ──────────────────────────────────────────────────────────

def wrap_sections(body: str) -> str:
    """Split body HTML at <h2> boundaries and wrap each chunk in a coloured section div."""
    # Split while keeping the delimiters
    parts = re.split(r'(<h2[^>]*>[\s\S]*?</h2>)', body)

    result = []
    pre_buf = []          # content before first h2
    open_sec = False

    for part in parts:
        m = re.match(r'<h2[^>]*>([\s\S]*?)</h2>', part)
        if m:
            # Close previous section
            if open_sec:
                result.append('</div>\n')
            elif pre_buf:
                # Wrap pre-h2 content (h1 + hr)
                result.append('<div class="content-section sec-0 title-card">\n')
                result.extend(pre_buf)
                result.append('</div>\n')
                pre_buf = []

            # Determine colour class from section number
            heading_txt = re.sub(r'<[^>]+>', '', m.group(1)).strip()
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
            if open_sec:
                result.append(part)
            else:
                pre_buf.append(part)

    if open_sec:
        result.append('</div>\n')
    elif pre_buf:
        result.extend(pre_buf)

    return ''.join(result)


# ── CSS ───────────────────────────────────────────────────────────────────────

def gen_section_css() -> str:
    lines = []
    for (n, accent, light, dark_a) in SECTIONS:
        lines.append(f"""
.sec-{n} h2 {{ background: {accent}; }}
.sec-{n} h3 {{ color: {accent}; border-bottom-color: {light}; }}
.sec-{n} table thead {{ background: {accent}; }}
.sec-{n} blockquote {{ border-left-color: {accent}; background: {light}; }}
.sec-{n} .img-ctrl input[type=range] {{ accent-color: {accent}; }}
.sec-{n} a {{ color: {accent}; }}
[data-theme=dark] .sec-{n} h3 {{ color: {dark_a}; border-bottom-color: #1e293b; }}
[data-theme=dark] .sec-{n} blockquote {{ background: rgba(30,41,59,.7); border-left-color: {dark_a}; }}
[data-theme=dark] .sec-{n} a {{ color: {dark_a}; }}""")
    return "\n".join(lines)


def get_css() -> str:
    return """
/* ─ Reset ──────────────────────────────────────────────── */
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}

/* ─ Variables ──────────────────────────────────────────── */
:root{
  --bg:#F1F5F9;--bg-card:#FFF;--bg-sub:#F8FAFC;
  --text:#0F172A;--muted:#64748B;--border:#E2E8F0;
  --shadow:0 1px 4px rgba(0,0,0,.07),0 2px 12px rgba(0,0,0,.05);
  --sw:264px;--th:52px;--r:10px;
  --font:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",sans-serif;
  --mono:"JetBrains Mono",Consolas,Monaco,"Courier New",monospace;
}
[data-theme=dark]{
  --bg:#0F172A;--bg-card:#1E293B;--bg-sub:#162032;
  --text:#E2E8F0;--muted:#94A3B8;--border:#334155;
  --shadow:0 2px 8px rgba(0,0,0,.4);
}

/* ─ Base ────────────────────────────────────────────────── */
html{scroll-behavior:smooth}
body{font-family:var(--font);background:var(--bg);color:var(--text);
  line-height:1.75;font-size:15.5px;transition:background .2s,color .2s}
a{text-decoration:none}
a:hover{text-decoration:underline}
p{margin:8px 0}
ul,ol{padding-left:22px;margin:8px 0}
li{margin:3px 0}
strong{font-weight:600}
hr{border:none;border-top:1px solid var(--border);margin:18px 0}
h1{font-size:1.9rem;font-weight:800;letter-spacing:-.04em;margin-bottom:4px}

/* ─ Top bar ─────────────────────────────────────────────── */
#topbar{
  position:fixed;top:0;left:0;right:0;height:var(--th);
  background:var(--bg-card);border-bottom:1px solid var(--border);
  display:flex;align-items:center;gap:10px;padding:0 16px;
  z-index:200;box-shadow:var(--shadow);
}
#tog-btn{
  font-size:1rem;background:none;border:1px solid var(--border);
  cursor:pointer;color:var(--muted);padding:5px 9px;border-radius:6px;line-height:1;
}
#tog-btn:hover{background:var(--bg-sub)}
#bar-title{font-weight:700;font-size:.95rem;color:var(--text);flex:1;
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
#dark-btn{
  font-size:.8rem;background:none;border:1px solid var(--border);
  cursor:pointer;color:var(--muted);padding:4px 12px;border-radius:20px;white-space:nowrap;
}
#dark-btn:hover{background:var(--bg-sub)}

/* ─ Sidebar ──────────────────────────────────────────────── */
#sidebar{
  position:fixed;top:var(--th);left:0;width:var(--sw);
  height:calc(100vh - var(--th));background:var(--bg-card);
  border-right:1px solid var(--border);overflow-y:auto;overflow-x:hidden;
  z-index:150;transition:transform .25s ease;
}
#sidebar.hidden{transform:translateX(calc(-1 * var(--sw)))}
#sidebar::-webkit-scrollbar{width:4px}
#sidebar::-webkit-scrollbar-thumb{background:var(--border);border-radius:2px}
#sb-inner{padding:10px 0 50px}

.sb-sec{}
.sb-h2-btn{
  display:flex;align-items:center;gap:9px;
  width:100%;padding:7px 14px;border:none;background:none;
  text-align:left;cursor:pointer;color:var(--text);
  font-weight:600;font-size:.83rem;transition:background .12s;
}
.sb-h2-btn:hover,.sb-h2-btn.active{background:var(--bg-sub)}
.sb-dot{width:9px;height:9px;border-radius:50%;flex-shrink:0}
.sb-label{flex:1;line-height:1.3}
.sb-chev{font-size:.65rem;color:var(--muted);transition:transform .2s;flex-shrink:0}
.sb-h2-btn.open .sb-chev{transform:rotate(90deg)}

.sb-subs{display:none;padding-left:32px}
.sb-subs.open{display:block}
.sb-h3-link{
  display:block;padding:4px 14px 4px 0;
  color:var(--muted);font-size:.77rem;line-height:1.4;transition:color .1s;
}
.sb-h3-link:hover{color:var(--text);text-decoration:none}
.sb-h3-link.active{color:var(--text);font-weight:600}

/* ─ Main layout ──────────────────────────────────────────── */
#main-wrap{margin-left:var(--sw);padding-top:var(--th);transition:margin-left .25s ease}
#main-wrap.wide{margin-left:0}
#content{max-width:880px;margin:0 auto;padding:28px 22px 100px}

/* ─ Section cards ────────────────────────────────────────── */
.content-section{
  background:var(--bg-card);border-radius:var(--r);
  margin-bottom:26px;box-shadow:var(--shadow);overflow:hidden;
  padding:0 22px 22px;
}
.content-section h2{
  font-size:1.42rem;font-weight:700;letter-spacing:-.025em;
  color:#fff;padding:14px 22px;margin-bottom:16px;
  position:relative;left:-22px;width:calc(100% + 44px);line-height:1.3;
}
.content-section h3{
  font-size:1.05rem;font-weight:600;padding-bottom:5px;
  margin-top:22px;margin-bottom:10px;
  border-bottom-width:2px;border-bottom-style:solid;
}

/* title card (h1 + hr before first section) */
.title-card{padding:0}
.title-card h1{
  padding:22px 24px 18px;
  background:linear-gradient(135deg,#0F172A,#1E293B);
  color:#F8FAFC;border-radius:var(--r) var(--r) 0 0;
}
.title-card hr{display:none}

/* ─ Tables ───────────────────────────────────────────────── */
table{
  width:100%;border-collapse:collapse;margin:14px 0;font-size:.9rem;
  border-radius:8px;overflow:hidden;box-shadow:0 1px 3px rgba(0,0,0,.08);
}
table thead{color:#fff}
table thead th{
  padding:10px 14px;text-align:left;font-weight:600;
  font-size:.8rem;letter-spacing:.03em;text-transform:uppercase;
}
table td{
  padding:9px 14px;border-bottom:1px solid var(--border);vertical-align:top;
}
table tr:last-child td{border-bottom:none}
table tbody tr:nth-child(even){background:var(--bg-sub)}
table tbody tr:hover{filter:brightness(.97)}
[data-theme=dark] table tbody tr:hover{filter:brightness(1.08)}

/* ─ Code ─────────────────────────────────────────────────── */
pre{
  background:#1E1E2E;color:#CDD6F4;padding:16px 20px;border-radius:8px;
  overflow-x:auto;font-family:var(--mono);font-size:.84rem;
  line-height:1.65;margin:14px 0;
}
code{
  font-family:var(--mono);font-size:.855em;
  background:var(--bg-sub);color:var(--muted);
  padding:1px 5px;border-radius:4px;border:1px solid var(--border);
}
pre code{background:none;color:inherit;padding:0;border:none}
code.img-badge{
  display:inline-block;background:var(--bg-sub);color:var(--muted);
  font-size:.71rem;padding:2px 9px;border-radius:20px;
  border:1px solid var(--border);margin-bottom:6px;
}

/* ─ Blockquotes ──────────────────────────────────────────── */
blockquote{
  border-left-width:4px;border-left-style:solid;
  padding:10px 16px;margin:12px 0;border-radius:0 6px 6px 0;
  font-style:italic;font-size:.95rem;
}

/* ─ Math ─────────────────────────────────────────────────── */
.math-display{
  overflow-x:auto;padding:4px 0;text-align:center;margin:10px 0;
}

/* ─ Images ───────────────────────────────────────────────── */
.img-wrap{text-align:center;margin:16px 0}
.img-wrap img{
  display:block;margin:0 auto;max-width:100%;
  border-radius:6px;height:auto;
}
.img-ctrl{
  display:flex;align-items:center;justify-content:center;
  gap:8px;margin-top:8px;font-size:.78rem;color:var(--muted);
}
.img-ctrl input[type=range]{width:110px;cursor:pointer}
.img-ctrl .pct{min-width:34px;text-align:center;font-weight:600}
.img-ctrl button{
  background:none;border:1px solid var(--border);border-radius:4px;
  padding:1px 8px;cursor:pointer;font-size:.72rem;color:var(--muted);
}
.img-ctrl button:hover{background:var(--bg-sub)}

/* ─ Back to top ──────────────────────────────────────────── */
#back-top{
  position:fixed;bottom:28px;right:22px;
  background:#334155;color:#fff;border:none;border-radius:50%;
  width:40px;height:40px;font-size:1.1rem;cursor:pointer;
  display:none;z-index:300;box-shadow:0 3px 10px rgba(0,0,0,.25);
  align-items:center;justify-content:center;transition:background .15s;
}
#back-top:hover{background:#475569}
#back-top.show{display:flex}

/* ─ Responsive ───────────────────────────────────────────── */
@media(max-width:768px){
  #sidebar{transform:translateX(calc(-1 * var(--sw)))}
  #sidebar.mobile-open{transform:translateX(0)}
  #main-wrap{margin-left:0!important}
}
""" + gen_section_css()


# ── JavaScript ────────────────────────────────────────────────────────────────

def get_js() -> str:
    colors_json = "{" + ",".join(
        f'"{n}":{{"accent":"{a}","light":"{l}","dark":"{d}"}}'
        for (n, a, l, d) in SECTIONS
    ) + "}"

    return f"""
const SEC = {colors_json};

/* ── Dark mode ──────────────────────────────────────────────────────────── */
const darkBtn = document.getElementById('dark-btn');
const saved   = localStorage.getItem('theme') || 'light';
if (saved === 'dark') document.documentElement.setAttribute('data-theme','dark');
darkBtn.textContent = saved === 'dark' ? '☀️  Light' : '🌙  Dark';

darkBtn.addEventListener('click', () => {{
  const dark = document.documentElement.getAttribute('data-theme') === 'dark';
  document.documentElement.setAttribute('data-theme', dark ? 'light' : 'dark');
  darkBtn.textContent = dark ? '🌙  Dark' : '☀️  Light';
  localStorage.setItem('theme', dark ? 'light' : 'dark');
}});

/* ── Sidebar toggle ─────────────────────────────────────────────────────── */
const sidebar  = document.getElementById('sidebar');
const mainWrap = document.getElementById('main-wrap');
const togBtn   = document.getElementById('tog-btn');

togBtn.addEventListener('click', () => {{
  sidebar.classList.toggle('hidden');
  mainWrap.classList.toggle('wide');
}});

/* ── Build sidebar ──────────────────────────────────────────────────────── */
function buildSidebar() {{
  const inner = document.getElementById('sb-inner');
  inner.innerHTML = '';

  document.querySelectorAll('.content-section').forEach(sec => {{
    const cls   = sec.className.match(/sec-(\\d+)/);
    const secN  = cls ? +cls[1] : 0;
    const color = (SEC[secN] || SEC[0]).accent;

    const h2 = sec.querySelector('h2');
    if (!h2) return;

    const h3s = sec.querySelectorAll('h3');

    const secDiv = document.createElement('div');
    secDiv.className = 'sb-sec';

    /* H2 button */
    const btn = document.createElement('button');
    btn.className = 'sb-h2-btn';
    btn.dataset.id = h2.id || '';
    btn.innerHTML = `
      <span class="sb-dot" style="background:${{color}}"></span>
      <span class="sb-label">${{h2.textContent}}</span>
      ${{h3s.length ? '<span class="sb-chev">▶</span>' : ''}}`;

    /* Subsections */
    const subs = document.createElement('div');
    subs.className = 'sb-subs';

    h3s.forEach(h3 => {{
      const a = document.createElement('a');
      a.className   = 'sb-h3-link';
      a.href        = '#' + (h3.id || '');
      a.textContent = h3.textContent;
      subs.appendChild(a);
    }});

    /* Chevron toggles subs */
    const chev = btn.querySelector('.sb-chev');
    if (chev) {{
      chev.addEventListener('click', e => {{
        e.stopPropagation();
        subs.classList.toggle('open');
        btn.classList.toggle('open');
      }});
    }}

    /* Button click → scroll to h2 + expand */
    btn.addEventListener('click', e => {{
      if (e.target === chev) return;
      subs.classList.add('open');
      btn.classList.add('open');
      if (h2.id) document.getElementById(h2.id)
        ?.scrollIntoView({{behavior:'smooth',block:'start'}});
    }});

    secDiv.appendChild(btn);
    secDiv.appendChild(subs);
    inner.appendChild(secDiv);
  }});
}}

/* ── Scroll spy ─────────────────────────────────────────────────────────── */
function initScrollSpy() {{
  const heads = Array.from(document.querySelectorAll('h2[id],h3[id]'));
  let raf = false;

  function update() {{
    const cut = window.scrollY + window.innerHeight * 0.22;
    let active = null;
    for (const h of heads) {{
      if (h.getBoundingClientRect().top + window.scrollY <= cut) active = h;
      else break;
    }}
    if (!active) return;

    document.querySelectorAll('.sb-h2-btn.active,.sb-h3-link.active')
      .forEach(el => el.classList.remove('active'));

    const id = active.id;
    if (active.tagName === 'H2') {{
      document.querySelector(`.sb-h2-btn[data-id="${{id}}"]`)?.classList.add('active');
    }} else {{
      const link = document.querySelector(`.sb-h3-link[href="#${{id}}"]`);
      if (link) {{
        link.classList.add('active');
        const par = link.closest('.sb-subs');
        if (par) {{
          par.classList.add('open');
          par.previousElementSibling?.classList.add('open');
        }}
      }}
    }}
  }}

  window.addEventListener('scroll', () => {{
    if (!raf) {{ raf = true; requestAnimationFrame(() => {{ update(); raf = false; }}); }}
  }});
  update();
}}

/* ── Image resize ───────────────────────────────────────────────────────── */
function initImages() {{
  document.querySelectorAll('img').forEach(img => {{
    const origW = parseInt(img.getAttribute('width')) || 700;
    img.removeAttribute('width');
    img.removeAttribute('height');
    Object.assign(img.style, {{
      display:'block', margin:'0 auto', maxWidth: origW + 'px', width:'70%',
      borderRadius:'6px', height:'auto'
    }});

    const wrap = document.createElement('div');
    wrap.className = 'img-wrap';
    img.parentNode.insertBefore(wrap, img);
    wrap.appendChild(img);

    const ctrl    = document.createElement('div');
    ctrl.className = 'img-ctrl';

    const lbl = document.createElement('span');
    lbl.textContent = 'Size:';

    const slider = document.createElement('input');
    slider.type = 'range'; slider.min = '10'; slider.max = '100'; slider.value = '70';

    const pct = document.createElement('span');
    pct.className = 'pct'; pct.textContent = '70%';

    const rst = document.createElement('button');
    rst.textContent = '↺ Reset';

    slider.addEventListener('input', () => {{
      img.style.width = slider.value + '%'; pct.textContent = slider.value + '%';
    }});
    rst.addEventListener('click', () => {{
      slider.value = '70'; img.style.width = '70%'; pct.textContent = '70%';
    }});

    ctrl.append(lbl, slider, pct, rst);
    wrap.appendChild(ctrl);
  }});
}}

/* ── Image filename badges ──────────────────────────────────────────────── */
function badgeImageLabels() {{
  document.querySelectorAll('code').forEach(c => {{
    if (/^image\\d+\\.(png|jpg|gif)$/i.test(c.textContent.trim()))
      c.classList.add('img-badge');
  }});
}}

/* ── Back to top ────────────────────────────────────────────────────────── */
const bt = document.getElementById('back-top');
window.addEventListener('scroll', () => bt.classList.toggle('show', window.scrollY > 350));
bt.addEventListener('click', () => window.scrollTo({{top:0,behavior:'smooth'}}));

/* ── Init ────────────────────────────────────────────────────────────────── */
document.addEventListener('DOMContentLoaded', () => {{
  buildSidebar();
  initScrollSpy();
  initImages();
  badgeImageLabels();
}});
"""


# ── Assemble full HTML ────────────────────────────────────────────────────────

def assemble(body: str) -> str:
    body = wrap_sections(body)
    css  = get_css()
    js   = get_js()

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>ML / DL Interview Preparation Notes</title>

<!-- MathJax: render $...$ and $$...$$ -->
<script>
MathJax = {{
  tex: {{
    inlineMath: [['$','$'],['\\\\(','\\\\)']],
    displayMath: [['$$','$$'],['\\\\[','\\\\]']],
    processEscapes: true,
    processEnvironments: true
  }},
  options: {{ skipHtmlTags: ['script','noscript','style','textarea','pre'] }}
}};
</script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js" async></script>

<style>
{css}
</style>
</head>
<body>

<!-- Top bar -->
<div id="topbar">
  <button id="tog-btn">☰</button>
  <span id="bar-title">ML / DL Interview Preparation Notes</span>
  <button id="dark-btn">🌙  Dark</button>
</div>

<!-- Sidebar -->
<nav id="sidebar">
  <div id="sb-inner"></div>
</nav>

<!-- Main content -->
<div id="main-wrap">
  <main id="content">
{body}
  </main>
</div>

<!-- Back to top -->
<button id="back-top" title="Back to top">↑</button>

<script>
{js}
</script>
</body>
</html>"""


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    print(f"Source : {MD_FILE}")
    print("Converting markdown …")
    body = md_to_html(MD_FILE)

    print("Assembling HTML …")
    html = assemble(body)

    OUT_FILE.write_text(html, encoding='utf-8')
    kb = OUT_FILE.stat().st_size // 1024
    print(f"Output : {OUT_FILE}  ({kb} KB)")
    print("Done — open ML_DL_Notes.html in your browser.")


if __name__ == '__main__':
    main()
