#!/usr/bin/env python3
"""brain_viz.py — snapshot the current contents of Hermes' memory as shareable docs.

Reads the unified store (memory_store.db: facts + sqlite-vec + graph) and writes,
to an output dir:
  • BRAIN_SNAPSHOT.md   — stats + an inline Mermaid map of the core (renders on GitHub)
  • brain_graph.html    — full interactive graph (vis.js, self-contained data)
  • brain_graph.json    — raw node-link graph

Usage: brain_viz.py <output_dir> [--date YYYY-MM-DD]
"""

import json
import os
import re
import sqlite3
import sys
from collections import Counter
from pathlib import Path

HERMES_HOME = Path(os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes")))
sys.path.insert(0, str(HERMES_HOME / "scripts"))
import memstore as ms  # noqa: E402

TYPE_COLOR = {
    "person": "#e15759", "company": "#4e79a7", "project": "#59a14f",
    "location": "#f28e2b", "idea": "#b07aa1", "concept": "#9c9c9c",
}
_PII = re.compile(r"\+?\d[\d\-\(\)\.\s]{7,}\d|[\w.+-]+@[\w-]+\.\w+")


def _safe(s):
    return str(s).replace('"', "'")


def main():
    outdir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("./brain_snapshot")
    outdir.mkdir(parents=True, exist_ok=True)
    date = "today"
    if "--date" in sys.argv:
        date = sys.argv[sys.argv.index("--date") + 1]

    con = ms.connect()
    ms.init_schema(con)
    G = ms.load_graph(con)
    st = ms.stats(con)
    try:
        n_facts = con.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
    except Exception:
        n_facts = 0

    import networkx as nx
    U = G.to_undirected()
    deg = dict(U.degree())
    type_counts = Counter(G.nodes[n].get("node_type", "concept") for n in G.nodes())
    rel_counts = Counter(d.get("relation", "?") for _, _, d in G.edges(data=True))
    top_hubs = sorted(deg.items(), key=lambda x: -x[1])[:20]

    # ---- raw node-link json ----
    (outdir / "brain_graph.json").write_text(json.dumps(nx.node_link_data(G), ensure_ascii=False, indent=1))

    # ---- interactive vis.js html ----
    # Facts per entity (substring match) so clicking a node shows "what Hermes knows".
    try:
        fc = sqlite3.connect(str(HERMES_HOME / "memory_store.db"))
        _allfacts = [r[0].strip() for r in fc.execute("SELECT content FROM facts ORDER BY trust_score DESC") if r[0]]
        fc.close()
    except Exception:
        _allfacts = []

    def _facts_for(name):
        nl = name.lower()
        if len(nl) < 3:
            return []
        return [f for f in _allfacts if nl in f.lower()][:6]

    vnodes = [{"id": n, "label": n, "group": G.nodes[n].get("node_type", "concept"),
               "value": deg.get(n, 1),
               "attrs": {k: v for k, v in G.nodes[n].items()
                         if k != "node_type" and isinstance(v, (str, int, float, bool))},
               "facts": _facts_for(n)} for n in G.nodes()]
    vedges = [{"from": u, "to": v, "label": d.get("relation", "")}
              for u, v, d in G.edges(data=True)]
    html = _HTML.replace("__NODES__", json.dumps(vnodes, ensure_ascii=False)) \
                .replace("__EDGES__", json.dumps(vedges, ensure_ascii=False)) \
                .replace("__DATE__", date) \
                .replace("__N__", str(G.number_of_nodes())).replace("__E__", str(G.number_of_edges()))
    (outdir / "brain_graph.html").write_text(html, encoding="utf-8")

    # ---- Mermaid map of the CORE (skip the 120 prospect leaves; show the structure) ----
    is_lead = {n for n in G.nodes() if "stage" in G.nodes[n]}
    core = [n for n in G.nodes() if n not in is_lead]
    core = sorted(core, key=lambda n: -deg.get(n, 0))[:26]
    cset = set(core)
    nid = {n: f"N{i}" for i, n in enumerate(core)}
    mer = ["```mermaid", "graph LR"]
    for n in core:
        c = TYPE_COLOR.get(G.nodes[n].get("node_type", "concept"), "#9c9c9c")
        mer.append(f'  {nid[n]}["{_safe(n)}"]')
    seen = set()
    for u, v, d in G.edges(data=True):
        if u in cset and v in cset and (u, v) not in seen:
            seen.add((u, v))
            mer.append(f'  {nid[u]} -->|{_safe(d.get("relation",""))[:24]}| {nid[v]}')
    mer.append("```")
    mermaid = "\n".join(mer)

    # ---- a few non-sensitive sample facts ----
    samples = []
    try:
        fc = sqlite3.connect(str(HERMES_HOME / "memory_store.db"))
        for (txt,) in fc.execute("SELECT content FROM facts ORDER BY trust_score DESC, fact_id DESC"):
            t = (txt or "").strip()
            if t and not _PII.search(t) and 20 <= len(t) <= 160:
                samples.append(t)
            if len(samples) >= 8:
                break
        fc.close()
    except Exception:
        pass

    # ---- markdown ----
    md = []
    md.append("# 🧠 Hermes Brain — Memory Snapshot")
    md.append(f"\n*Snapshot taken {date}. This is a read-only picture of what the Hermes agent currently "
              "knows — its facts, its semantic memory, and the relationship graph it reasons over.*\n")
    md.append("## At a glance\n")
    md.append("| | count |\n|---|---|")
    md.append(f"| Facts (structured memory) | {n_facts} |")
    md.append(f"| Semantic vectors (recall) | {st['vectors']} |")
    md.append(f"| Graph entities (nodes) | {G.number_of_nodes()} |")
    md.append(f"| Graph relationships (edges) | {G.number_of_edges()} |")
    md.append("\n**Entity types:** " + ", ".join(f"{t} {c}" for t, c in type_counts.most_common()))
    md.append("\n## The core graph\n")
    md.append("How the main entities connect (the ~120 sales prospects are omitted here for legibility — "
              "they cluster by city + vertical in the interactive graph below):\n")
    md.append(mermaid)
    md.append("\n## Biggest hubs\n")
    md.append("| entity | connections |\n|---|---|")
    for n, d in top_hubs:
        md.append(f"| {n} | {d} |")
    md.append("\n## Relationship vocabulary\n")
    md.append(", ".join(f"`{r}`×{c}" for r, c in rel_counts.most_common(18)))
    if samples:
        md.append("\n## A few things Hermes knows\n")
        for s in samples:
            md.append(f"- {s}")
    md.append("\n## Explore the full graph\n")
    md.append(f"Open **`brain_graph.html`** in a browser for the complete interactive map "
              f"({G.number_of_nodes()} entities, {G.number_of_edges()} relationships) — zoom, drag, search. "
              "Raw data is in `brain_graph.json`.\n")
    md.append("\n---\n*Generated by `scripts/brain_viz.py` from the unified memory store "
              "(`memory_store.db`). Nothing here is editable — it's a mirror.*")
    (outdir / "BRAIN_SNAPSHOT.md").write_text("\n".join(md), encoding="utf-8")

    con.close()
    print(f"wrote BRAIN_SNAPSHOT.md, brain_graph.html, brain_graph.json to {outdir}")
    print(f"  {n_facts} facts, {st['vectors']} vectors, {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")


_HTML = """<!DOCTYPE html><html><head><meta charset="utf-8"><title>Hermes Brain — __DATE__</title>
<script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
<style>
  *{box-sizing:border-box}
  body{margin:0;font-family:-apple-system,Segoe UI,Roboto,sans-serif;background:#16181d;color:#e6e6e6;overflow:hidden}
  #top{position:fixed;top:0;left:0;right:0;height:46px;background:#0e0f12;border-bottom:1px solid #2a2d34;display:flex;align-items:center;gap:8px;padding:0 14px;z-index:5}
  #top b{color:#fff} .meta{color:#7a8089;font-size:12px}
  input,button{background:#21242b;border:1px solid #3a3e47;color:#e6e6e6;padding:6px 9px;border-radius:5px;font-size:13px}
  button{cursor:pointer} button.on{background:#2d6cdf;border-color:#2d6cdf;color:#fff}
  #legend{display:flex;gap:9px;flex-wrap:wrap;font-size:12px;margin-left:auto}
  .lg{display:flex;align-items:center;gap:4px;cursor:pointer;user-select:none}
  .lg.off{opacity:.32} .dot{width:11px;height:11px;border-radius:50%}
  #net{position:fixed;top:46px;left:0;bottom:0;right:344px}
  #panel{position:fixed;top:46px;right:0;bottom:0;width:344px;background:#0e0f12;border-left:1px solid #2a2d34;overflow-y:auto;padding:16px}
  #panel h2{margin:0 0 6px;font-size:18px;color:#fff;word-break:break-word}
  .badge{display:inline-block;padding:2px 8px;border-radius:10px;font-size:11px;color:#fff;margin-right:6px}
  .sec{margin-top:15px} .sec h3{font-size:11px;text-transform:uppercase;letter-spacing:.6px;color:#7a8089;margin:0 0 7px;border-bottom:1px solid #23262d;padding-bottom:4px}
  .kv{display:flex;justify-content:space-between;font-size:13px;padding:3px 0;border-bottom:1px solid #1a1d23}
  .kv span{color:#9aa0a8} .kv b{color:#fff;font-weight:500}
  .lnk{font-size:13px;padding:4px 0;cursor:pointer;color:#cdd3da;border-bottom:1px solid #1a1d23}
  .lnk:hover{color:#6fa8ff} .rel{color:#7a8089;font-size:11px}
  .fact{font-size:12.5px;line-height:1.5;color:#c4cad2;padding:7px 0;border-bottom:1px solid #1a1d23}
  .hint{color:#7a8089;font-size:13px;margin-top:40px;text-align:center;line-height:1.6}
</style></head>
<body>
<div id="top">
  <b>🧠 Hermes Brain</b><span class="meta">__DATE__ · __N__ entities · __E__ links</span>
  <input id="q" placeholder="search…" style="width:140px" oninput="search(this.value)">
  <button id="tEdge" onclick="toggleEdges()">labels</button>
  <button id="tLeaf" onclick="toggleLeaves()">hide prospects</button>
  <button id="tPhys" class="on" onclick="togglePhysics()">physics</button>
  <div id="legend"></div>
</div>
<div id="net"></div>
<div id="panel"><div class="hint">Click any entity to see its value, its links, and what Hermes knows about it.</div></div>
<script>
const NODES=__NODES__, EDGES=__EDGES__;
const colors={person:"#e15759",company:"#4e79a7",project:"#59a14f",location:"#f28e2b",idea:"#b07aa1",concept:"#9c9c9c",platform:"#76b7b2"};
const nodes=new vis.DataSet(NODES.map(n=>({id:n.id,label:n.label,group:n.group,value:n.value,color:colors[n.group]||"#9c9c9c",scaling:{min:8,max:48},font:{color:"#cfd5dc",size:13}})));
const edges=new vis.DataSet(EDGES.map((e,i)=>({id:i,from:e.from,to:e.to,label:"",arrows:"to",color:{color:"#3a3e47",highlight:"#9aa0a8"},font:{color:"#8b919a",size:10,strokeWidth:0},smooth:false})));
const net=new vis.Network(document.getElementById('net'),{nodes,edges},{
  nodes:{shape:'dot'}, edges:{arrows:{to:{scaleFactor:.4}}},
  physics:{barnesHut:{gravitationalConstant:-9000,springLength:135,springConstant:.02},stabilization:{iterations:220}},
  interaction:{hover:true}});
const byId={}; NODES.forEach(n=>byId[n.id]=n);
const active=new Set(Object.keys(colors)); let hideLeaves=false,showLabels=false,physics=true;
const lg=document.getElementById('legend');
[...new Set(NODES.map(n=>n.group))].forEach(g=>{const d=document.createElement('div');d.className='lg';
  d.innerHTML='<span class="dot" style="background:'+(colors[g]||'#9c9c9c')+'"></span>'+g;
  d.onclick=()=>{d.classList.toggle('off');active.has(g)?active.delete(g):active.add(g);applyFilter();};lg.appendChild(d);});
function applyFilter(){nodes.forEach(n=>{const s=byId[n.id];let hidden=!active.has(s.group);
  if(hideLeaves&&s.attrs&&('stage'in s.attrs))hidden=true;nodes.update({id:n.id,hidden});});}
function toggleLeaves(){hideLeaves=!hideLeaves;document.getElementById('tLeaf').classList.toggle('on',hideLeaves);applyFilter();}
function toggleEdges(){showLabels=!showLabels;document.getElementById('tEdge').classList.toggle('on',showLabels);
  edges.forEach(e=>edges.update({id:e.id,label:showLabels?EDGES[e.id].label:""}));}
function togglePhysics(){physics=!physics;document.getElementById('tPhys').classList.toggle('on',physics);net.setOptions({physics:{enabled:physics}});}
function highlight(id){const nb=new Set(net.getConnectedNodes(id));nb.add(id);
  nodes.forEach(n=>nodes.update({id:n.id,opacity:nb.has(n.id)?1:0.16}));}
function select(id){if(!byId[id])return;net.selectNodes([id]);highlight(id);show(id);net.focus(id,{scale:1.15,animation:true});}
net.on('selectNode',p=>{highlight(p.nodes[0]);show(p.nodes[0]);});
net.on('deselectNode',()=>{nodes.forEach(n=>nodes.update({id:n.id,opacity:1}));
  document.getElementById('panel').innerHTML='<div class="hint">Click any entity to see its value, its links, and what Hermes knows about it.</div>';});
function esc(s){return String(s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));}
function show(id){const n=byId[id];if(!n)return;
  const out=EDGES.filter(e=>e.from===id),inc=EDGES.filter(e=>e.to===id);
  let h='<h2>'+esc(n.label)+'</h2><div><span class="badge" style="background:'+(colors[n.group]||'#9c9c9c')+'">'+n.group+'</span><span class="meta">'+(out.length+inc.length)+' links · '+(n.facts?n.facts.length:0)+' facts</span></div>';
  if(n.attrs&&Object.keys(n.attrs).length){h+='<div class="sec"><h3>Attributes</h3>'+Object.entries(n.attrs).map(kv=>'<div class="kv"><span>'+esc(kv[0])+'</span><b>'+esc(kv[1])+'</b></div>').join('')+'</div>';}
  if(out.length){h+='<div class="sec"><h3>Links out ('+out.length+')</h3>'+out.map(e=>'<div class="lnk" data-go="'+esc(e.to)+'"><span class="rel">'+esc(e.label)+' &#8594;</span> '+esc(e.to)+'</div>').join('')+'</div>';}
  if(inc.length){h+='<div class="sec"><h3>Links in ('+inc.length+')</h3>'+inc.map(e=>'<div class="lnk" data-go="'+esc(e.from)+'">'+esc(e.from)+' <span class="rel">&#8594; '+esc(e.label)+'</span></div>').join('')+'</div>';}
  if(n.facts&&n.facts.length){h+='<div class="sec"><h3>What Hermes knows</h3>'+n.facts.map(f=>'<div class="fact">'+esc(f)+'</div>').join('')+'</div>';}
  document.getElementById('panel').innerHTML=h;}
document.getElementById('panel').addEventListener('click',e=>{const t=e.target.closest('[data-go]');if(t)select(t.getAttribute('data-go'));});
function search(q){q=q.toLowerCase();if(!q)return;const m=NODES.find(n=>n.id.toLowerCase().includes(q));if(m)select(m.id);}
</script></body></html>"""


if __name__ == "__main__":
    main()
