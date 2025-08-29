<?php
/*
* PHPAiModel-Transformer — index.php
* chat UI for PHP Transformer runtime with model picker (PHP 7.4+)
*  PHP 7.4 compatible. Loads JSON weights ONLY from php/Models/.
*
* Developed by: Artur Strazewicz — concept, architecture, PHP N-gram runtime, UI.
* Year: 2025. License: MIT.
*
* Links:
*   GitHub:      https://github.com/iStark/PHPAiModel-Transformer
*   LinkedIn:    https://www.linkedin.com/in/arthur-stark/
*   TruthSocial: https://truthsocial.com/@strazewicz
*   X (Twitter): https://x.com/strazewicz
*/

$MODELS_DIR = __DIR__ . DIRECTORY_SEPARATOR . 'Models';
@mkdir($MODELS_DIR, 0777, true);
@ini_set('memory_limit', '2048M'); // или больше, если модель крупная
// Сканируем модели без str_ends_with
$models = [];
if (is_dir($MODELS_DIR)) {
    foreach (scandir($MODELS_DIR) as $f) {
        if ($f === '.' || $f === '..') continue;
        $p = $MODELS_DIR . DIRECTORY_SEPARATOR . $f;
        if (is_file($p) && strtolower(pathinfo($p, PATHINFO_EXTENSION)) === 'json') {
            $models[] = $f;
        }
    }
}
?><!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>PHPAiModel-Transformer — Chat</title>
    <style>
        :root { color-scheme: light; }
        body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; background:#f7f7fb; margin:0; }
        .wrap { max-width: 900px; margin: 0 auto; display: flex; flex-direction: column; height: 100vh; }
        header { padding: 16px; background: #fff; border-bottom: 1px solid #e7e7ef; }
        h1 { font-size: 16px; margin: 0; color:#222; }
        #log { flex: 1; overflow-y: auto; padding: 16px; display:flex; flex-direction: column-reverse; }
        .msg { background:#fff; border:1px solid #eee; padding:12px 14px; border-radius:10px; margin:10px 0; box-shadow: 0 1px 2px rgba(0,0,0,0.04); white-space:pre-wrap; }
        .ai { border-left: 4px solid #4f46e5; }
        .user { border-left: 4px solid #9ca3af; }
        footer { background:#fff; border-top:1px solid #e7e7ef; padding:10px; }
        textarea { width:100%; min-height: 90px; resize: vertical; padding:12px; border-radius:10px; border:1px solid #ddd; box-sizing:border-box; font: inherit; }
        .row { display:flex; gap:8px; align-items:center; flex-wrap: wrap; }
        .grow { flex:1; }
        button { padding:10px 14px; border-radius:10px; border:1px solid #4f46e5; background:#4f46e5; color:#fff; cursor:pointer; }
        .controls { display:flex; gap:10px; margin-top:8px; flex-wrap: wrap; }
        input[type="number"] { width:90px; padding:6px; border:1px solid #ddd; border-radius:8px; }
        select { padding:8px; border:1px solid #ddd; border-radius:8px; }
        .bar { height:6px; background:#e9e9f5; border-radius:999px; overflow:hidden; margin-top:6px; }
        .bar > i { display:block; height:100%; width:0%; background:#4f46e5; transition: width .2s; }
        .muted { color:#666; font-size:12px; }
        footer a{color:inherit}
        .links a{margin-right:.75rem}
    </style>
</head>
<body>
<div class="wrap">
    <header>
        <h1>PHPAiModel-Transformer — Char-level (pre-LN • GELU • MHA)</h1>
        <div class="muted">Models folder: <code>php/Models/</code>. Current: <span id="curModel">—</span></div>
        <div class="row" style="margin-top:8px">
            <label>Model
                <select id="model">
                    <?php if (!$models): ?>
                        <option value="">(put .json into php/Models/)</option>
                    <?php else: foreach ($models as $m): ?>
                        <option value="<?= htmlspecialchars($m, ENT_QUOTES,'UTF-8') ?>"><?= htmlspecialchars($m, ENT_QUOTES,'UTF-8') ?></option>
                    <?php endforeach; endif; ?>
                </select>
            </label>
            <label>Max new <input id="max_new" type="number" value="200" min="1" max="1024"/></label>
            <label>Temp <input id="temp" type="number" step="0.1" value="0.9" min="0.1" max="2.0"/></label>
            <label>Top-k <input id="topk" type="number" value="40" min="0" max="200"/></label>
            <button id="reload" title="Reload models list" onclick="location.reload()">↻</button>
        </div>
    </header>

    <div id="log"></div>

    <footer>
        <div class="row">
            <div class="grow"><textarea id="ta" placeholder="Type your prompt…"></textarea></div>
            <div><button id="btn">Send</button></div>
        </div>
        <div class="bar"><i id="pbar"></i></div>
        <div><strong>PHPAiModel-Transformer</strong> — pre-LN, GELU, MHA, tied output head.</div>
        <div>© <span id="year">2025</span>. Developed by <strong>Artur Strazewicz</strong> — concept, architecture, PHPAiModel-Transformer, UI,  <strong>MIT license</strong>.</div>
        <div class="links">
            <a href="https://github.com/iStark/PHPAiModel-Transformer" target="_blank" rel="noopener">GitHub</a>
            <a href="https://www.linkedin.com/in/arthur-stark/" target="_blank" rel="noopener">LinkedIn</a>
            <a href="https://truthsocial.com/@strazewicz" target="_blank" rel="noopener">Truth Social</a>
            <a href="https://x.com/strazewicz" target="_blank" rel="noopener">X (Twitter)</a>
        </div>
    </footer>
</div>

<script>
    const log = document.getElementById('log');
    const pbar = document.getElementById('pbar');
    const ta = document.getElementById('ta');
    const btn = document.getElementById('btn');
    const sel = document.getElementById('model');
    const curModel = document.getElementById('curModel');

    (function initModel(){
        const saved = localStorage.getItem('php_transformer_model');
        if (saved) {
            for (let i=0;i<sel.options.length;i++){
                if (sel.options[i].value === saved){ sel.selectedIndex = i; break; }
            }
        }
        curModel.textContent = sel.value || '—';
        sel.addEventListener('change', ()=>{
            localStorage.setItem('php_transformer_model', sel.value);
            curModel.textContent = sel.value || '—';
        });
    })();

    function add(role, text){
        const el = document.createElement('div');
        el.className = 'msg ' + (role==='ai'?'ai':'user');
        el.textContent = text;
        log.prepend(el);
    }

    async function call(){
        const prompt = ta.value.trim();
        const model = sel.value.trim();
        if(!prompt){ return; }
        if(!model){ add('ai','Error: select a model (.json) in php/Models/'); return; }

        add('user', prompt); ta.value=''; pbar.style.width='5%';

        const payload = {
            prompt,
            max_new: +document.getElementById('max_new').value,
            temperature: +document.getElementById('temp').value,
            top_k: +document.getElementById('topk').value,
            model: model // filename only
        };

        try{
            const r = await fetch('aicore.php', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload)});
            pbar.style.width='60%';
            const txt = await r.text();
            let j;
            try { j = JSON.parse(txt); }
            catch(e){
                pbar.style.width='0%';
                add('ai', 'Server error (non-JSON):\\n' + txt.slice(0, 800));
                return;
            }
            pbar.style.width='100%'; setTimeout(()=>pbar.style.width='0%', 400);
            if(j.ok){ add('ai', j.text); } else { add('ai', 'Error: '+j.error); }
        }catch(e){
            pbar.style.width='0%'; add('ai','Network error: '+ (e && e.message ? e.message : 'fetch failed'));
        }
    }
    btn.onclick = call;
    ta.onkeydown = (e)=>{ if(e.key==='Enter' && (e.ctrlKey||e.metaKey)) call(); };

    // Set current year in footer (kept static if JS disabled)
    document.getElementById('year').textContent = String(new Date().getFullYear());
</script>
</body>
</html>
