<?php
/*
 * PHPAiModel-Transformer — aicore.php
 * char-level Transformer runtime in PHP (pre-LN, GELU, MHA, tied output head)
 * PHP 7.4 compatible. Loads JSON weights ONLY from php/Models/.
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

declare(strict_types=1);
@ini_set('max_execution_time', '600'); // 5 минут
@set_time_limit(600);
@ini_set('memory_limit', '2048M'); // или больше, если модель крупная
mb_internal_encoding('UTF-8');
header('Content-Type: application/json; charset=utf-8');

$MODELS_DIR = __DIR__ . DIRECTORY_SEPARATOR . 'Models';

// -------- helpers --------
function read_json_body(): array {
    $raw = file_get_contents('php://input') ?: '';
    $data = json_decode($raw, true);
    return is_array($data) ? $data : [];
}
function fail(string $msg){ echo json_encode(['ok'=>false,'error'=>$msg], JSON_UNESCAPED_UNICODE); exit; }

// безопасный путь (PHP 7.4 — без str_*_with)
function safe_model_path(string $fname, string $MODELS_DIR): string {
    $fname = trim($fname);
    if ($fname === '') throw new RuntimeException('Model name is empty');
    // запрет поддиректорий
    if (strpos($fname, '/') !== false || strpos($fname, '\\') !== false) {
        throw new RuntimeException('Invalid model name');
    }
    // только .json
    if (strtolower(pathinfo($fname, PATHINFO_EXTENSION)) !== 'json') {
        throw new RuntimeException('Model must be a .json file');
    }
    $path = $MODELS_DIR . DIRECTORY_SEPARATOR . $fname;
    $real = realpath($path);
    $realModels = realpath($MODELS_DIR);
    if (!$real || !$realModels || substr($real, 0, strlen($realModels)) !== $realModels) {
        throw new RuntimeException('Model file not found in Models/');
    }
    return $real;
}

function load_weights(string $path): array {
    $s = @file_get_contents($path);
    if ($s === false) throw new RuntimeException("Cannot read weights: $path");
    $w = json_decode($s, true);
    if (!is_array($w)) {
        $err = function_exists('json_last_error_msg') ? json_last_error_msg() : 'unknown';
        throw new RuntimeException("Invalid JSON in weights ($err): $path");
    }
    return $w;
}


// ---- math utils ----
function layernorm(array $x, array $gamma, array $beta, float $eps=1e-5): array {
    $T = count($x); $D = count($x[0]);
    $y = array_fill(0, $T, array_fill(0, $D, 0.0));
    for ($t=0; $t<$T; $t++) {
        $mu = 0.0; for ($i=0; $i<$D; $i++) $mu += $x[$t][$i]; $mu /= $D;
        $var = 0.0; for ($i=0; $i<$D; $i++) { $d = $x[$t][$i]-$mu; $var += $d*$d; }
        $var /= $D; $inv = 1.0 / sqrt($var + $eps);
        for ($i=0; $i<$D; $i++) { $n = ($x[$t][$i]-$mu) * $inv; $y[$t][$i] = $n * $gamma[$i] + $beta[$i]; }
    }
    return $y;
}
function gelu(float $x): float { return 0.5*$x*(1.0 + tanh(0.7978845608028654*($x + 0.044715*$x*$x*$x))); }
function matmul(array $A, array $W): array { // A: [T x D], W: [O x D] -> [T x O]
    $T = count($A); $D = count($A[0]); $O = count($W);
    $Y = array_fill(0, $T, array_fill(0, $O, 0.0));
    for ($t=0; $t<$T; $t++) {
        for ($o=0; $o<$O; $o++) {
            $s = 0.0; $row = $W[$o];
            for ($i=0; $i<$D; $i++) $s += $A[$t][$i] * $row[$i];
            $Y[$t][$o] = $s;
        }
    }
    return $Y;
}
function add_bias(array $X, array $b): array { $T=count($X); $O=count($b); for($t=0;$t<$T;$t++)for($o=0;$o<$O;$o++) $X[$t][$o]+=$b[$o]; return $X; }
function softmax_row(array $v): array { $m=max($v); $s=0.0; foreach($v as &$x){ $x=exp($x-$m); $s+=$x; } foreach($v as &$x){ $x/=$s?:1.0; } return $v; }

// ---- attention (causal) ----
function mha(array $X, array $qkv_w, array $qkv_b, array $proj_w, array $proj_b, int $n_head): array {
    $T = count($X); $D = count($X[0]);
    $QKV = add_bias(matmul($X, $qkv_w), $qkv_b); // [T x 3D]
    $Q=$K=$V = array_fill(0,$T,array_fill(0,$D,0.0));
    for ($t=0;$t<$T;$t++){ for($i=0;$i<$D;$i++){ $Q[$t][$i]=$QKV[$t][$i]; $K[$t][$i]=$QKV[$t][$i+$D]; $V[$t][$i]=$QKV[$t][$i+2*$D]; } }
    $dh = intdiv($D, $n_head);
    $heads = array_fill(0,$T,array_fill(0,$D,0.0));
    $scale = 1.0 / sqrt((float)$dh);
    for ($h=0; $h<$n_head; $h++) {
        $q=$k=$v=[]; for($t=0;$t<$T;$t++){ $q[$t]=array_slice($Q[$t],$h*$dh,$dh); $k[$t]=array_slice($K[$t],$h*$dh,$dh); $v[$t]=array_slice($V[$t],$h*$dh,$dh); }
        $scores = array_fill(0,$T,array_fill(0,$T,-INF));
        for ($t=0;$t<$T;$t++){ for($u=0;$u<=$t;$u++){ $s=0.0; for($i=0;$i<$dh;$i++) $s+=$q[$t][$i]*$k[$u][$i]; $scores[$t][$u]=$s*$scale; } }
        for ($t=0;$t<$T;$t++){ $scores[$t]=softmax_row($scores[$t]); }
        $out_h = array_fill(0,$T,array_fill(0,$dh,0.0));
        for ($t=0;$t<$T;$t++){ for($i=0;$i<$dh;$i++){ $s=0.0; for($u=0;$u<$T;$u++) $s+=$scores[$t][$u]*$v[$u][$i]; $out_h[$t][$i]=$s; } }
        for ($t=0;$t<$T;$t++) for($i=0;$i<$dh;$i++) $heads[$t][$h*$dh+$i]=$out_h[$t][$i];
    }
    return add_bias(matmul($heads, $proj_w), $proj_b); // [T x D]
}

// ---- sampling ----
function top_k_filter(array $logits, int $k): array {
    $N = count($logits); $idx = range(0,$N-1);
    $vals = $logits; array_multisort($vals, SORT_DESC, $idx);
    $k = max(1, min($k, $N)); $th = $vals[$k-1];
    $out = array_fill(0,$N, -INF);
    for ($i=0; $i<$N; $i++) if ($logits[$i] >= $th) $out[$i] = $logits[$i];
    return $out;
}
function sample_next(array $logits, float $temperature=1.0, int $top_k=0): int {
    $N = count($logits);
    for ($i=0; $i<$N; $i++) $logits[$i] /= max(1e-6, $temperature);
    if ($top_k > 0) $logits = top_k_filter($logits, $top_k);
    $m = max($logits); $sum=0.0;
    for ($i=0; $i<$N; $i++){ $logits[$i] = exp($logits[$i]-$m); $sum += $logits[$i]; }
    $r = lcg_value() * ($sum ?: 1.0); $acc=0.0;
    for ($i=0; $i<$N; $i++){ $acc += $logits[$i]; if ($r <= $acc) return $i; }
    return $N-1;
}

// ---- tokenizer ----
function str_to_ids(string $s, array $stoi, int $unk): array {
    $ids = []; $len = mb_strlen($s);
    for ($i=0; $i<$len; $i++) { $ch = mb_substr($s, $i, 1); $ids[] = $stoi[$ch] ?? $unk; }
    return $ids;
}
function ids_to_str(array $ids, array $itos): string {
    $out = ''; foreach ($ids as $i) $out .= $itos[$i] ?? ''; return $out;
}

// ---- forward ----
function forward_logits(array $ctx_ids, array $W): array {
    $cfg = $W['config'];
    $V = (int)$cfg['vocab_size']; $D = (int)$cfg['d_model']; $H = (int)$cfg['n_head']; $Tmax = (int)$cfg['max_seq'];
    $T = min(count($ctx_ids), $Tmax);
    $X = array_fill(0,$T,array_fill(0,$D,0.0));
    for ($t=0; $t<$T; $t++) {
        $tok = $ctx_ids[count($ctx_ids)-$T+$t];
        $row = $W['tok_emb'][$tok];
        for ($i=0; $i<$D; $i++) $X[$t][$i] = $row[$i] + $W['pos_emb'][$t][$i];
    }
    foreach ($W['layers'] as $layer) {
        $Y = layernorm($X, $layer['ln1_w'], $layer['ln1_b']);
        $A = mha($Y, $layer['attn_qkv_w'], $layer['attn_qkv_b'], $layer['attn_proj_w'], $layer['attn_proj_b'], (int)$W['config']['n_head']);
        for ($t=0; $t<$T; $t++) for ($i=0; $i<$D; $i++) $X[$t][$i] += $A[$t][$i];
        $Z = layernorm($X, $layer['ln2_w'], $layer['ln2_b']);
        $H1 = add_bias(matmul($Z, $layer['fc1_w']), $layer['fc1_b']);
        $H1D = count($H1[0]); for ($t=0; $t<$T; $t++) for ($i=0; $i<$H1D; $i++) $H1[$t][$i] = gelu($H1[$t][$i]);
        $H2 = add_bias(matmul($H1, $layer['fc2_w']), $layer['fc2_b']);
        for ($t=0; $t<$T; $t++) for ($i=0; $i<$D; $i++) $X[$t][$i] += $H2[$t][$i];
    }
    if (isset($W['ln_f_w'])) $X = layernorm($X, $W['ln_f_w'], $W['ln_f_b']);
    $last = $X[$T-1]; $logits = array_fill(0,$V,0.0);
    for ($v=0; $v<$V; $v++) { $row = $W['tok_emb'][$v]; $s=0.0; for ($i=0; $i<$D; $i++) $s += $last[$i]*$row[$i]; $logits[$v] = $s; }
    return $logits;
}

function generate(array $ctx_ids, array $W, int $max_new=128, float $temperature=1.0, int $top_k=40): array {
    for ($n=0; $n<$max_new; $n++) {
        $logits = forward_logits($ctx_ids, $W);
        $next = sample_next($logits, $temperature, $top_k);
        $ctx_ids[] = $next;
        if (count($ctx_ids) > (int)$W['config']['max_seq']) array_shift($ctx_ids);
    }
    return $ctx_ids;
}

// ---- entry point ----
try {
    $req = read_json_body();
    $prompt = (string)($req['prompt'] ?? '');
    $temperature = (float)($req['temperature'] ?? 0.9);
    $top_k = (int)($req['top_k'] ?? 40);
    $max_new = (int)($req['max_new'] ?? 200);
    $model_name = (string)($req['model'] ?? ''); // filename only

    if ($model_name === '') fail('Model is required (select .json in php/Models/)');
    $model_path = safe_model_path($model_name, $MODELS_DIR);
    $W = load_weights($model_path);
    // --- schema adapter: accept alternative keys ---
    if (!isset($W['config']) && isset($W['cfg'])) {
        $W['config'] = $W['cfg'];
    }
    if (!isset($W['config']) && isset($W['metadata']['config'])) {
        $W['config'] = $W['metadata']['config'];
    }
    // ---- validate weights structure ----
    function ensure_keys(array $W, array $keys){
        foreach($keys as $k){
            if (!array_key_exists($k, $W)) {
                throw new RuntimeException("Model JSON missing key: '$k'");
            }
        }
    }
    if (!isset($W['config'])) {
        $keys = implode(', ', array_keys($W));
        throw new RuntimeException("Model JSON missing key: 'config' (top-level keys: {$keys}, file=".basename($model_path).")");
    }
    ensure_keys($W, ['config','vocab','tok_emb','pos_emb','layers']);

    $cfg = $W['config'];
    foreach (['vocab_size','d_model','n_head','n_layer','d_ff','max_seq'] as $k) {
        if (!isset($cfg[$k])) throw new RuntimeException("config.$k is missing");
    }

// quick sanity checks
    $V = (int)$cfg['vocab_size'];
    $L = (int)$cfg['n_layer'];
    if (!is_array($W['vocab']) || count($W['vocab']) !== $V) {
        throw new RuntimeException("vocab size mismatch: expected $V, got ".(is_array($W['vocab'])?count($W['vocab']):'non-array'));
    }
    if (!is_array($W['layers']) || count($W['layers']) !== $L) {
        throw new RuntimeException("layers count mismatch: expected $L");
    }

    $vocab = $W['vocab'] ?? [];
    if (!$vocab) fail('Weights missing vocab');
    $stoi = []; $itos = [];
    foreach ($vocab as $i=>$ch) { $stoi[$ch] = $i; $itos[$i] = $ch; }
    $unk = array_search('<unk>', $vocab, true); if ($unk===false) $unk = 0;

    $ctx = str_to_ids($prompt, $stoi, (int)$unk);
    $out_ids = generate($ctx, $W, $max_new, $temperature, $top_k);
    $gen_ids = array_slice($out_ids, count($ctx));
    $text = ids_to_str($gen_ids, $itos);

    $debug = [
        'model_file' => basename($model_path),
        'vocab' => count($W['vocab']),
        'd_model' => (int)$cfg['d_model'],
        'n_layer' => (int)$cfg['n_layer'],
        'n_head'  => (int)$cfg['n_head'],
        'max_seq' => (int)$cfg['max_seq'],
    ];
    echo json_encode(['ok'=>true, 'text'=>$text, 'meta'=>$debug], JSON_UNESCAPED_UNICODE);
} catch (Throwable $e) {
    echo json_encode(['ok'=>false, 'error'=>$e->getMessage()], JSON_UNESCAPED_UNICODE);
}
