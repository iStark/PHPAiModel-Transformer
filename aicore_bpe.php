<?php
/*
 * aicore.php — GPT-style Transformer inference (PHP) with **BPE** tokenization.
 * - Loads JSON weights exported by app.py
 * - Loads tokenizer Models/<base>_tokenizer.json
 * - Uses bpe.php (encode/decode)
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
@ini_set('memory_limit','2048M');
@set_time_limit(0);

header('Access-Control-Allow-Origin: *');
header('Access-Control-Allow-Headers: Content-Type');

require_once __DIR__ . '/bpe.php';

// ---------- utils ----------
function json_body(): array {
    $raw = file_get_contents('php://input') ?: '';
    $b = json_decode($raw, true);
    return is_array($b) ? $b : [];
}
function http_error(int $code, string $msg) {
    http_response_code($code);
    header('Content-Type: application/json; charset=utf-8');
    echo json_encode(['error'=>$msg], JSON_UNESCAPED_UNICODE);
    exit;
}
function clamp(float $x, float $a, float $b){ return max($a, min($b, $x)); }
function zeros(int $r, int $c){ $row=array_fill(0,$c,0.0); $M=[]; for($i=0;$i<$r;$i++) $M[$i]=$row; return $M; }
function matmul(array $A, array $B): array {
    $m=count($A); $n=count($A[0]??[0]); $p=count($B[0]??[0]);
    $C=zeros($m,$p);
    for($i=0;$i<$m;$i++){
        for($k=0;$k<$n;$k++){
            $a=$A[$i][$k]; if($a==0.0) continue;
            for($j=0;$j<$p;$j++){ $C[$i][$j]+=$a*$B[$k][$j]; }
        }
    }
    return $C;
}
function add_bias(array $X, array $b): array {
    $m=count($X); $n=count($X[0]??[0]);
    for($i=0;$i<$m;$i++){ for($j=0;$j<$n;$j++){ $X[$i][$j]+=$b[$j]; } }
    return $X;
}
function layernorm(array $X, array $g, array $b, float $eps=1e-5): array {
    $m=count($X); $n=count($X[0]??[0]); $Y=zeros($m,$n);
    for($i=0;$i<$m;$i++){
        $mu=0.0; for($j=0;$j<$n;$j++) $mu+=$X[$i][$j]; $mu/=$n?:1;
        $var=0.0; for($j=0;$j<$n;$j++){ $d=$X[$i][$j]-$mu; $var+=$d*$d; } $var/=$n?:1;
        $inv=1.0/sqrt($var+$eps);
        for($j=0;$j<$n;$j++){ $norm=($X[$i][$j]-$mu)*$inv; $Y[$i][$j]=$norm*$g[$j]+$b[$j]; }
    }
    return $Y;
}
function gelu(array $X): array {
    $m=count($X); $n=count($X[0]??[0]);
    for($i=0;$i<$m;$i++){
        for($j=0;$j<$n;$j++){
            $x=$X[$i][$j]; $X[$i][$j]=0.5*$x*(1.0+tanh(0.7978845608*($x+0.044715*$x*$x*$x)));
        }
    }
    return $X;
}
function softmax_row(array $row): array {
    $max=-INF; foreach($row as $v) if($v>$max) $max=$v;
    $sum=0.0; $out=[];
    foreach($row as $v){ $e=exp($v-$max); $out[]=$e; $sum+=$e; }
    if ($sum<=0.0) $sum=1e-12;
    foreach($out as $i=>$e){ $out[$i]=$e/$sum; }
    return $out;
}
function logits_to_probs(array $logits, float $temperature): array {
    if ($temperature<=0) $temperature=1e-6;
    $max=-INF; foreach($logits as $l){ $max=max($max,$l/$temperature); }
    $sum=0.0; $probs=[];
    foreach($logits as $l){ $e=exp($l/$temperature - $max); $probs[]=$e; $sum+=$e; }
    if ($sum<=0.0) $sum=1e-12;
    foreach($probs as $i=>$p){ $probs[$i]=$p/$sum; }
    return $probs;
}
function top_k_filter(array &$probs, int $k){
    if ($k<=0 || $k>=count($probs)) return;
    arsort($probs);
    $top=array_slice($probs,0,$k,true);
    $sum=array_sum($top) ?: 1e-12;
    $new=array_fill(0,count($probs),0.0);
    foreach($top as $i=>$p){ $new[$i]=$p/$sum; }
    $probs=$new;
}
function top_p_filter(array &$probs, float $p_keep){
    $idx=[]; foreach($probs as $i=>$p){ $idx[]=['i'=>$i,'p'=>$p]; }
    usort($idx, fn($a,$b)=>$b['p']<=>$a['p']);
    $acc=0.0; $keep=[]; foreach($idx as $it){ $keep[]=$it['i']; $acc+=$it['p']; if($acc>=$p_keep) break; }
    $sum=0.0; foreach($keep as $i){ $sum+=$probs[$i]; } if($sum<=0)$sum=1e-12;
    $new=array_fill(0,count($probs),0.0);
    foreach($keep as $i){ $new[$i]=$probs[$i]/$sum; }
    $probs=$new;
}
function sample_from_probs(array $probs): int {
    $r=mt_rand()/mt_getrandmax(); $acc=0.0;
    foreach($probs as $i=>$p){ $acc+=$p; if($r<=$acc) return $i; }
    end($probs); return key($probs) ?? 0;
}

// ---------- attention block ----------
function attention_block(array $X, array $Wqkv, array $bqkv, array $Wo, array $bo, int $n_heads, int $head_dim): array {
    $L=count($X); $d=count($X[0]??[0]);
    $QKV = add_bias(matmul($X, $Wqkv), $bqkv); // (L x 3d)
    $q=zeros($L,$d); $k=zeros($L,$d); $v=zeros($L,$d);
    for($i=0;$i<$L;$i++){ for($j=0;$j<$d;$j++){ $q[$i][$j]=$QKV[$i][$j]; }}
    for($i=0;$i<$L;$i++){ for($j=0;$j<$d;$j++){ $k[$i][$j]=$QKV[$i][$j+$d]; }}
    for($i=0;$i<$L;$i++){ for($j=0;$j<$d;$j++){ $v[$i][$j]=$QKV[$i][$j+2*$d]; }}

    $scale = 1.0 / sqrt($head_dim);
    $H = zeros($L,$d);
    for($h=0;$h<$n_heads;$h++){
        $qs=zeros($L,$head_dim); $ks=zeros($L,$head_dim); $vs=zeros($L,$head_dim);
        for($i=0;$i<$L;$i++){ for($j=0;$j<$head_dim;$j++){
            $qs[$i][$j]=$q[$i][$h*$head_dim+$j];
            $ks[$i][$j]=$k[$i][$h*$head_dim+$j];
            $vs[$i][$j]=$v[$i][$h*$head_dim+$j];
        }}
        $scores=zeros($L,$L);
        for($i=0;$i<$L;$i++){
            for($j=0;$j<=$i;$j++){
                $s=0.0; for($t=0;$t<$head_dim;$t++){ $s+=$qs[$i][$t]*$ks[$j][$t]; }
                $scores[$i][$j]=$s*$scale;
            }
            for($j=$i+1;$j<$L;$j++){ $scores[$i][$j]=-1e30; }
        }
        $probs=zeros($L,$L);
        for($i=0;$i<$L;$i++){ $probs[$i]=softmax_row($scores[$i]); }
        $ctx=zeros($L,$head_dim);
        for($i=0;$i<$L;$i++){
            for($t=0;$t<$head_dim;$t++){
                $sum=0.0; for($j=0;$j<$L;$j++){ $sum+=$probs[$i][$j]*$vs[$j][$t]; }
                $ctx[$i][$t]=$sum;
            }
        }
        for($i=0;$i<$L;$i++){ for($j=0;$j<$head_dim;$j++){ $H[$i][$h*$head_dim+$j]=$ctx[$i][$j]; }}
    }
    return add_bias(matmul($H,$Wo), $bo);
}

function forward_logits(array $tokens, array $W): array {
    $vocab=$W['meta']['vocab_size']; $d=$W['meta']['d_model'];
    $nL=$W['meta']['n_layers']; $nH=$W['meta']['n_heads']; $maxL=$W['meta']['max_seq_len'];
    $hd=intdiv($d,$nH);
    $L=count($tokens); if($L>$maxL){ $tokens=array_slice($tokens,-$maxL); $L=$maxL; }

    $X=zeros($L,$d);
    for($i=0;$i<$L;$i++){
        $tid=$tokens[$i] % $vocab;
        $te=$W['tok_emb'][$tid]; $pe=$W['pos_emb'][$i];
        for($j=0;$j<$d;$j++) $X[$i][$j]=$te[$j]+$pe[$j];
    }
    for($l=0;$l<$nL;$l++){
        $blk=$W['blocks'][$l];
        $Ain = layernorm($X, $blk['ln1_g'], $blk['ln1_b']);
        $Aout= attention_block($Ain, $blk['Wqkv'], $blk['bqkv'], $blk['Wo'], $blk['bo'], $nH, $hd);
        for($i=0;$i<$L;$i++){ for($j=0;$j<$d;$j++){ $X[$i][$j]+=$Aout[$i][$j]; }}

        $Min = layernorm($X, $blk['ln2_g'], $blk['ln2_b']);
        $H   = add_bias(matmul($Min, $blk['W1']), $blk['b1']); $H=gelu($H);
        $Mout= add_bias(matmul($H, $blk['W2']),  $blk['b2']);
        for($i=0;$i<$L;$i++){ for($j=0;$j<$d;$j++){ $X[$i][$j]+=$Mout[$i][$j]; }}
    }
    $X=layernorm($X, $W['ln_f_g'], $W['ln_f_b']);
    $h=$X[$L-1];

    // logits = h @ Wte^T
    $logits=array_fill(0,$vocab,0.0);
    for($v=0;$v<$vocab;$v++){
        $dot=0.0; $emb=$W['tok_emb'][$v];
        for($j=0;$j<$d;$j++) $dot+=$h[$j]*$emb[$j];
        $logits[$v]=$dot;
    }
    return $logits;
}

// ---------- main ----------
$body = json_body();
$model_path = (string)($body['model'] ?? '');
$prompt     = (string)($body['prompt'] ?? '');
$max_new    = max(1, min(512, intval($body['max_new_tokens'] ?? 64)));
$temp       = clamp(floatval($body['temperature'] ?? 0.9), 0.0, 5.0);
$top_k      = intval($body['top_k'] ?? 40);
$top_p      = floatval($body['top_p'] ?? 0.95);

if ($model_path==='') http_error(400, "'model' is required");
$s = file_get_contents($model_path);
if ($s===false) http_error(400, "Cannot read model file");
$W = json_decode($s, true);
if (!is_array($W) || !isset($W['meta'])) http_error(400, "Bad model JSON");

// load tokenizer (Models/<base>_tokenizer.json)
$base = preg_replace('/\.json$/','', basename($model_path));
$tpath = dirname($model_path) . DIRECTORY_SEPARATOR . $base . '_tokenizer.json';
if (!is_file($tpath)) http_error(400, "Tokenizer not found: $tpath");
$bpe = new BPE($tpath);

// encode prompt with BPE
$ctx = $bpe->encode($prompt);
if (count($ctx)==0) $ctx = [$W['meta']['vocab_size']-1]; // fallback: last id (arbitrary)

$generated = [];
for ($t=0; $t<$max_new; $t++){
    $tokens = array_merge($ctx, $generated);
    $logits = forward_logits($tokens, $W);
    $probs  = logits_to_probs($logits, $temp);
    if ($top_k>0) top_k_filter($probs, $top_k);
    if ($top_p>0.0 && $top_p<1.0) top_p_filter($probs, $top_p);
    $next = sample_from_probs($probs);
    $generated[] = $next;
}

$out_text = $bpe->decode($generated);
header('Content-Type: application/json; charset=utf-8');
echo json_encode(['ok'=>true,'text'=>$out_text,'tokens_generated'=>count($generated)], JSON_UNESCAPED_UNICODE);
