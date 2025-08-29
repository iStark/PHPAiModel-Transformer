<?php
// bpe.php — ByteLevel BPE codec compatible with HuggingFace tokenizer.json (BPE + ByteLevel)
// MIT 2025

declare(strict_types=1);

class BPE {
    private array $encoder = [];   // token(string) -> id(int)
    private array $decoder = [];   // id -> token(string)
    private array $bpe_ranks = []; // "a b" -> rank(int)
    private int   $unk_id = -1;

    private array $byte_encoder = []; // byte (0..255) -> unicode char
    private array $byte_decoder = []; // unicode codepoint -> byte (0..255)

    public function __construct(string $tokenizerJsonPath) {
        $raw = @file_get_contents($tokenizerJsonPath);
        if ($raw === false) {
            throw new RuntimeException("Failed to read $tokenizerJsonPath");
        }
        $json = json_decode($raw, true);
        if (!is_array($json)) {
            throw new RuntimeException("Bad JSON in $tokenizerJsonPath");
        }
        if (($json['model']['type'] ?? '') !== 'BPE') {
            throw new RuntimeException("Unsupported tokenizer model: " . ($json['model']['type'] ?? 'unknown'));
        }

        // --- vocab: token -> id
        $vocab = $json['model']['vocab'] ?? [];
        if (!is_array($vocab)) throw new RuntimeException("model.vocab is not an object");
        foreach ($vocab as $tok => $id) {
            $id = (int)$id;
            $this->encoder[$tok] = $id;
            $this->decoder[$id]  = $tok;
        }
        $this->unk_id = (int)($vocab['<unk>'] ?? -1);

        // --- merges: accept both ["A B", ...] and [[ "A","B" ], ...]
        $merges = $json['model']['merges'] ?? [];
        if (!is_array($merges)) throw new RuntimeException("model.merges is not an array");
        $rank = 0;
        foreach ($merges as $m) {
            if (is_array($m)) {
                if (count($m) !== 2) continue;
                $pair = $m[0] . ' ' . $m[1];
            } else {
                $pair = (string)$m; // "A B"
            }
            $this->bpe_ranks[$pair] = $rank++;
        }

        // --- ByteLevel tables (GPT-2 style)
        $bs = [];
        for ($i=33; $i<=126; $i++) $bs[] = $i;
        for ($i=161; $i<=172; $i++) $bs[] = $i;
        for ($i=174; $i<=255; $i++) $bs[] = $i;

        $cs = $bs;
        $n  = 0;
        for ($b=0; $b<256; $b++){
            if (!in_array($b, $bs, true)){
                $bs[] = $b;
                $cs[] = 256 + $n;
                $n++;
            }
        }
        $this->byte_encoder = [];
        $this->byte_decoder = [];
        foreach ($bs as $i => $b) {
            $cp = $cs[$i];
            $ch = $this->utf8_chr($cp);
            $this->byte_encoder[$b] = $ch;
            $this->byte_decoder[$cp] = $b;
        }
    }

    // ===== helper: unicode codepoint -> UTF-8 char
    private function utf8_chr(int $codepoint): string {
        if (function_exists('mb_chr')) return mb_chr($codepoint, 'UTF-8');
        return iconv('UCS-4LE', 'UTF-8', pack('V', $codepoint));
    }
    // helper: UTF-8 char -> codepoint
    private function utf8_ord(string $ch): int {
        if (function_exists('IntlChar::ord')) {
            return \IntlChar::ord($ch);
        }
        if (function_exists('mb_ord')) {
            return mb_ord($ch, 'UTF-8');
        }
        // fallback
        $u = mb_convert_encoding($ch, 'UCS-4LE', 'UTF-8');
        $arr = unpack('V', $u);
        return (int)$arr[1];
    }

    // ===== core BPE =====
    private function get_pairs(array $symbols): array {
        $pairs = [];
        $n = count($symbols);
        for ($i=0; $i<$n-1; $i++){
            $pairs[$symbols[$i].' '.$symbols[$i+1]] = true;
        }
        return $pairs;
    }

    private function bpe_token(string $token): array {
        // split into unicode chars
        $word = preg_split('//u', $token, -1, PREG_SPLIT_NO_EMPTY);
        if (!$word) return [$token];

        $pairs = $this->get_pairs($word);
        if (!$pairs) return [$token];

        while (true) {
            // find best-ranked pair
            $minRank = PHP_INT_MAX; $best = null;
            foreach ($pairs as $p => $_) {
                if (isset($this->bpe_ranks[$p]) && $this->bpe_ranks[$p] < $minRank) {
                    $minRank = $this->bpe_ranks[$p];
                    $best = $p;
                }
            }
            if ($best === null) break;

            [$first, $second] = explode(' ', $best);
            $new_word = [];
            $i = 0; $n = count($word);
            while ($i < $n) {
                $j = $i;
                while ($j < $n) {
                    if ($word[$j] === $first && $j+1 < $n && $word[$j+1] === $second) break;
                    $new_word[] = $word[$j];
                    $j++;
                }
                if ($j >= $n) { $word = $new_word; break; }

                // merge found pair
                $new_word[] = $first.$second;
                $i = $j + 2;

                while ($i < $n) {
                    if ($i+1 < $n && $word[$i] === $first && $word[$i+1] === $second) break;
                    $new_word[] = $word[$i]; $i++;
                }
                $word = $new_word;
                if (count($word) <= 1) break;
            }
            if (count($word) <= 1) break;
            $pairs = $this->get_pairs($word);
        }
        return $word;
    }

    // ===== public API =====
    public function encode(string $text): array {
        // ByteLevel: bytes -> visible unicode via byte_encoder
        $bytes = array_values(unpack('C*', $text)); // [0..255]
        $mapped = '';
        foreach ($bytes as $b) { $mapped .= $this->byte_encoder[$b]; }

        // Run BPE over the whole mapped stream (ByteLevel pretokenizer = charwise)
        $subtokens = $this->bpe_token($mapped);

        // Map to ids
        $ids = [];
        foreach ($subtokens as $t) {
            if (isset($this->encoder[$t])) {
                $ids[] = $this->encoder[$t];
            } elseif ($this->unk_id >= 0) {
                $ids[] = $this->unk_id;
            }
        }
        return $ids;
    }

    public function decode(array $ids): string {
        $s = '';
        foreach ($ids as $id) {
            $tid = (int)$id;
            if (isset($this->decoder[$tid])) $s .= $this->decoder[$tid];
        }
        // ByteLevel: visible unicode back to raw bytes
        $out_bytes = [];
        $chars = preg_split('//u', $s, -1, PREG_SPLIT_NO_EMPTY);
        foreach ($chars as $ch) {
            $cp = $this->utf8_ord($ch);
            if (isset($this->byte_decoder[$cp])) {
                $out_bytes[] = $this->byte_decoder[$cp];
            }
        }
        return pack('C*', ...$out_bytes); // UTF-8
    }
}
