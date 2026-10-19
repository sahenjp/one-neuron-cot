/* 1ニューロン文字レベル言語モデル (ブラウザ/Node 兼用)。
 *
 * コーパスから「次の1文字」を予測する単一ニューロン (重みベクトル w 1本) を
 * パーセプトロンで学習し、自己回帰に文字を生成する。
 *
 * - 学習: 各位置で「次の文字」をBビットの二進コードに分解し、各ビットを
 *   スロット (前々文字,前文字,ビット位置) / (前文字,ビット位置) の1hot窓で
 *   予測するパーセプトロン (コーパス全体で w 1本を共有)。
 * - 生成: 候補文字 c のエネルギー E(c) = Σ_j w[スロット_j]・(2bit_j(c)-1) を
 *   全語彙で採点し、温度つきソフトマックスでサンプリング (temp=0 で決定論)。
 *   サンプリング乱数は位置のハッシュで決定論化してある (再生成で接頭辞不変)。
 */
(function (root) {
  "use strict";

  function buildLM(text, opts) {
    opts = opts || {};
    var maxEpochs = opts.maxEpochs || 80;
    var chars = [], index = {}, freq = [];
    var i, ch;
    for (i = 0; i < text.length; i++) {
      ch = text[i];
      if (!(ch in index)) { index[ch] = chars.length; chars.push(ch); freq.push(0); }
      freq[index[ch]]++;
    }
    var D = chars.length;
    var B = 1;
    while ((1 << B) < D) B++;
    var S0 = D * D * B;
    var F = S0 + D * B;
    var mostFreq = 0;
    for (i = 1; i < D; i++) if (freq[i] > freq[mostFreq]) mostFreq = i;

    function codeOf(idx) {
      var a = [];
      for (var b = 0; b < B; b++) a.push((idx >> (B - 1 - b)) & 1);
      return a;
    }
    function pairIdx(c2, c1, j) { return (c2 * D + c1) * B + j; }
    function singleIdx(c1, j) { return S0 + c1 * B + j; }
    function dot(f, w) {
      var s = 0;
      for (var k = 0; k < f.length; k++) s += f[k].v * w[f[k].i];
      return s;
    }

    var samples = [];
    for (i = 2; i < text.length; i++) {
      samples.push({ c2: index[text[i - 2]], c1: index[text[i - 1]],
                     code: codeOf(index[text[i]]) });
    }

    var w = new Float64Array(F);
    var trained = new Uint8Array(F);
    var bestW = Float64Array.from(w);
    var bestErr = Infinity, epochs = 0;
    var ep, sI, jI, kI;
    for (ep = 0; ep < maxEpochs; ep++) {
      var err = 0, errS = 0;
      epochs = ep + 1;
      // ペア文脈と1文字文脈は同一 w 内の別領域で独立学習させる
      // (特徴が互いに素なので交叉干渉がなく、ペア重みは正しい多数決へ収束する)
      for (sI = 0; sI < samples.length; sI++) {
        var sm = samples[sI];
        for (jI = 0; jI < B; jI++) {
          var y = sm.code[jI];
          var sign = y === 1 ? 1 : -1;
          var iP = pairIdx(sm.c2, sm.c1, jI);
          var iS = singleIdx(sm.c1, jI);
          var predP = dot([{ i: iP, v: 1 }], w) >= 0 ? 1 : 0;
          var predS = dot([{ i: iS, v: 1 }], w) >= 0 ? 1 : 0;
          if (predP !== y) err++;
          if (predS !== y) errS++;
          // 無条件の多数決カウンタ更新 (+1 / -1):
          // label=1 のサンプルも必ず重みを動かし、重み0放置の偏りを防ぐ
          w[iP] += sign;
          w[iS] += sign;
          trained[iP] = 1; trained[iS] = 1;
        }
      }
      if (err <= bestErr) { bestErr = err; bestW = Float64Array.from(w); }
      if (err === 0) break;
    }
    var total = samples.length * B;
    var acc = total ? 1 - bestErr / total : 0;

    function startPair(prompt) {
      var idxs = [];
      for (var p = prompt.length - 1; p >= 0 && idxs.length < 2; p--) {
        var pc = prompt[p];
        if (pc in index) idxs.unshift(index[pc]);
      }
      while (idxs.length < 2) idxs.unshift(index[chars[idxs.length % chars.length]]);
      return idxs;
    }
    function hash01(a, b, c) {
      var h = (Math.imul(a ^ 0x9E3779B9, 668265263) ^
               Math.imul(b + 1, 2246822519) ^
               Math.imul(c + 1, 3266489917)) >>> 0;
      return (h & 0x7fffffff) / 0x7fffffff;
    }

    function generate(prompt, maxChars, opts2) {
      maxChars = maxChars || 60;
      opts2 = opts2 || {};
      var T = opts2.temp === undefined ? 0.7 : opts2.temp;
      var pair = startPair(prompt);
      var out = "", log = [];
      for (var t2 = 0; t2 < maxChars; t2++) {
        var wj = [];
        var usePair = true, j2;
        for (j2 = 0; j2 < B; j2++) if (!trained[pairIdx(pair[0], pair[1], j2)]) usePair = false;
        for (j2 = 0; j2 < B; j2++) {
          var idxW = usePair ? pairIdx(pair[0], pair[1], j2) : singleIdx(pair[1], j2);
          wj.push(trained[idxW] ? bestW[idxW] : 0);
        }
        var logits = [], Z = 0, mx = -Infinity, bestC = 0;
        for (var c = 0; c < D; c++) {
          var sc = 0;
          for (j2 = 0; j2 < B; j2++) sc += wj[j2] * (2 * ((c >> (B - 1 - j2)) & 1) - 1);
          logits.push(sc);
          if (sc > mx) { mx = sc; bestC = c; }
        }
        var pick = bestC;
        if (T > 0) {
          var sum = 0, probs = [];
          for (c = 0; c < D; c++) {
            var e = Math.exp((logits[c] - mx) / T);
            probs.push(e); sum += e;
          }
          var r = hash01(t2, pair[0], pair[1]) * sum, acc2 = 0;
          for (c = 0; c < D; c++) { acc2 += probs[c]; if (r <= acc2) { pick = c; break; } }
        }
        var ch2 = chars[pick];
        out += ch2;
        var order = [];
        for (c = 0; c < D; c++) order.push(c);
        order.sort(function (a, b) {
          return (logits[b] - logits[a]) || (freq[b] - freq[a]);
        });
        var top = [];
        for (c = 0; c < Math.min(3, D); c++) top.push([chars[order[c]], logits[order[c]]]);
        log.push({ pair: [chars[pair[0]], chars[pair[1]]], ch: ch2, top: top,
                   bigram: usePair, scored: D });
        pair = [pair[1], pick];
      }
      return { text: out, log: log };
    }

    return { D: D, B: B, F: F, chars: chars, w: bestW,
             epochs: epochs, bestErr: bestErr, samples: samples.length,
             acc: acc, generate: generate, mostFreq: chars[mostFreq] };
  }

  var api = { buildLM: buildLM };
  root.OneNeuronLM = api;
  if (typeof module !== "undefined" && module.exports) module.exports = api;
})(typeof window !== "undefined" ? window : globalThis);