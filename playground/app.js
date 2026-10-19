/* 1ニューロン自己回帰CoT — 構成の JS 再実装 (ブラウザ/Node 兼用)。
 *
 * python (one_neuron_cot/, arXiv:2610.17283 App. C.1) と同一の構成を
 * 依存なしの素の JS で再現する。python が生成した正解トークン列
 * (fixtures.js) との相互検証は playground/selftest.js を参照。
 *
 * 構成: t_i = a_{s+i}, t'_j = a_j (a は Sidon/Golomb 集合)、
 * 指定ステップ t_i でゲート v_i が閾値で発火し、それ以外のステップは
 * サプレッション値で 0 になる。トークン列の末尾は最終ゲートの値。
 */
(function (root) {
  "use strict";

  // ---- Sidon/Golomb 集合 (sidon.py と同一のレシピ) ----
  function nextPrime(m) {
    function isPrime(v) {
      if (v < 2) return false;
      if (v % 2 === 0) return v === 2;
      for (var d = 3; d * d <= v; d += 2) { if (v % d === 0) return false; }
      return true;
    }
    var p = Math.max(m, 2);
    while (!isPrime(p)) p += 1;
    return p;
  }

  function sidonGolomb(m) {
    if (m === 1) return [1];
    var p = nextPrime(m), B = 2 * m - 1;
    var vals = [];
    for (var x = 0; x < m; x++) vals.push(x + B * ((x * x) % p));
    vals.sort(function (a, b) { return a - b; });
    return vals.map(function (v) { return v + 1; });
  }

  // ---- 回路ビルダー (circuits.py と同一) ----
  function gRef(j) { return ["g", j]; }
  function xRef(q) { return ["x", q]; }

  function parity(n) {
    var gates = [], cur = xRef(0);
    for (var i = 1; i < n; i++) {
      gates.push({ name: "x" + i + "_or", inputs: [[cur, 1], [xRef(i), 1]], bias: -1 });
      var oi = gates.length - 1;
      gates.push({ name: "x" + i + "_and", inputs: [[cur, 1], [xRef(i), 1]], bias: -2 });
      var hi = gates.length - 1;
      gates.push({ name: "x" + i + "_xor", inputs: [[gRef(oi), 1], [gRef(hi), -2]], bias: -1 });
      cur = gRef(gates.length - 1);
    }
    return { n: n, name: "parity" + n, gates: gates, activation: "threshold" };
  }

  function adder(k) {
    var gates = [];
    gates.push({ name: "and0", inputs: [[xRef(0), 1], [xRef(k), 1]], bias: -2 });
    var and0 = gates.length - 1;
    gates.push({ name: "or0", inputs: [[xRef(0), 1], [xRef(k), 1]], bias: -1 });
    var or0 = gates.length - 1;
    gates.push({ name: "s0", inputs: [[gRef(or0), 1], [gRef(and0), -2]], bias: -1 });
    var carry = gRef(and0);
    for (var i = 1; i < k; i++) {
      gates.push({ name: "c" + (i + 1), inputs: [[xRef(i), 1], [xRef(k + i), 1], [carry, 1]], bias: -2 });
      var maj = gates.length - 1;
      gates.push({ name: "s" + i, inputs: [[xRef(i), 1], [xRef(k + i), 1], [carry, 1], [gRef(maj), -2]], bias: -1 });
      carry = gRef(maj);
    }
    return { n: 2 * k, name: "adder" + k, gates: gates, activation: "threshold" };
  }

  function majority(n) {
    var inputs = [];
    for (var q = 0; q < n; q++) inputs.push([xRef(q), 1]);
    return { n: n, name: "maj" + n,
      gates: [{ name: "maj", inputs: inputs, bias: -(Math.floor(n / 2) + 1) }],
      activation: "threshold" };
  }

  // ---- 構成 (build_neuron に対応) ----
  function buildPlan(circuit, margin) {
    margin = (margin === undefined) ? 0.5 : margin;
    var n = circuit.n, s = circuit.gates.length;
    var a = sidonGolomb(2 * s);
    var t = a.slice(s), tp = a.slice(0, s);
    var T = t[s - 1];
    var Tprime = T + tp[s - 1];
    var k = Math.ceil(n / s);
    var d = 2 * T + k * Tprime;
    var edges = circuit.gates.map(function (g) {
      return g.inputs.map(function (pr) {
        var src = pr[0], w = pr[1];
        if (src[0] === "x") return { kind: "x", idx: src[1], w: w };
        if (src[0] === "g") return { kind: "g", idx: src[1], w: w };
        return { kind: "one", idx: 0, w: w };
      });
    });
    var designated = [];
    for (var i = 0; i < s; i++) designated.push(t[i]);
    return {
      name: circuit.name, n: n, s: s, gates: circuit.gates, edges: edges,
      biases: circuit.gates.map(function (g) { return g.bias + margin; }),
      t: t, tp: tp, T: T, Tprime: Tprime, k: k, d: d,
      designated: designated, margin: margin, activation: circuit.activation,
    };
  }

  // ---- 自己回帰シミュレーション (run_fast の意味論) ----
  // 指定ステップ t_i: preact = (theta_i + margin) + sum 辺の寄与
  //   x_q -> 入力ビット, g_j -> 既に生成済みの指定トークン C_j
  // それ以外のステップ: サプレッション値で強制的に 0 を出力
  function simulate(plan, bits) {
    var gateVals = [], tokens = [], steps = [];
    var i;
    for (i = 0; i < plan.s; i++) gateVals.push(0);
    var desAt = {};
    for (i = 0; i < plan.s; i++) desAt[plan.designated[i]] = i;
    for (var tau = 1; tau <= plan.T; tau++) {
      if (Object.prototype.hasOwnProperty.call(desAt, tau)) {
        var gi = desAt[tau];
        var pre = plan.biases[gi], terms = [];
        var es = plan.edges[gi];
        for (var e = 0; e < es.length; e++) {
          var ed = es[e], val, label;
          if (ed.kind === "x") { val = bits[ed.idx]; label = "x" + (ed.idx + 1); }
          else if (ed.kind === "g") { val = gateVals[ed.idx]; label = plan.gates[ed.idx].name; }
          else { val = 1; label = "1"; }
          var prod = ed.w * val;
          terms.push({ label: label, w: ed.w, val: val, prod: prod });
          pre += prod;
        }
        var out = pre >= 0 ? 1 : 0;
        gateVals[gi] = out;
        tokens.push(out);
        steps.push({ tau: tau, gate: gi, gateName: plan.gates[gi].name,
                     theta: plan.biases[gi], terms: terms, pre: pre, out: out });
      } else {
        tokens.push(0);
      }
    }
    return { tokens: tokens, steps: steps, gateVals: gateVals };
  }

  // ---- 独立実装の回路評価 (検証用の比較対象) ----
  function evalCircuit(plan, bits) {
    var vals = [];
    for (var i = 0; i < plan.s; i++) {
      var pre = plan.gates[i].bias, ins = plan.gates[i].inputs;
      for (var e = 0; e < ins.length; e++) {
        var src = ins[e][0], w = ins[e][1];
        if (src[0] === "x") pre += w * bits[src[1]];
        else if (src[0] === "g") pre += w * vals[src[1]];
        else pre += w;
      }
      vals.push(pre >= 0 ? 1 : 0);
    }
    return vals;
  }

  // ---- 全 2^n 入力での網羅検証 ----
  function checkAll(plan) {
    var n = plan.n, total = 1 << n, e2e = 0, prop = 0;
    for (var v = 0; v < total; v++) {
      var b = [];
      for (var q = 0; q < n; q++) b.push((v >> q) & 1);
      var sim = simulate(plan, b), gv = evalCircuit(plan, b);
      if (sim.gateVals[plan.s - 1] === gv[plan.s - 1]) e2e += 1;
      var okP = true;
      for (var gi = 0; gi < plan.s; gi++) {
        if (sim.tokens[plan.designated[gi] - 1] !== gv[gi]) okP = false;
      }
      if (okP) prop += 1;
    }
    return { total: total, e2e: e2e, prop: prop };
  }

  // ---- python フィクスチャとの相互検証 ----
  function checkFixtures(fixtures) {
    var builders = { parity: parity, adder: adder, maj: majority };
    var out = [];
    for (var i = 0; i < fixtures.programs.length; i++) {
      var p = fixtures.programs[i];
      var plan = buildPlan(builders[p.kind](p.param));
      var structOk = (plan.d === p.d && plan.T === p.T);
      var match = 0;
      for (var j = 0; j < p.samples.length; j++) {
        var sim = simulate(plan, p.samples[j].bits);
        if (JSON.stringify(sim.tokens) === JSON.stringify(p.samples[j].tokens)) match += 1;
      }
      out.push({ program: p.kind + p.param, structOk: structOk, match: match,
                 samples: p.samples.length });
    }
    return out;
  }

  var OneNeuron = {
    nextPrime: nextPrime, sidonGolomb: sidonGolomb,
    parity: parity, adder: adder, majority: majority,
    buildPlan: buildPlan, simulate: simulate, evalCircuit: evalCircuit,
    checkAll: checkAll, checkFixtures: checkFixtures,
  };
  root.OneNeuron = OneNeuron;
  if (typeof module !== "undefined" && module.exports) module.exports = OneNeuron;
})(typeof window !== "undefined" ? window : globalThis);
