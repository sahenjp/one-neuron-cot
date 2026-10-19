/* node docs/grammar_selftest.js - 構成型ジェネレータの自己検証 */
const g = require("./grammar.js");
const W = g.WEIGHTS;
const okW = W.w instanceof Float64Array && W.total === W.baseCount + W.baseCount * g.TOPICS.length;
const det1 = g.compose("雨が降る", { jitter: false, sentences: 1 }).text;
const det2 = g.compose("雨が降る", { jitter: false, sentences: 1 }).text;
const rain = g.compose("雨が降る", { jitter: false, sentences: 1 }).text;
const p1 = g.compose("雨の夜、珈琲", { sentences: 2 }).text;
const p2 = g.compose("静かに本を読みたい", { sentences: 2 }).text;
const p3 = g.compose("星がきれいな夜", { sentences: 2 }).text;
const uniqN = new Set([p1, p2, p3]).size;
console.log("w elems=" + W.total, "base=" + W.baseCount);
console.log("det  : " + det1);
console.log("雨   : " + rain);
console.log("p1   : " + p1);
console.log("distinct=" + uniqN + "/3");
const ok = okW && det1 === det2 && rain.indexOf("雨") >= 0 && rain.indexOf("BGM") >= 0
  && p1.endsWith("。") && uniqN >= 3 && det1.length >= 20;
console.log(ok ? "SELFTEST PASS" : "SELFTEST FAIL");
process.exit(ok ? 0 : 1);