/* node playground/selftest.js — JS再実装の自己検証 */
"use strict";
const app = require("./app.js");
const fixtures = require("./fixtures.js");

let ok = true;
const checks = app.checkFixtures(fixtures);
for (const c of checks) {
  console.log(c.program.padEnd(8) +
    " struct=" + (c.structOk ? "OK" : "NG") +
    " fixtures=" + c.match + "/" + c.samples);
  ok = ok && c.structOk && c.match === c.samples;
}
// 代表プログラムの網羅検証も node で実施
const coverage = [["parity", 4], ["adder", 3], ["maj", 5], ["parity", 8], ["adder", 4]];
const builders = { parity: app.parity, adder: app.adder, maj: app.majority };
for (const [kind, n] of coverage) {
  const plan = app.buildPlan(builders[kind](n));
  const c = app.checkAll(plan);
  console.log((kind + n).padEnd(8) +
    " d=" + plan.d + " T=" + plan.T +
    " e2e=" + c.e2e + "/" + c.total + " prop=" + c.prop + "/" + c.total);
  ok = ok && c.e2e === c.total && c.prop === c.total;
}
console.log(ok ? "ALL PASS" : "FAIL");
process.exit(ok ? 0 : 1);
