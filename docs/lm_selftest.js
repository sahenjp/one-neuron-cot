/* node docs/lm_selftest.js - 単一ニューロン文字LMの自己検証 */
const lm = require("./lm.js");
const corpus = "今日は晴れて心が軽い。空は澄みわたって遠くまで見える。午後には雲が出て、風が涼しくなる。雨の日は部屋で本を読むのが好きだ。窓をたたく音を聞きながら、温かい茶を入れる。雨が上がると、匂いが変わる。夜は空に星がにじんで見える。明日はまた別の空が来る。";
const m = lm.buildLM(corpus, { maxEpochs: 80 });
const out = m.generate("今日は", 40);           // 温度0.7 (既定)
const det1 = m.generate("今日は", 40, { temp: 0 });
const det2 = m.generate("今日は", 40, { temp: 0 });
console.log("D=" + m.D, "B=" + m.B, "F=" + m.F, "acc=" + (m.acc * 100).toFixed(1) + "%");
console.log("temp  : " + out.text);
console.log("det   : " + det1.text);
let maxRun = 1, runx = 1;
for (let i = 1; i < out.text.length; i++) {
  runx = out.text[i] === out.text[i - 1] ? runx + 1 : 1;
  if (runx > maxRun) maxRun = runx;
}
const uniq = new Set(out.text.split("")).size;
const detStable = det1.text === det2.text;
const ok = m.acc >= 0.85 && out.text.length === 40 && uniq >= 6 && maxRun <= 8 && detStable;
console.log("uniq=" + uniq, "maxRun=" + maxRun, "detStable=" + detStable, ok ? "SELFTEST PASS" : "SELFTEST FAIL");
process.exit(ok ? 0 : 1);