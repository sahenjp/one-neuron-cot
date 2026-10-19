/* 構成型1ニューロン文章ジェネレータ (ブラウザ/Node 兼用)。
 *
 * 学習データなし。重みベクトル w はコードで構成する (Theorem 2 流)。
 * w = [各フレーズの基本スコア] + [フレーズ x トピックのアフィニティ]
 * を1本の Float64Array に連結したもの。入力プロンプトから辞書で
 * トピックを検出し、トピック1hot を特徴として
 *   E(c) = w_base[c] + sum_t w_aff[c][t] * x_t
 * を全フレーズで採点する。同一の w を全スロットで再利用する。
 */
(function (root) {
  "use strict";

  var TOPICS = ["雨", "晴", "空", "夜", "星", "月", "音楽", "珈琲", "茶", "本", "散歩", "道", "街", "風", "朝", "窓", "手紙", "仕事", "笑い", "喜", "疲", "冬", "夏", "花", "海", "山"];
  var TIDX = {};
  TOPICS.forEach(function (t, i) { TIDX[t] = i; });

  var LEX = {};
  function lex(keys, topic) {
    keys.split(" ").forEach(function (k) { if (k) LEX[k] = topic; });
  }
  lex("雨 降 傘 濡 しとしと 梅雨", "雨");
  lex("晴 快晴 青空 カラッ", "晴");
  lex("空 雲 夕焼", "空");
  lex("夜 夜更 月 眠 深夜 星ふ", "夜");
  lex("星 星空 流れ星", "星");
  lex("月 満月 三日月", "月");
  lex("音楽 曲 プレイリ スピーカ 耳 コード", "音楽");
  lex("珈琲 コーヒ 豆 淹 ドリップ", "珈琲");
  lex("茶 お茶 緑茶 煎", "茶");
  lex("本 読書 積読 ペー ジ 文庫", "本");
  lex("散歩 歩 つ てくてく", "散歩");
  lex("道 路 角", "道");
  lex("街 町 駅 人混", "街");
  lex("風 そよ 吹", "風");
  lex("朝 朝食 早朝 起", "朝");
  lex("窓", "窓");
  lex("手紙 便棺 iOSカード", "手紙");
  lex("仕事 仕 事 デスク 職 タスク", "仕事");
  lex("笑 笑い 声が 上機嫌", "笑い");
  lex("うれし 嬉 楽しい わくわ 楽しみ", "喜");
  lex("疲 だる 重い 頭が", "疲");
  lex("冬 寒 こた 雪コート", "冬");
  lex("夏 暑 蝉 麦茶", "夏");
  lex("花 咲 桜 花びら", "花");
  lex("海 海辺 砂浜 潮", "海");
  lex("山 登り 樹 森", "山");
  function p(t, base, want) { return { t: t, base: base, want: want }; }
  var ROLES = [
    { name: "背景", parts: [
      p("けさは", 2.4, ["朝"]),
      p("窓の外を見ると", 2.2, ["窓"]),
      p("ふと気づけば", 2.0, []),
      p("空のようすが変わって", 2.2, ["空", "晴"]),
      p("静かな午後で", 2.0, []),
      p("町の音を聞きながら", 2.1, ["街"]),
      p("予定より早く目が覚めて", 1.8, ["朝"]),
      p("冷たい風が通り抜けて", 2.1, ["風", "冬"]),
      p("長い一日の終わりに", 1.9, ["夜"]),
      p("海のほうから風が来て", 1.8, ["海"])
    ] },
    { name: "気分", parts: [
      p("気分は上向きで", 2.2, ["喜"]),
      p("少しだけ余裕があって", 2.0, []),
      p("なんとなく落ち着かなくて", 1.9, ["疲"]),
      p("ほっとひと息ついて", 2.0, []),
      p("背中が伸びるような気分で", 2.1, ["喜"]),
      p("少しだけ眠くて", 1.8, ["夜", "疲"]),
      p("静かな高揚感があって", 2.0, ["喜"]),
      p("肩の力がようやく抜けて", 2.0, ["疲"]),
      p("心の声がうるさくて", 1.8, []),
      p("自分でも理由は分からず上機嫌で", 1.9, ["喜"])
    ] },
    { name: "行動", parts: [
      p("お気に入りの曲をかけ直して", 2.4, ["音楽"]),
      p("雨音をBGMに", 2.4, ["雨"]),
      p("珈琲を淹れ直して", 2.4, ["珈琲"]),
      p("積んでいた本を開いて", 2.4, ["本"]),
      p("いつもと違う道を歩いて", 2.3, ["散歩", "道"]),
      p("温かいお茶を入れて", 2.2, ["茶", "冬"]),
      p("夜空をしばらく眺めて", 2.4, ["夜", "星", "月"]),
      p("机の上を一度だけ片付けて", 2.2, ["仕事"]),
      p("窓を少し開けて", 2.2, ["窓"]),
      p("書きかけの手紙を広げて", 2.2, ["手紙"]),
      p("花の写真を眺めて", 2.1, ["花"]),
      p("海の音を思い出して", 2.1, ["海"])
    ] },
    { name: "詳細", parts: [
      p("細部まで時間をかけて", 2.0, []),
      p("全体の7割で切り上げて", 1.9, []),
      p("紙に書き出してみたら", 2.1, ["仕事"]),
      p("思ったより早く終わって", 1.9, []),
      p("途中で方針を変えてみて", 1.9, []),
      p("小さな一歩に分解してみたら", 2.1, ["仕事"]),
      p("完璧を目指さずに進めて", 1.9, []),
      p("五分だけ続けてみたら", 2.0, []),
      p("記録をひとつ残してみたら", 1.8, []),
      p("途中の風景を楽しみながら", 2.1, ["散歩", "道"])
    ] },
    { name: "転回", parts: [
      p("気がつけば視界が晴れて", 2.3, ["晴"]),
      p("うまくいかない部分もあって", 2.0, ["疲"]),
      p("思いがけない助け舟が来て", 2.1, []),
      p("逆にそれが正解だったりして", 2.0, []),
      p("雨はまだ降り続いていて", 2.3, ["雨"]),
      p("星がひとつ増えたように見えて", 2.2, ["星", "夜"]),
      p("遠くで笑い声が聞こえて", 2.1, ["笑い", "街"]),
      p("風の向きが変わったのを感じて", 2.2, ["風"]),
      p("月が雲の間から出て", 2.2, ["月", "夜"]),
      p("波の音が近くで重なって", 2.1, ["海", "夏"])
    ] },
    { name: "締め", parts: [
      p("それで十分な一日だった。", 2.0, []),
      p("今日はここで手を止める。", 1.9, []),
      p("続けられそうな気がする。", 2.0, []),
      p("明日の自分に引き継ごう。", 2.0, []),
      p("小さな発見を持ち帰る。", 2.1, []),
      p("夜はゆっくり長めにとる。", 2.0, ["夜"]),
      p("記録はここまでにしておく。", 1.8, []),
      p("また明日、同じ場所から。", 1.8, []),
      p("もやもやはそっと置いておく。", 1.9, ["疲"]),
      p("そしてまた少しだけ進む。", 1.9, ["喜"])
    ] }
  ];
  // --- 単一重みベクトル w を構成する (学習なし) ---
  // 領域1: base (候補ごとの基本スコア) / 領域2: affinity (候補 x 26 トピック)
  var WEIGHTS = (function () {
    var base = [], rows = [], roleOff = [], k = 0;
    ROLES.forEach(function (role) {
      roleOff.push(k);
      role.parts.forEach(function (part) {
        base.push(part.base);
        rows.push(TOPICS.map(function (t) { return part.want.indexOf(t) >= 0 ? 3.2 : 0; }));
        k++;
      });
    });
    var flat = base.slice();
    rows.forEach(function (row) { row.forEach(function (v) { flat.push(v); }); });
    return { w: Float64Array.from(flat), baseCount: base.length, total: flat.length, roleOff: roleOff };
  })();
  var ROLE_OFF = WEIGHTS.roleOff;
  function hash01(a, b, c) {
    var h = (Math.imul(a ^ 0x9E3779B9, 668265263) ^ Math.imul(b + 1, 2246822519) ^ Math.imul(c + 1, 3266489917)) >>> 0;
    return (h & 0x7fffffff) / 0x7fffffff;
  }
  function promptHash(s) {
    var h = 2166136261 >>> 0;
    for (var i = 0; i < s.length; i++) h = Math.imul(h ^ s.charCodeAt(i), 16777619) >>> 0;
    return h;
  }
  function detectTopics(prompt) {
    var found = [];
    TOPICS.forEach(function (t) { if (prompt.indexOf(t) >= 0 && found.indexOf(t) < 0) found.push(t); });
    Object.keys(LEX).forEach(function (k) {
      if (prompt.indexOf(k) >= 0) {
        var t = LEX[k];
        if (found.indexOf(t) < 0) found.push(t);
      }
    });
    return found.slice(0, 3);
  }
  function compose(prompt, opts) {
    opts = opts || {};
    var sentences = opts.sentences;
    var jw = opts.jitter === undefined ? 1 : (opts.jitter === true ? 1 : (opts.jitter === false ? 0 : opts.jitter));
    var topics = detectTopics(prompt || "");
    var nonce = opts.nonce === undefined ? 0 : opts.nonce;
    var ph = (promptHash(prompt || "") ^ Math.imul(nonce + 1, 2654435761)) >>> 0;
    if (sentences === undefined) sentences = 2 + (hash01(ph, 5, 9) > 0.55 ? 1 : 0);
    var w = WEIGHTS.w, BASE_N = WEIGHTS.baseCount, TN = TOPICS.length;
    var out = "", log = [], prevTail = "", used = {};
    for (var s2 = 0; s2 < sentences; s2++) {
      ROLES.forEach(function (role, ri) {
        var scores = role.parts.map(function (part, ci) {
          var g = ROLE_OFF[ri] + ci;
          var e = w[g];   // base 重み (w 領域1)
          if (used[g]) e -= 2.5;   // 同じフレーズの再使用は減点
          part.want.forEach(function (t) {
            if (topics.indexOf(t) >= 0) e += w[BASE_N + g * TN + TIDX[t]];   // affinity (w 領域2)
          });
          if (jw > 0 && role.name !== "締め") e += (hash01(ph, ri * 31 + ci, s2) - 0.5) * 3.2 * jw;
          return e;
        });
        var best = 0;
        for (var ci2 = 1; ci2 < scores.length; ci2++) if (scores[ci2] > scores[best]) best = ci2;
        if (jw > 0) {
          var pool = [];
          role.parts.forEach(function (part, ci3) { if (scores[ci3] > scores[best] - 1.9) pool.push(ci3); });
          best = pool[Math.floor(hash01(s2 * 97 + ri, ph, 777) * pool.length) % pool.length];
        }
        var chosen = role.parts[best];
        used[ROLE_OFF[ri] + best] = 1;
        var order = role.parts.map(function (part, ci4) { return ci4; });
        order.sort(function (a, b) { return scores[b] - scores[a]; });
        var top = order.slice(0, 3).map(function (ci5) { return [role.parts[ci5].t.slice(0, 6), scores[ci5]]; });
        if (prevTail === ",") out += "、";
        out += chosen.t;
        prevTail = role.name === "締め" ? "。" : ",";
        if (prevTail === "。" && s2 < sentences - 1) out += "\n";
        log.push({ role: role.name, t: chosen.t, top: top, topics: topics.slice(), scored: role.parts.length });
      });
    }
    return { text: out, log: log, topics: topics, weights: WEIGHTS };
  }
  var api = { compose: compose, detectTopics: detectTopics, TOPICS: TOPICS, WEIGHTS: WEIGHTS };
  root.OneNeuronGrammar = api;
  if (typeof module !== "undefined" && module.exports) module.exports = api;
})(typeof window !== "undefined" ? window : globalThis);