/**
 * 驗證 simulator.html 的取樣+內插近似，與完整積分 300 顆墨滴的誤差。
 *
 * 用法：node web/verify-interpolation.js
 *
 * 頁面為了維持流暢動畫，只積分 NS=31 個取樣噴嘴再沿陣列內插。
 * 這支腳本抽出頁面的物理核心，對照 full=true（完整 300 顆）的結果，
 * 確認近似誤差可忽略。同時列印基準案例的數值，供與 Python 核心比對。
 */
const fs = require("fs");
const path = require("path");
const vm = require("vm");

const html = fs.readFileSync(path.join(__dirname, "simulator.html"), "utf8");
const js = html.match(/<script>\n([\s\S]*)\n<\/script>/)[1];

// 只取到繪圖工具之前 —— 之後的程式碼需要 DOM。切在該區塊的分隔線上。
const marker = js.indexOf("// 繪圖工具");
if (marker < 0) throw new Error("找不到繪圖工具區塊分隔線");
const core = js.slice(0, js.lastIndexOf("// ====", marker));

const ctx = { console, performance: require("perf_hooks").performance };
vm.createContext(ctx);
// const/let 宣告不會成為 context 的屬性，需顯式掛到 globalThis
vm.runInContext(
  core + "\nglobalThis.__core = { P, belt, beltState, wind, fireBurst, burst," +
         " rebuildNozzles, gapSpeedStats };\n",
  ctx);
const { P, belt, beltState, wind, fireBurst, burst, rebuildNozzles, gapSpeedStats } = ctx.__core;

// --- 基準案例：350 m/min、4mm、30pL、箱體中段（穩態 Couette） ---
P.speedMMin = 350; P.mass = 20; P.gapMm = 4; P.dropletPl = 30;
P.spacingCm = 40; P.eject = 8; P.printHeightIn = 1.0;
rebuildNozzles();

belt.vBelt = P.beltSpeed;
belt.vBox = P.beltSpeed;
belt.dist = P.boxLen * 0.5;
belt.simTime = belt.dist / P.beltSpeed;
const st = beltState();

function snapshot() {
  return { x: Float64Array.from(burst.driftX), y: Float64Array.from(burst.driftY) };
}

const t0 = performance.now();
fireBurst(st, true);            // 完整 300 顆
const tFull = performance.now() - t0;
const exact = snapshot();

const t1 = performance.now();
fireBurst(st, false);           // 取樣 31 顆 + 內插
const tFast = performance.now() - t1;
const approx = snapshot();

let maxErrX = 0, maxErrY = 0, maxAbsX = 0, maxAbsY = 0;
for (let i = 0; i < P.nNozzles; i++) {
  maxErrX = Math.max(maxErrX, Math.abs(approx.x[i] - exact.x[i]));
  maxErrY = Math.max(maxErrY, Math.abs(approx.y[i] - exact.y[i]));
  maxAbsX = Math.max(maxAbsX, Math.abs(exact.x[i]));
  maxAbsY = Math.max(maxAbsY, Math.abs(exact.y[i]));
}

const w = [0, 0, 0];
const bc = [];
for (const [z, name] of [[0, "z=0  箱頂面"], [st.gap / 2, "z=h/2 中線"], [st.gap, "z=h  噴嘴板"]]) {
  wind(0, 0, z, st, w);
  bc.push(`    ${name}: ux=${w[0].toFixed(6)}  uy=${w[1].toExponential(2)}  uz=${w[2].toExponential(2)}`);
}
const stats = gapSpeedStats(st);

console.log(`
=== 邊界條件（拘束因子 c=${st.conf.toFixed(4)}）===
${bc.join("\n")}
    間隙平均風速 ${stats.mean.toFixed(4)} m/s   （理論 U/2 = ${(P.beltSpeed / 2).toFixed(4)}）

=== 基準案例（完整 300 顆積分）===
    噴發間隙      ${(st.gap * 1000).toFixed(4)} mm
    飛行時間      ${(burst.flightTime * 1e3).toFixed(4)} ms
    Δx            ${(maxAbsX * 1e6).toFixed(3)} µm
    Δy 兩端       ±${(maxAbsY * 1e6).toFixed(3)} µm
    換算點距      ${(Math.hypot(maxAbsX, maxAbsY) / P.nozzlePitch).toFixed(4)}

=== 取樣(${31})+內插 vs 完整 300 顆 ===
    Δx 最大誤差   ${(maxErrX * 1e9).toFixed(2)} nm  (${(maxErrX / maxAbsX * 100).toFixed(4)}%)
    Δy 最大誤差   ${(maxErrY * 1e9).toFixed(2)} nm  (${(maxErrY / maxAbsY * 100).toFixed(4)}%)
    耗時          完整 ${tFull.toFixed(1)} ms  ->  取樣 ${tFast.toFixed(1)} ms  (${(tFull / tFast).toFixed(1)}x)
`);

const okX = maxErrX / maxAbsX < 0.01;
const okY = maxErrY / maxAbsY < 0.01;
console.log(okX && okY ? "PASS：內插誤差 < 1%" : "FAIL：內插誤差過大");
process.exit(okX && okY ? 0 : 1);
