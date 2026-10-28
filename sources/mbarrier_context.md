# mbarrier 融合图 — 自包含 context

来源：Blackwell FA4（`flash_fwd_sm100.py` warp 分工＋`mbar_*_offset` 令牌）。
以下 SVG 即最终版全文，可直接存成 .svg 修改；配色与 tiling 系列 paste 同体系。

## 图的内容（三层同轴，不同的是“谁在等、谁放行”）
1. 顶：LDS 槽位行 s0/s1（偶数/奇数 tile）。四种状态：搬入蓝 / 满等黄 /
   在吃绿 / 空灰。empty 管“空槽可搬”，full 管“满槽可吃”。
2. 中：五行 warp 时间线（w15 TMA / w12 MMA / w0-7 softmax /
   w8-11 correction / w13-14 epilogue）。彩块＝干活，红块＝睡（块上写 barrier 名）。
   红色竖虚线＝放行时刻，线上标 count（`S 0→1/1`、`sc 0→128/128`），
   线连着哪两行＝谁 arrive、谁在等。
3. 底：13 格 cycle 带（每两拍），格里是本拍点了啥（发L0、MMA0、M0P0…收尾）。

## 关键结论（也在 previous 讨论里）
- mbarrier＝SMEM 里 8 字节计数器：init 写 cap（warp1-7 分头），arrive 涨数，
  try_wait 睡到数够，放行翻 phase 清零下一轮。只认数不认谁。
- 死锁只一种：expected 人数写错，一边永远等不够数。
- full＝能读了，empty＝能写了；被覆写的数据（TMEM S/P、SMEM O 槽）都配一对。

## 配色约定（照抄即可）
- 搬运蓝 `#bfdbfe`、S 蓝 `#93c5fd`、softmax 黄 `#fde68a`、修正紫 `#e9d5ff`、
  PV 橙 `#fed7aa`、收尾深橙 `#fdba74`、睡红 `#fecaca`、放行线深红 `#b91c1c`。
- 画布 1240x700，字体 `Noto Sans CJK SC, PingFang SC, Microsoft YaHei`。

## 复用方法
- 转 PNG：`cairosvg.svg2png(url='mbarrier.svg', write_to='mbarrier.png', scale=2)`。
- AMD 版对照见 `mbarrier_amd.svg`（同一拨 waves 齐步走，无跨队令牌）。

## SVG 全文

```svg
<svg xmlns='http://www.w3.org/2000/svg' width='1240' height='680' viewBox='0 0 1240 680' style='background:#ffffff'>
<text x='30' y='32' font-size='20' font-weight='bold' fill='#0f172a' font-family='Noto Sans CJK SC,PingFang SC,Microsoft YaHei,sans-serif'>cycle 带 ＋ 状态隙：白带干活，灰隙对账（槽/数/调度）</text>
<text x='180' y='85' text-anchor='end' font-size='13' fill='#0f172a' font-family='Noto Sans CJK SC,PingFang SC,Microsoft YaHei,sans-serif'>w15</text>
<text x='180' y='99' text-anchor='end' font-size='10' fill='#64748b' font-family='Noto Sans CJK SC,PingFang SC,Microsoft YaHei,sans-serif'>TMA</text>
<text x='180' y='125' text-anchor='end' font-size='13' fill='#0f172a' font-family='Noto Sans CJK SC,PingFang SC,Microsoft YaHei,sans-serif'>w12</text>
<text x='180' y='139' text-anchor='end' font-size='10' fill='#64748b' font-family='Noto Sans CJK SC,PingFang SC,Microsoft YaHei,sans-serif'>MMA</text>
<text x='180' y='165' text-anchor='end' font-size='13' fill='#0f172a' font-family='Noto Sans CJK SC,PingFang SC,Microsoft YaHei,sans-serif'>w0-3 s0</text>
<text x='180' y='179' text-anchor='end' font-size='10' fill='#64748b' font-family='Noto Sans CJK SC,PingFang SC,Microsoft YaHei,sans-serif'>softmax0</text>
<text x='180' y='205' text-anchor='end' font-size='13' fill='#0f172a' font-family='Noto Sans CJK SC,PingFang SC,Microsoft YaHei,sans-serif'>w4-7 s1</text>
<text x='180' y='219' text-anchor='end' font-size='10' fill='#64748b' font-family='Noto Sans CJK SC,PingFang SC,Microsoft YaHei,sans-serif'>softmax1</text>
<text x='180' y='245' text-anchor='end' font-size='13' fill='#0f172a' font-family='Noto Sans CJK SC,PingFang SC,Microsoft YaHei,sans-serif'>w8-11</text>
<text x='180' y='259' text-anchor='end' font-size='10' fill='#64748b' font-family='Noto Sans CJK SC,PingFang SC,Microsoft YaHei,sans-serif'>correction</text>
<text x='180' y='285' text-anchor='end' font-size='13' fill='#0f172a' font-family='Noto Sans CJK SC,PingFang SC,Microsoft YaHei,sans-serif'>w13-14</text>
<text x='180' y='299' text-anchor='end' font-size='10' fill='#64748b' font-family='Noto Sans CJK SC,PingFang SC,Microsoft YaHei,sans-serif'>epilogue</text>
<rect x='190' y='40' width='58' height='268' rx='8' fill='#ffffff' stroke='#475569' stroke-width='1.5'/>
<text x='219.0' y='58' text-anchor='middle' font-size='12' font-weight='bold' fill='#0f172a' font-family='Noto Sans CJK SC,PingFang SC,Microsoft YaHei,sans-serif'>t0-2</text>
<rect x='194' y='72' width='50' height='26' rx='4' fill='#bfdbfe' stroke='#475569'/>
<text x='219.0' y='90' text-anchor='middle' font-size='10' fill='#0f172a' font-family='Noto Sans CJK SC,PingFang SC,Microsoft YaHei,sans-serif'>L0</text>
<rect x='194' y='112' width='50' height='26' rx='4' fill='#fecaca' stroke='#475569'/>
<text x='219.0' y='130' text-anchor='middle' font-size='10' fill='#b91c1c' font-family='Noto Sans CJK SC,PingFang SC,Microsoft YaHei,sans-serif'>睡</text>
<rect x='194' y='152' width='50' height='26' rx='4' fill='#fecaca' stroke='#475569'/>
<text x='219.0' y='170' text-anchor='middle' font-size='10' fill='#b91c1c' font-family='Noto Sans CJK SC,PingFang SC,Microsoft YaHei,sans-serif'>init</text>
<rect x='194' y='192' width='50' height='26' rx='4' fill='#fecaca' stroke='#475569'/>
<text x='219.0' y='210' text-anchor='middle' font-size='10' fill='#b91c1c' font-family='Noto Sans CJK SC,PingFang SC,Microsoft YaHei,sans-serif'>睡</text>
<rect x='194' y='232' width='50' height='26' rx='4' fill='#fecaca' stroke='#475569'/>
<text x='219.0' y='250' text-anchor='middle' font-size='10' fill='#b91c1c' font-family='Noto Sans CJK SC,PingFang SC,Microsoft YaHei,sans-serif'>睡</text>
<rect x='194' y='272' width='50' height='26' rx='4' fill='#fecaca' stroke='#475569'/>
<text x='219.0' y='290' text-anchor='middle' font-size='10' fill='#b91c1c' font-family='Noto Sans CJK SC,PingFang SC,Microsoft YaHei,sans-serif'>睡</text>
<rect x='354' y='40' width='58' height='268' rx='8' fill='#ffffff' stroke='#475569' stroke-width='1.5'/>
<text x='383.0' y='58' text-anchor='middle' font-size='12' font-weight='bold' fill='#0f172a' font-family='Noto Sans CJK SC,PingFang SC,Microsoft YaHei,sans-serif'>t2-4</text>
<rect x='358' y='72' width='50' height='26' rx='4' fill='#bfdbfe' stroke='#475569'/>
<text x='383.0' y='90' text-anchor='middle' font-size='10' fill='#0f172a' font-family='Noto Sans CJK SC,PingFang SC,Microsoft YaHei,sans-serif'>L1</text>
<rect x='358' y='112' width='50' height='26' rx='4' fill='#93c5fd' stroke='#475569'/>
<text x='383.0' y='130' text-anchor='middle' font-size='10' fill='#0f172a' font-family='Noto Sans CJK SC,PingFang SC,Microsoft YaHei,sans-serif'>S0</text>
<rect x='358' y='152' width='50' height='26' rx='4' fill='#fecaca' stroke='#475569'/>
<text x='383.0' y='170' text-anchor='middle' font-size='10' fill='#b91c1c' font-family='Noto Sans CJK SC,PingFang SC,Microsoft YaHei,sans-serif'>睡</text>
<rect x='358' y='192' width='50' height='26' rx='4' fill='#fecaca' stroke='#475569'/>
<text x='383.0' y='210' text-anchor='middle' font-size='10' fill='#b91c1c' font-family='Noto Sans CJK SC,PingFang SC,Microsoft YaHei,sans-serif'>睡</text>
<rect x='358' y='232' width='50' height='26' rx='4' fill='#fecaca' stroke='#475569'/>
<text x='383.0' y='250' text-anchor='middle' font-size='10' fill='#b91c1c' font-family='Noto Sans CJK SC,PingFang SC,Microsoft YaHei,sans-serif'>睡</text>
<rect x='358' y='272' width='50' height='26' rx='4' fill='#fecaca' stroke='#475569'/>
<text x='383.0' y='290' text-anchor='middle' font-size='10' fill='#b91c1c' font-family='Noto Sans CJK SC,PingFang SC,Microsoft YaHei,sans-serif'>睡</text>
<rect x='518' y='40' width='58' height='268' rx='8' fill='#ffffff' stroke='#475569' stroke-width='1.5'/>
<text x='547.0' y='58' text-anchor='middle' font-size='12' font-weight='bold' fill='#0f172a' font-family='Noto Sans CJK SC,PingFang SC,Microsoft YaHei,sans-serif'>t4-6</text>
<rect x='522' y='72' width='50' height='26' rx='4' fill='#bfdbfe' stroke='#475569'/>
<text x='547.0' y='90' text-anchor='middle' font-size='10' fill='#0f172a' font-family='Noto Sans CJK SC,PingFang SC,Microsoft YaHei,sans-serif'>L2</text>
<rect x='522' y='112' width='50' height='26' rx='4' fill='#fed7aa' stroke='#475569'/>
<text x='547.0' y='130' text-anchor='middle' font-size='10' fill='#0f172a' font-family='Noto Sans CJK SC,PingFang SC,Microsoft YaHei,sans-serif'>P0</text>
<rect x='522' y='152' width='50' height='26' rx='4' fill='#fde68a' stroke='#475569'/>
<text x='547.0' y='170' text-anchor='middle' font-size='10' fill='#0f172a' font-family='Noto Sans CJK SC,PingFang SC,Microsoft YaHei,sans-serif'>M0</text>
<rect x='522' y='192' width='50' height='26' rx='4' fill='#fecaca' stroke='#475569'/>
<text x='547.0' y='210' text-anchor='middle' font-size='10' fill='#b91c1c' font-family='Noto Sans CJK SC,PingFang SC,Microsoft YaHei,sans-serif'>备</text>
<rect x='522' y='232' width='50' height='26' rx='4' fill='#e9d5ff' stroke='#475569'/>
<text x='547.0' y='250' text-anchor='middle' font-size='10' fill='#0f172a' font-family='Noto Sans CJK SC,PingFang SC,Microsoft YaHei,sans-serif'>C0</text>
<rect x='522' y='272' width='50' height='26' rx='4' fill='#fecaca' stroke='#475569'/>
<text x='547.0' y='290' text-anchor='middle' font-size='10' fill='#b91c1c' font-family='Noto Sans CJK SC,PingFang SC,Microsoft YaHei,sans-serif'>睡</text>
<rect x='682' y='40' width='58' height='268' rx='8' fill='#ffffff' stroke='#475569' stroke-width='1.5'/>
<text x='711.0' y='58' text-anchor='middle' font-size='12' font-weight='bold' fill='#0f172a' font-family='Noto Sans CJK SC,PingFang SC,Microsoft YaHei,sans-serif'>t6-8</text>
<rect x='686' y='72' width='50' height='26' rx='4' fill='#fecaca' stroke='#475569'/>
<text x='711.0' y='90' text-anchor='middle' font-size='10' fill='#b91c1c' font-family='Noto Sans CJK SC,PingFang SC,Microsoft YaHei,sans-serif'>—</text>
<rect x='686' y='112' width='50' height='26' rx='4' fill='#93c5fd' stroke='#475569'/>
<text x='711.0' y='130' text-anchor='middle' font-size='10' fill='#0f172a' font-family='Noto Sans CJK SC,PingFang SC,Microsoft YaHei,sans-serif'>S1</text>
<rect x='686' y='152' width='50' height='26' rx='4' fill='#fecaca' stroke='#475569'/>
<text x='711.0' y='170' text-anchor='middle' font-size='10' fill='#b91c1c' font-family='Noto Sans CJK SC,PingFang SC,Microsoft YaHei,sans-serif'>等</text>
<rect x='686' y='192' width='50' height='26' rx='4' fill='#fecaca' stroke='#475569'/>
<text x='711.0' y='210' text-anchor='middle' font-size='10' fill='#b91c1c' font-family='Noto Sans CJK SC,PingFang SC,Microsoft YaHei,sans-serif'>备</text>
<rect x='686' y='232' width='50' height='26' rx='4' fill='#fecaca' stroke='#475569'/>
<text x='711.0' y='250' text-anchor='middle' font-size='10' fill='#b91c1c' font-family='Noto Sans CJK SC,PingFang SC,Microsoft YaHei,sans-serif'>等</text>
<rect x='686' y='272' width='50' height='26' rx='4' fill='#fdba74' stroke='#475569'/>
<text x='711.0' y='290' text-anchor='middle' font-size='10' fill='#0f172a' font-family='Noto Sans CJK SC,PingFang SC,Microsoft YaHei,sans-serif'>E0</text>
<rect x='846' y='40' width='58' height='268' rx='8' fill='#ffffff' stroke='#475569' stroke-width='1.5'/>
<text x='875.0' y='58' text-anchor='middle' font-size='12' font-weight='bold' fill='#0f172a' font-family='Noto Sans CJK SC,PingFang SC,Microsoft YaHei,sans-serif'>t8-10</text>
<rect x='850' y='72' width='50' height='26' rx='4' fill='#fecaca' stroke='#475569'/>
<text x='875.0' y='90' text-anchor='middle' font-size='10' fill='#b91c1c' font-family='Noto Sans CJK SC,PingFang SC,Microsoft YaHei,sans-serif'>—</text>
<rect x='850' y='112' width='50' height='26' rx='4' fill='#fed7aa' stroke='#475569'/>
<text x='875.0' y='130' text-anchor='middle' font-size='10' fill='#0f172a' font-family='Noto Sans CJK SC,PingFang SC,Microsoft YaHei,sans-serif'>P1</text>
<rect x='850' y='152' width='50' height='26' rx='4' fill='#fde68a' stroke='#475569'/>
<text x='875.0' y='170' text-anchor='middle' font-size='10' fill='#0f172a' font-family='Noto Sans CJK SC,PingFang SC,Microsoft YaHei,sans-serif'>M1</text>
<rect x='850' y='192' width='50' height='26' rx='4' fill='#fecaca' stroke='#475569'/>
<text x='875.0' y='210' text-anchor='middle' font-size='10' fill='#b91c1c' font-family='Noto Sans CJK SC,PingFang SC,Microsoft YaHei,sans-serif'>等</text>
<rect x='850' y='232' width='50' height='26' rx='4' fill='#fecaca' stroke='#475569'/>
<text x='875.0' y='250' text-anchor='middle' font-size='10' fill='#b91c1c' font-family='Noto Sans CJK SC,PingFang SC,Microsoft YaHei,sans-serif'>等</text>
<rect x='850' y='272' width='50' height='26' rx='4' fill='#fecaca' stroke='#475569'/>
<text x='875.0' y='290' text-anchor='middle' font-size='10' fill='#b91c1c' font-family='Noto Sans CJK SC,PingFang SC,Microsoft YaHei,sans-serif'>等</text>
<rect x='1010' y='40' width='58' height='268' rx='8' fill='#ffffff' stroke='#475569' stroke-width='1.5'/>
<text x='1039.0' y='58' text-anchor='middle' font-size='12' font-weight='bold' fill='#0f172a' font-family='Noto Sans CJK SC,PingFang SC,Microsoft YaHei,sans-serif'>t10-12</text>
<rect x='1014' y='72' width='50' height='26' rx='4' fill='#fecaca' stroke='#475569'/>
<text x='1039.0' y='90' text-anchor='middle' font-size='10' fill='#b91c1c' font-family='Noto Sans CJK SC,PingFang SC,Microsoft YaHei,sans-serif'>—</text>
<rect x='1014' y='112' width='50' height='26' rx='4' fill='#fecaca' stroke='#475569'/>
<text x='1039.0' y='130' text-anchor='middle' font-size='10' fill='#b91c1c' font-family='Noto Sans CJK SC,PingFang SC,Microsoft YaHei,sans-serif'>—</text>
<rect x='1014' y='152' width='50' height='26' rx='4' fill='#fecaca' stroke='#475569'/>
<text x='1039.0' y='170' text-anchor='middle' font-size='10' fill='#b91c1c' font-family='Noto Sans CJK SC,PingFang SC,Microsoft YaHei,sans-serif'>等</text>
<rect x='1014' y='192' width='50' height='26' rx='4' fill='#fecaca' stroke='#475569'/>
<text x='1039.0' y='210' text-anchor='middle' font-size='10' fill='#b91c1c' font-family='Noto Sans CJK SC,PingFang SC,Microsoft YaHei,sans-serif'>等</text>
<rect x='1014' y='232' width='50' height='26' rx='4' fill='#e9d5ff' stroke='#475569'/>
<text x='1039.0' y='250' text-anchor='middle' font-size='10' fill='#0f172a' font-family='Noto Sans CJK SC,PingFang SC,Microsoft YaHei,sans-serif'>C1</text>
<rect x='1014' y='272' width='50' height='26' rx='4' fill='#fdba74' stroke='#475569'/>
<text x='1039.0' y='290' text-anchor='middle' font-size='10' fill='#0f172a' font-family='Noto Sans CJK SC,PingFang SC,Microsoft YaHei,sans-serif'>E1</text>
<rect x='248' y='40' width='106' height='268' rx='8' fill='#f1f5f9' stroke='#94a3b8' stroke-width='1' stroke-dasharray='6 4'/>
<text x='301.0' y='58' text-anchor='middle' font-size='11' font-weight='bold' fill='#64748b' font-family='Noto Sans CJK SC,PingFang SC,Microsoft YaHei,sans-serif'>隙1：本拍后</text>
<rect x='253' y='74' width='96' height='70' rx='6' fill='#ffffff' stroke='#94a3b8'/>
<text x='301.0' y='94' text-anchor='middle' font-size='11' font-weight='bold' fill='#0f172a' font-family='Noto Sans CJK SC,PingFang SC,Microsoft YaHei,sans-serif'>槽</text>
<text x='301.0' y='114' text-anchor='middle' font-size='10' fill='#0f172a' font-family='Noto Sans CJK SC,PingFang SC,Microsoft YaHei,sans-serif'>s0:搬</text>
<text x='301.0' y='130' text-anchor='middle' font-size='10' fill='#64748b' font-family='Noto Sans CJK SC,PingFang SC,Microsoft YaHei,sans-serif'>s1:空</text>
<rect x='253' y='152' width='96' height='70' rx='6' fill='#ffffff' stroke='#94a3b8'/>
<text x='301.0' y='172' text-anchor='middle' font-size='11' font-weight='bold' fill='#0f172a' font-family='Noto Sans CJK SC,PingFang SC,Microsoft YaHei,sans-serif'>数</text>
<text x='301.0' y='192' text-anchor='middle' font-size='10' fill='#0f172a' font-family='Noto Sans CJK SC,PingFang SC,Microsoft YaHei,sans-serif'>load</text>
<text x='301.0' y='208' text-anchor='middle' font-size='10' fill='#64748b' font-family='Noto Sans CJK SC,PingFang SC,Microsoft YaHei,sans-serif'>0/1</text>
<rect x='253' y='230' width='96' height='70' rx='6' fill='#ffffff' stroke='#94a3b8'/>
<text x='301.0' y='250' text-anchor='middle' font-size='11' font-weight='bold' fill='#0f172a' font-family='Noto Sans CJK SC,PingFang SC,Microsoft YaHei,sans-serif'>调度</text>
<text x='301.0' y='270' text-anchor='middle' font-size='10' fill='#0f172a' font-family='Noto Sans CJK SC,PingFang SC,Microsoft YaHei,sans-serif'>点w15＋init</text>
<rect x='412' y='40' width='106' height='268' rx='8' fill='#f1f5f9' stroke='#94a3b8' stroke-width='1' stroke-dasharray='6 4'/>
<text x='465.0' y='58' text-anchor='middle' font-size='11' font-weight='bold' fill='#64748b' font-family='Noto Sans CJK SC,PingFang SC,Microsoft YaHei,sans-serif'>隙2：本拍后</text>
<rect x='417' y='74' width='96' height='70' rx='6' fill='#ffffff' stroke='#94a3b8'/>
<text x='465.0' y='94' text-anchor='middle' font-size='11' font-weight='bold' fill='#0f172a' font-family='Noto Sans CJK SC,PingFang SC,Microsoft YaHei,sans-serif'>槽</text>
<text x='465.0' y='114' text-anchor='middle' font-size='10' fill='#0f172a' font-family='Noto Sans CJK SC,PingFang SC,Microsoft YaHei,sans-serif'>s0:满</text>
<text x='465.0' y='130' text-anchor='middle' font-size='10' fill='#64748b' font-family='Noto Sans CJK SC,PingFang SC,Microsoft YaHei,sans-serif'>s1:搬</text>
<rect x='417' y='152' width='96' height='70' rx='6' fill='#ffffff' stroke='#94a3b8'/>
<text x='465.0' y='172' text-anchor='middle' font-size='11' font-weight='bold' fill='#0f172a' font-family='Noto Sans CJK SC,PingFang SC,Microsoft YaHei,sans-serif'>数</text>
<text x='465.0' y='192' text-anchor='middle' font-size='10' fill='#0f172a' font-family='Noto Sans CJK SC,PingFang SC,Microsoft YaHei,sans-serif'>S</text>
<text x='465.0' y='208' text-anchor='middle' font-size='10' fill='#64748b' font-family='Noto Sans CJK SC,PingFang SC,Microsoft YaHei,sans-serif'>0/1</text>
<rect x='417' y='230' width='96' height='70' rx='6' fill='#ffffff' stroke='#94a3b8'/>
<text x='465.0' y='250' text-anchor='middle' font-size='11' font-weight='bold' fill='#0f172a' font-family='Noto Sans CJK SC,PingFang SC,Microsoft YaHei,sans-serif'>调度</text>
<text x='465.0' y='270' text-anchor='middle' font-size='10' fill='#0f172a' font-family='Noto Sans CJK SC,PingFang SC,Microsoft YaHei,sans-serif'>点w12</text>
<rect x='576' y='40' width='106' height='268' rx='8' fill='#f1f5f9' stroke='#94a3b8' stroke-width='1' stroke-dasharray='6 4'/>
<text x='629.0' y='58' text-anchor='middle' font-size='11' font-weight='bold' fill='#64748b' font-family='Noto Sans CJK SC,PingFang SC,Microsoft YaHei,sans-serif'>隙3：本拍后</text>
<rect x='581' y='74' width='96' height='70' rx='6' fill='#ffffff' stroke='#94a3b8'/>
<text x='629.0' y='94' text-anchor='middle' font-size='11' font-weight='bold' fill='#0f172a' font-family='Noto Sans CJK SC,PingFang SC,Microsoft YaHei,sans-serif'>槽</text>
<text x='629.0' y='114' text-anchor='middle' font-size='10' fill='#0f172a' font-family='Noto Sans CJK SC,PingFang SC,Microsoft YaHei,sans-serif'>s0:吃</text>
<text x='629.0' y='130' text-anchor='middle' font-size='10' fill='#64748b' font-family='Noto Sans CJK SC,PingFang SC,Microsoft YaHei,sans-serif'>s1:满</text>
<rect x='581' y='152' width='96' height='70' rx='6' fill='#ffffff' stroke='#94a3b8'/>
<text x='629.0' y='172' text-anchor='middle' font-size='11' font-weight='bold' fill='#0f172a' font-family='Noto Sans CJK SC,PingFang SC,Microsoft YaHei,sans-serif'>数</text>
<text x='629.0' y='192' text-anchor='middle' font-size='10' fill='#0f172a' font-family='Noto Sans CJK SC,PingFang SC,Microsoft YaHei,sans-serif'>sc</text>
<text x='629.0' y='208' text-anchor='middle' font-size='10' fill='#64748b' font-family='Noto Sans CJK SC,PingFang SC,Microsoft YaHei,sans-serif'>0/128</text>
<rect x='581' y='230' width='96' height='70' rx='6' fill='#ffffff' stroke='#94a3b8'/>
<text x='629.0' y='250' text-anchor='middle' font-size='11' font-weight='bold' fill='#0f172a' font-family='Noto Sans CJK SC,PingFang SC,Microsoft YaHei,sans-serif'>调度</text>
<text x='629.0' y='270' text-anchor='middle' font-size='10' fill='#0f172a' font-family='Noto Sans CJK SC,PingFang SC,Microsoft YaHei,sans-serif'>点s0＋w12</text>
<rect x='740' y='40' width='106' height='268' rx='8' fill='#f1f5f9' stroke='#94a3b8' stroke-width='1' stroke-dasharray='6 4'/>
<text x='793.0' y='58' text-anchor='middle' font-size='11' font-weight='bold' fill='#64748b' font-family='Noto Sans CJK SC,PingFang SC,Microsoft YaHei,sans-serif'>隙4：本拍后</text>
<rect x='745' y='74' width='96' height='70' rx='6' fill='#ffffff' stroke='#94a3b8'/>
<text x='793.0' y='94' text-anchor='middle' font-size='11' font-weight='bold' fill='#0f172a' font-family='Noto Sans CJK SC,PingFang SC,Microsoft YaHei,sans-serif'>槽</text>
<text x='793.0' y='114' text-anchor='middle' font-size='10' fill='#0f172a' font-family='Noto Sans CJK SC,PingFang SC,Microsoft YaHei,sans-serif'>s0:吃</text>
<text x='793.0' y='130' text-anchor='middle' font-size='10' fill='#64748b' font-family='Noto Sans CJK SC,PingFang SC,Microsoft YaHei,sans-serif'>s1:搬</text>
<rect x='745' y='152' width='96' height='70' rx='6' fill='#ffffff' stroke='#94a3b8'/>
<text x='793.0' y='172' text-anchor='middle' font-size='11' font-weight='bold' fill='#0f172a' font-family='Noto Sans CJK SC,PingFang SC,Microsoft YaHei,sans-serif'>数</text>
<text x='793.0' y='192' text-anchor='middle' font-size='10' fill='#0f172a' font-family='Noto Sans CJK SC,PingFang SC,Microsoft YaHei,sans-serif'>ce</text>
<text x='793.0' y='208' text-anchor='middle' font-size='10' fill='#64748b' font-family='Noto Sans CJK SC,PingFang SC,Microsoft YaHei,sans-serif'>到</text>
<rect x='745' y='230' width='96' height='70' rx='6' fill='#ffffff' stroke='#94a3b8'/>
<text x='793.0' y='250' text-anchor='middle' font-size='11' font-weight='bold' fill='#0f172a' font-family='Noto Sans CJK SC,PingFang SC,Microsoft YaHei,sans-serif'>调度</text>
<text x='793.0' y='270' text-anchor='middle' font-size='10' fill='#0f172a' font-family='Noto Sans CJK SC,PingFang SC,Microsoft YaHei,sans-serif'>点w12＋w13</text>
<rect x='904' y='40' width='106' height='268' rx='8' fill='#f1f5f9' stroke='#94a3b8' stroke-width='1' stroke-dasharray='6 4'/>
<text x='957.0' y='58' text-anchor='middle' font-size='11' font-weight='bold' fill='#64748b' font-family='Noto Sans CJK SC,PingFang SC,Microsoft YaHei,sans-serif'>隙5：本拍后</text>
<rect x='909' y='74' width='96' height='70' rx='6' fill='#ffffff' stroke='#94a3b8'/>
<text x='957.0' y='94' text-anchor='middle' font-size='11' font-weight='bold' fill='#0f172a' font-family='Noto Sans CJK SC,PingFang SC,Microsoft YaHei,sans-serif'>槽</text>
<text x='957.0' y='114' text-anchor='middle' font-size='10' fill='#0f172a' font-family='Noto Sans CJK SC,PingFang SC,Microsoft YaHei,sans-serif'>s0:满</text>
<text x='957.0' y='130' text-anchor='middle' font-size='10' fill='#64748b' font-family='Noto Sans CJK SC,PingFang SC,Microsoft YaHei,sans-serif'>s1:吃</text>
<rect x='909' y='152' width='96' height='70' rx='6' fill='#ffffff' stroke='#94a3b8'/>
<text x='957.0' y='172' text-anchor='middle' font-size='11' font-weight='bold' fill='#0f172a' font-family='Noto Sans CJK SC,PingFang SC,Microsoft YaHei,sans-serif'>数</text>
<text x='957.0' y='192' text-anchor='middle' font-size='10' fill='#0f172a' font-family='Noto Sans CJK SC,PingFang SC,Microsoft YaHei,sans-serif'>S1</text>
<text x='957.0' y='208' text-anchor='middle' font-size='10' fill='#64748b' font-family='Noto Sans CJK SC,PingFang SC,Microsoft YaHei,sans-serif'>0/1</text>
<rect x='909' y='230' width='96' height='70' rx='6' fill='#ffffff' stroke='#94a3b8'/>
<text x='957.0' y='250' text-anchor='middle' font-size='11' font-weight='bold' fill='#0f172a' font-family='Noto Sans CJK SC,PingFang SC,Microsoft YaHei,sans-serif'>调度</text>
<text x='957.0' y='270' text-anchor='middle' font-size='10' fill='#0f172a' font-family='Noto Sans CJK SC,PingFang SC,Microsoft YaHei,sans-serif'>点s0＋w12</text>
<rect x='1068' y='40' width='106' height='268' rx='8' fill='#f1f5f9' stroke='#94a3b8' stroke-width='1' stroke-dasharray='6 4'/>
<text x='1121.0' y='58' text-anchor='middle' font-size='11' font-weight='bold' fill='#64748b' font-family='Noto Sans CJK SC,PingFang SC,Microsoft YaHei,sans-serif'>隙6：本拍后</text>
<rect x='1073' y='74' width='96' height='70' rx='6' fill='#ffffff' stroke='#94a3b8'/>
<text x='1121.0' y='94' text-anchor='middle' font-size='11' font-weight='bold' fill='#0f172a' font-family='Noto Sans CJK SC,PingFang SC,Microsoft YaHei,sans-serif'>槽</text>
<text x='1121.0' y='114' text-anchor='middle' font-size='10' fill='#0f172a' font-family='Noto Sans CJK SC,PingFang SC,Microsoft YaHei,sans-serif'>s0:吃</text>
<text x='1121.0' y='130' text-anchor='middle' font-size='10' fill='#64748b' font-family='Noto Sans CJK SC,PingFang SC,Microsoft YaHei,sans-serif'>s1:满</text>
<rect x='1073' y='152' width='96' height='70' rx='6' fill='#ffffff' stroke='#94a3b8'/>
<text x='1121.0' y='172' text-anchor='middle' font-size='11' font-weight='bold' fill='#0f172a' font-family='Noto Sans CJK SC,PingFang SC,Microsoft YaHei,sans-serif'>数</text>
<text x='1121.0' y='192' text-anchor='middle' font-size='10' fill='#0f172a' font-family='Noto Sans CJK SC,PingFang SC,Microsoft YaHei,sans-serif'>sc1</text>
<text x='1121.0' y='208' text-anchor='middle' font-size='10' fill='#64748b' font-family='Noto Sans CJK SC,PingFang SC,Microsoft YaHei,sans-serif'>0/128</text>
<rect x='1073' y='230' width='96' height='70' rx='6' fill='#ffffff' stroke='#94a3b8'/>
<text x='1121.0' y='250' text-anchor='middle' font-size='11' font-weight='bold' fill='#0f172a' font-family='Noto Sans CJK SC,PingFang SC,Microsoft YaHei,sans-serif'>调度</text>
<text x='1121.0' y='270' text-anchor='middle' font-size='10' fill='#0f172a' font-family='Noto Sans CJK SC,PingFang SC,Microsoft YaHei,sans-serif'>点s1＋w8</text>
<text x='30' y='628' font-size='13' fill='#0f172a' font-family='Noto Sans CJK SC,PingFang SC,Microsoft YaHei,sans-serif'>s0/s1 两拨执行顺序不同：s0 先算 M0（t4），s1 等 S1（t9）——同 8 warp，分两批醒。</text>
<text x='30' y='650' font-size='12' fill='#64748b' font-family='Noto Sans CJK SC,PingFang SC,Microsoft YaHei,sans-serif'>隙列＝状态快照（槽谁在吃/谁满、数到几/cap、点了谁）；白带＝干活。死锁只一种：cap 人数错。</text>
</svg>
```
