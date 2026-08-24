#!/usr/bin/env python3
"""
QC GATE — Price & Power channel
Rules R001-R012. Ek video publish hone se pehle ye chalega.
Score 8/10 se kam ya koi hard fail => PUBLISH BLOCKED.

Usage:
    python3 qc_gate.py video.mp4 --meta meta.json
    python3 qc_gate.py video.mp4                 # sirf technical checks

meta.json (W2/W3 se nikalna hai):
{
  "title": "Why Renting in Texas Costs More Than Florida",
  "title_metric": "rent",
  "scenes": [{"narration": "...", "on_screen": "...", "visual_type": "bar_pair",
              "eyebrow": "METRIC 1 OF 8", "start": 0.0, "end": 4.2}],
  "metrics": [{"name": "Two-bedroom rent", "annual_gap": 972, "first_shown": 6.0}],
  "captions": ["TEXAS RENT: $1,931", "FLORIDA: $1,850"]
}

Exit codes: 0 = PASS, 1 = FAIL (publish blocked), 2 = usage/probe error
"""
import subprocess, json, sys, re, os, argparse, tempfile, glob, collections

# ---------------------------------------------------------------- thresholds
T = {
    "runtime_max":        420,    # 7:00 — is se lambi video tab tak nahi jab tak APV 40%+ na ho
    "runtime_min":        240,    # 4:00
    "avg_shot_max":       6.0,    # seconds
    "longest_hold_max":   8.0,    # seconds
    "layout_share_max":   0.15,   # ek layout max 15% runtime
    "title_metric_max_t": 20.0,   # title ka metric pehle 20s me
    "hook_figure_max_t":  15.0,   # pehla number pehle 15s me
    "caption_max_chars":  32,
    "safe_zone_y":        880,    # is se neeche sirf subtitle
    "lufs_target":        -14.0,
    "lufs_tol":           1.5,
    "true_peak_max":      -1.0,
    "trivial_gap_usd":    150,    # is se chhota saalana gap = trivial
    "trivial_lockout_t":  120.0,  # pehle 2 min me trivial metric mana
    "min_score":          8.0,
}

# ---------------------------------------------------------------- helpers
def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True)

def probe_duration(path):
    r = sh(f'ffprobe -v error -show_entries format=duration -of csv=p=0 "{path}"')
    try:
        return float(r.stdout.strip())
    except ValueError:
        sys.exit(f"[!] ffprobe fail: {path}")

def scene_cuts(path, thresh=0.2):
    """Hard cut timestamps."""
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
        meta = f.name
    sh(f'ffmpeg -hide_banner -nostats -i "{path}" '
       f'-filter_complex "select=\'gt(scene,{thresh})\',metadata=print:file={meta}" '
       f'-an -f null - 2>/dev/null')
    cuts = []
    if os.path.exists(meta):
        cuts = [float(m) for m in re.findall(r"pts_time:([0-9.]+)", open(meta).read())]
        os.unlink(meta)
    return sorted(cuts)

def layout_signatures(path, every=2.0):
    """Frames ko 16x9 gray me girakar duplicate layouts count karta hai."""
    d = tempfile.mkdtemp()
    sh(f'ffmpeg -v error -i "{path}" -vf "fps=1/{every},crop=1920:700:0:100,scale=16:9" '
       f'-pix_fmt gray {d}/h_%05d.pgm -y')
    sigs = []
    for fp in sorted(glob.glob(f"{d}/h_*.pgm")):
        raw = open(fp, "rb").read().split(b"\n", 3)
        px = raw[3] if len(raw) > 3 else b""
        sigs.append(tuple(1 if b > 110 else 0 for b in px))
        os.unlink(fp)
    os.rmdir(d)
    return sigs

def audio_stats(path):
    v = sh(f'ffmpeg -hide_banner -nostats -i "{path}" -af volumedetect -f null /dev/null 2>&1').stdout
    out = {"max_db": None, "mean_db": None, "clipped": 0, "lufs": None, "tp": None}
    m = re.search(r"max_volume:\s*(-?[\d.]+) dB", v)
    if m: out["max_db"] = float(m.group(1))
    m = re.search(r"mean_volume:\s*(-?[\d.]+) dB", v)
    if m: out["mean_db"] = float(m.group(1))
    m = re.search(r"histogram_0db:\s*(\d+)", v)
    if m: out["clipped"] = int(m.group(1))
    l = sh(f'ffmpeg -hide_banner -nostats -i "{path}" -af loudnorm=I=-14:TP=-1:print_format=json '
           f'-f null /dev/null 2>&1').stdout
    m = re.search(r'"input_i"\s*:\s*"(-?[\d.]+)"', l)
    if m: out["lufs"] = float(m.group(1))
    m = re.search(r'"input_tp"\s*:\s*"(-?[\d.]+)"', l)
    if m: out["tp"] = float(m.group(1))
    return out

def has_music_bed(path, floor_db=-50):
    """Agar gaps -50dB tak girte hain to koi bed nahi hai."""
    o = sh(f'ffmpeg -hide_banner -nostats -i "{path}" '
           f'-af silencedetect=noise={floor_db}dB:d=0.3 -f null /dev/null 2>&1').stdout
    return o.count("silence_start") == 0

# ---------------------------------------------------------------- rules
class Report:
    def __init__(self):
        self.rows = []
    def add(self, rid, name, ok, detail, hard=True, skipped=False):
        self.rows.append(dict(id=rid, name=name, ok=ok, detail=detail,
                              hard=hard, skipped=skipped))
    @property
    def checked(self):  return [r for r in self.rows if not r["skipped"]]
    @property
    def failed(self):   return [r for r in self.checked if not r["ok"]]
    @property
    def hardfail(self): return [r for r in self.failed if r["hard"]]
    def score(self):
        c = self.checked
        if not c: return 0.0
        return round(10.0 * sum(1 for r in c if r["ok"]) / len(c), 1)


def run(video, meta):
    rep = Report()
    dur = probe_duration(video)
    cuts = scene_cuts(video)
    bounds = [0.0] + cuts + [dur]
    holds = [bounds[i+1] - bounds[i] for i in range(len(bounds)-1)]
    avg_shot = dur / max(len(cuts) + 1, 1)
    longest = max(holds) if holds else dur
    sigs = layout_signatures(video)
    counts = collections.Counter(sigs)
    top_share = counts.most_common(1)[0][1] / len(sigs) if sigs else 1.0
    aud = audio_stats(video)
    bed = has_music_bed(video)

    scenes   = meta.get("scenes", [])
    metrics  = meta.get("metrics", [])
    captions = meta.get("captions", [])
    title    = meta.get("title", "")
    tmetric  = (meta.get("title_metric") or "").lower().strip()

    # --- R001 PROMISE -------------------------------------------------
    if tmetric and metrics:
        hit = [m for m in metrics if tmetric in m.get("name", "").lower()]
        if hit:
            t0 = min(m.get("first_shown", 1e9) for m in hit)
            rep.add("R001", "Title ka metric pehle 20s me",
                    t0 <= T["title_metric_max_t"],
                    f"'{tmetric}' pehli dafa {t0:.1f}s par (limit {T['title_metric_max_t']:.0f}s)")
        else:
            rep.add("R001", "Title ka metric pehle 20s me", False,
                    f"'{tmetric}' metrics list me hai hi nahi — title kuch aur waada kar raha hai")
    else:
        rep.add("R001", "Title ka metric pehle 20s me", True,
                "meta me title_metric/metrics nahi — skip", skipped=True)

    # --- R002 COMPARISON ----------------------------------------------
    if scenes:
        types = [s.get("visual_type", "") for s in scenes]
        n_bar = sum(1 for t in types if t == "bar_pair")
        need  = max(len(metrics), 1)
        rep.add("R002", "Har metric me bar_pair (dono values ek frame me)",
                n_bar >= need,
                f"{n_bar} bar_pair shots vs {need} metrics")
        bad = [t for t in types if t == "flag_vs"]
        rep.add("R002b", "flag_vs data card ke taur par use nahi ho raha",
                len(bad) <= 1, f"{len(bad)} flag_vs shots (sirf 1 title card allowed)")
    else:
        rep.add("R002", "bar_pair per metric", True, "meta.scenes nahi — skip", skipped=True)
        rep.add("R002b", "flag_vs misuse", True, "skip", skipped=True)

    # --- R003 PACE ----------------------------------------------------
    rep.add("R003a", f"Avg shot < {T['avg_shot_max']}s",
            avg_shot < T["avg_shot_max"],
            f"{avg_shot:.1f}s avg ({len(cuts)+1} shots / {dur:.0f}s)")
    rep.add("R003b", f"Longest hold < {T['longest_hold_max']}s",
            longest < T["longest_hold_max"], f"{longest:.1f}s sab se lamba static frame")
    rep.add("R003c", f"Koi layout > {T['layout_share_max']*100:.0f}% nahi",
            top_share <= T["layout_share_max"],
            f"sab se zyada dohraya layout = {top_share*100:.0f}% runtime "
            f"({len(counts)} distinct layouts)")

    # --- R004 HOOK ----------------------------------------------------
    if scenes:
        op = scenes[0]
        blob = f"{op.get('narration','')} {op.get('on_screen','')}"
        rep.add("R004a", "Scene 1 me hard figure",
                bool(re.search(r"\d", blob)), f"scene 1: {blob[:70]!r}")
        rep.add("R004b", "Scene 1 verdict nahi deta",
                not re.search(r"\b(overall|winner|wins|beats|verdict|cheaper overall)\b",
                              op.get("narration", ""), re.I),
                "verdict leak check")
        first_num = next((s.get("start", 0.0) for s in scenes
                          if re.search(r"\d", f"{s.get('narration','')}{s.get('on_screen','')}")),
                         1e9)
        rep.add("R004c", f"Pehla number < {T['hook_figure_max_t']:.0f}s",
                first_num <= T["hook_figure_max_t"], f"pehla figure {first_num:.1f}s par")
    else:
        for k in ("R004a", "R004b", "R004c"):
            rep.add(k, "Hook check", True, "meta.scenes nahi — skip", skipped=True)

    # --- R005 CAPTIONS -------------------------------------------------
    if captions:
        longc = [c for c in captions if len(c) > T["caption_max_chars"]]
        rep.add("R005a", f"Caption <= {T['caption_max_chars']} chars",
                not longc, f"{len(longc)} cards limit se lambe")
        split = []
        for i in range(len(captions) - 1):
            if re.search(r"[\d,.]$", captions[i].strip()) and \
               re.match(r"^[\d,]", captions[i+1].strip()):
                split.append(i)
        rep.add("R005b", "Koi number do cards me tuta nahi",
                not split, f"{len(split)} tute hue numbers"
                           + (f" — misal card #{split[0]}" if split else ""))
    else:
        rep.add("R005a", "Caption length", True, "meta.captions nahi — skip", skipped=True)
        rep.add("R005b", "Number atomicity", True, "skip", skipped=True)

    # --- R006 LABELS ---------------------------------------------------
    if scenes:
        bad = []
        for s in scenes:
            eb = set(re.findall(r"[a-z]+", s.get("eyebrow", "").lower()))
            hd = set(re.findall(r"[a-z]+", s.get("on_screen", "").lower()))
            if eb and eb & hd:
                bad.append(s.get("eyebrow", ""))
        rep.add("R006", "Eyebrow headline se lafz nahi churata",
                not bad, f"{len(bad)} cards overlap"
                         + (f" — misal {bad[0]!r}" if bad else ""), hard=False)
    else:
        rep.add("R006", "Eyebrow overlap", True, "skip", skipped=True)

    # --- R008 AUDIO ----------------------------------------------------
    rep.add("R008a", "Zero clipped samples",
            aud["clipped"] == 0, f"{aud['clipped']} samples 0 dBFS par")
    if aud["lufs"] is not None:
        ok = abs(aud["lufs"] - T["lufs_target"]) <= T["lufs_tol"]
        rep.add("R008b", f"Loudness {T['lufs_target']} LUFS (±{T['lufs_tol']})",
                ok, f"{aud['lufs']:.1f} LUFS")
    else:
        rep.add("R008b", "Loudness", True, "measure fail — skip", skipped=True)
    if aud["tp"] is not None:
        rep.add("R008c", f"True peak <= {T['true_peak_max']} dBTP",
                aud["tp"] <= T["true_peak_max"], f"{aud['tp']:.1f} dBTP")
    else:
        rep.add("R008c", "True peak", True, "measure fail — skip", skipped=True)
    rep.add("R008d", "Music bed maujood", bed,
            "bed detected" if bed else "gaps -50dB tak girte hain = koi bed nahi",
            hard=False)

    # --- R009 RHYTHM ---------------------------------------------------
    if len(metrics) >= 3:
        lens = [m.get("duration", 0) for m in metrics if m.get("duration")]
        if len(lens) >= 3:
            ratio = max(lens) / max(min(lens), 0.01)
            rep.add("R009", "Section lengths me 2.5x farq",
                    ratio >= 2.5, f"longest/shortest = {ratio:.1f}x", hard=False)
        else:
            rep.add("R009", "Rhythm", True, "durations nahi — skip", skipped=True)
    else:
        rep.add("R009", "Rhythm", True, "skip", skipped=True)

    # --- R011 IMAGERY --------------------------------------------------
    if scenes:
        photos = sum(1 for s in scenes
                     if str(s.get("visual_type", "")).startswith("photo"))
        need = max(len(metrics), 1)
        rep.add("R011", "Har metric me kam az kam 1 photo",
                photos >= need, f"{photos} photo shots vs {need} metrics")
    else:
        rep.add("R011", "Imagery", True, "skip", skipped=True)

    # --- R012 DATA WEIGHT ----------------------------------------------
    if metrics:
        early_trivial = [m for m in metrics
                         if m.get("first_shown", 1e9) < T["trivial_lockout_t"]
                         and 0 < m.get("annual_gap", 1e9) < T["trivial_gap_usd"]]
        rep.add("R012", f"Pehle 2 min me koi <${T['trivial_gap_usd']}/saal metric nahi",
                not early_trivial,
                "; ".join(f"{m['name']} (${m['annual_gap']}/yr @ {m['first_shown']:.0f}s)"
                          for m in early_trivial) or "clear")
    else:
        rep.add("R012", "Data weight", True, "skip", skipped=True)

    # --- runtime -------------------------------------------------------
    rep.add("LEN", f"Runtime {T['runtime_min']/60:.0f}-{T['runtime_max']/60:.0f} min",
            T["runtime_min"] <= dur <= T["runtime_max"],
            f"{int(dur//60)}:{int(dur%60):02d}", hard=False)

    return rep, dict(duration=dur, cuts=len(cuts), avg_shot=avg_shot,
                     longest_hold=longest, layouts=len(counts),
                     top_layout_share=top_share, audio=aud, music_bed=bed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--meta", help="scenes/metrics/captions wali JSON")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    a = ap.parse_args()

    if not os.path.exists(a.video):
        sys.exit(2)
    meta = json.load(open(a.meta)) if a.meta and os.path.exists(a.meta) else {}

    rep, facts = run(a.video, meta)
    score = rep.score()
    blocked = bool(rep.hardfail) or score < T["min_score"]

    if a.json:
        print(json.dumps(dict(score=score, blocked=blocked, facts=facts,
                              rows=rep.rows), indent=2, default=str))
        sys.exit(1 if blocked else 0)

    W = 78
    print("=" * W)
    print(f"  QC GATE   {os.path.basename(a.video)}")
    print("=" * W)
    print(f"  runtime {int(facts['duration']//60)}:{int(facts['duration']%60):02d}   "
          f"cuts {facts['cuts']}   avg shot {facts['avg_shot']:.1f}s   "
          f"longest {facts['longest_hold']:.1f}s")
    print(f"  layouts {facts['layouts']}   top layout {facts['top_layout_share']*100:.0f}%   "
          f"clipped {facts['audio']['clipped']}   bed {'yes' if facts['music_bed'] else 'NO'}")
    print("-" * W)
    for r in rep.rows:
        if r["skipped"]:
            mark, tag = "  ", "skip"
        elif r["ok"]:
            mark, tag = "OK", "    "
        else:
            mark, tag = ("XX", "HARD") if r["hard"] else ("!!", "soft")
        print(f"  [{mark}] {tag} {r['id']:<7} {r['name']}")
        print(f"            {r['detail']}")
    print("-" * W)
    print(f"  SCORE {score}/10   (checked {len(rep.checked)}, "
          f"failed {len(rep.failed)}, hard fails {len(rep.hardfail)})")
    print(f"  >>> {'PUBLISH BLOCKED' if blocked else 'CLEARED TO PUBLISH'} <<<")
    print("=" * W)
    sys.exit(1 if blocked else 0)


if __name__ == "__main__":
    main()
