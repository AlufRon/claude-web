import os, re, json, subprocess, sys, shutil
from pathlib import Path
from rich import print
import numpy as np
import soundfile as sf
import librosa

# ---------- CONFIG ----------
YOUTUBE_URL = "https://www.youtube.com/watch?v=ugvHCXCOmm4"  # Lex #452 (Dario Amodei)
OUT_DIR = Path("out_podcast")
AUDIO_BASENAME = "podcast"
WHISPERX_DIR = OUT_DIR / "whisperx"
FORCE_TWO_SPEAKERS = True  # set False if you want WhisperX to estimate speaker count
# ----------------------------

OUT_DIR.mkdir(parents=True, exist_ok=True)

def run(cmd):
    print(f"[cyan]$ {' '.join(cmd)}[/cyan]")
    subprocess.run(cmd, check=True)

# 1) Download audio (m4a/opus) with yt-dlp (no ffmpeg needed)
def download_audio():
    out_template = str(OUT_DIR / f"{AUDIO_BASENAME}.%(ext)s")
    cmd = [
        sys.executable, "-m", "yt_dlp",
        "-f", "bestaudio/best",
        "-o", out_template,
        "--no-progress",
        YOUTUBE_URL,
    ]
    run(cmd)
    # pick the downloaded file
    files = list(OUT_DIR.glob(f"{AUDIO_BASENAME}.*"))
    if not files:
        raise RuntimeError("No audio file downloaded.")
    # prefer m4a if present
    files_sorted = sorted(files, key=lambda p: (p.suffix != ".m4a", str(p)))
    return files_sorted[0]

# 2) Diarize with WhisperX (creates RTTM + segments)
def whisperx_diarize(audio_path: Path):
    WHISPERX_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "whisperx", str(audio_path),
        "--diarize",
        "--output_dir", str(WHISPERX_DIR),
        "--language", "en",
        "--compute_type", "int8",  # CPU-compatible compute type
    ]
    if FORCE_TWO_SPEAKERS:
        cmd += ["--min_speakers", "2", "--max_speakers", "2"]
    run(cmd)
    # WhisperX RTTM lives as diarization.rttm or in alignment folder depending on version
    rttm = WHISPERX_DIR / "diarization.rttm"
    if not rttm.exists():
        # try find it
        candidates = list(WHISPERX_DIR.glob("**/*.rttm"))
        if not candidates:
            raise RuntimeError("Could not find diarization.rttm from WhisperX.")
        rttm = candidates[0]
    return rttm

# 3) Parse RTTM into speaker segments
def parse_rttm(rttm_path: Path):
    # RTTM format: SPEAKER <file> 1 <start> <dur> <ortho> <stype> <name> <conf>
    segs = {}
    for line in rttm_path.read_text().splitlines():
        if not line.strip() or line.startswith("#"): 
            continue
        parts = line.split()
        if parts[0] != "SPEAKER":
            continue
        start = float(parts[3]); dur = float(parts[4])
        spk = parts[7] if len(parts) > 7 else parts[1]
        segs.setdefault(spk, []).append((start, start + dur))
    # sort segments by time
    for k in segs:
        segs[k].sort()
    # If WhisperX names speakers as SPEAKER_00/01/etc, order them by first occurrence
    ordered = sorted(segs.items(), key=lambda kv: kv[1][0][0])
    return ordered  # list of (speaker_id, [(start,end), ...])

# 4) Load audio, slice per speaker, write mono WAVs and a stereo mix
def export_tracks(audio_path: Path, speaker_segs):
    y, sr = librosa.load(str(audio_path), sr=None, mono=True)  # load as mono; diarization is channel-agnostic
    print(f"[green]Loaded audio:[/green] {audio_path.name}, sr={sr}, duration={len(y)/sr:.1f}s")
    # Build continuous tracks per speaker by concatenating segments
    wav_paths = []
    mono_tracks = []
    for idx, (spk, segs) in enumerate(speaker_segs):
        pieces = []
        for (s, e) in segs:
            s_i = max(0, int(s * sr)); e_i = min(len(y), int(e * sr))
            if e_i > s_i:
                pieces.append(y[s_i:e_i])
        if not pieces:
            pieces = [np.zeros(1, dtype=np.float32)]
        track = np.concatenate(pieces)
        mono_tracks.append(track)
        out_wav = OUT_DIR / f"speaker{idx+1}_{spk}.wav"
        sf.write(out_wav, track, sr)
        wav_paths.append(out_wav)
        print(f"[yellow]Wrote[/yellow] {out_wav}  ({len(track)/sr:.1f}s)")

    # Pad shorter track so we can make a stereo side-by-side file
    max_len = max(len(t) for t in mono_tracks)
    padded = [np.pad(t, (0, max_len-len(t))) for t in mono_tracks]
    if len(padded) == 1:
        # duplicate to stereo if only one speaker found
        stereo = np.stack([padded[0], padded[0]], axis=1)
    else:
        stereo = np.stack([padded[0], padded[1]], axis=1)
    stereo_path = OUT_DIR / "two_channel.wav"
    sf.write(stereo_path, stereo, sr)
    print(f"[magenta]Wrote stereo mix[/magenta] {stereo_path} (L=speaker1, R=speaker2)")
    return wav_paths, stereo_path

def main():
    audio_file = download_audio()
    rttm = whisperx_diarize(audio_file)
    speakers = parse_rttm(rttm)
    if len(speakers) == 0:
        raise RuntimeError("No speakers found in RTTM.")
    if len(speakers) > 2:
        print(f"[red]Found {len(speakers)} speakers; using first two by start-time for export.[/red]")
        speakers = speakers[:2]
    wavs, stereo = export_tracks(audio_file, speakers)
    print("\n[bold green]Done![/bold green]")
    print(f"Speaker WAVs: {[str(p) for p in wavs]}")
    print(f"Stereo: {stereo}")

if __name__ == "__main__":
    main()
