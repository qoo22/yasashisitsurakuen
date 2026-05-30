#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
本物の 8bit(NES風) チップチューン・エンジン。
矩形波(duty可変) / 三角波(ベース) / ノイズ(ドラム) を合成し、
オリジナル作曲を WAV -> MP3 で書き出す。AIより クリアで 長尺・ループ可能。
"""
import numpy as np, wave, subprocess, sys, os

SR = 44100

def f(n):  # midi -> freq
    return 440.0 * 2.0 ** ((n - 69) / 12.0)

def env_adsr(n, a=0.005, d=0.04, s=0.6, r=0.05):
    a_n, d_n, r_n = int(a*SR), int(d*SR), int(r*SR)
    s_n = max(1, n - a_n - d_n - r_n)
    parts = []
    parts.append(np.linspace(0, 1, max(1, a_n), endpoint=False))
    parts.append(np.linspace(1, s, max(1, d_n), endpoint=False))
    parts.append(np.full(s_n, s))
    parts.append(np.linspace(s, 0, max(1, r_n)))
    e = np.concatenate(parts)
    if len(e) < n: e = np.concatenate([e, np.zeros(n-len(e))])
    return e[:n]

def square(freq, n, duty=0.5):
    t = np.arange(n)/SR
    ph = (t*freq) % 1.0
    return np.where(ph < duty, 1.0, -1.0)

def triangle(freq, n):
    t = np.arange(n)/SR
    ph = (t*freq) % 1.0
    return 2*np.abs(2*ph-1)-1

def render_note(buf, start, dur_s, midi, kind='sq', duty=0.5, vol=0.25,
                a=0.005, d=0.04, s=0.65, r=0.06):
    if midi is None: return
    n = int(dur_s*SR)
    if n <= 0: return
    if kind == 'sq':   w = square(f(midi), n, duty)
    elif kind == 'tri':w = triangle(f(midi), n)
    else:              w = square(f(midi), n, duty)
    e = env_adsr(n, a, d, s, r)
    seg = w*e*vol
    i0 = int(start*SR)
    i1 = min(len(buf), i0+n)
    buf[i0:i1] += seg[:i1-i0]

def render_drum(buf, start, kind='kick', vol=0.5):
    if kind == 'kick':
        n = int(0.12*SR); t = np.arange(n)/SR
        fr = np.linspace(150, 50, n)
        ph = 2*np.pi*np.cumsum(fr)/SR
        w = np.sin(ph) * np.exp(-t*22)
    elif kind == 'snare':
        n = int(0.13*SR); t = np.arange(n)/SR
        w = (np.random.uniform(-1,1,n)*0.9 + np.sin(2*np.pi*180*t)*0.3) * np.exp(-t*26)
    else:  # hat
        n = int(0.04*SR); t = np.arange(n)/SR
        w = np.random.uniform(-1,1,n) * np.exp(-t*90)
    seg = w*vol
    i0 = int(start*SR); i1 = min(len(buf), i0+n)
    buf[i0:i1] += seg[:i1-i0]

class Song:
    def __init__(self, bpm, beats_total):
        self.spb = 60.0/bpm
        self.n = int(beats_total*self.spb*SR) + SR
        self.buf = np.zeros(self.n, dtype=np.float64)
    def note(self, beat, dur_beats, midi, **kw):
        render_note(self.buf, beat*self.spb, dur_beats*self.spb, midi, **kw)
    def drum(self, beat, kind, vol=0.5):
        render_drum(self.buf, beat*self.spb, kind, vol)
    def save(self, name):
        b = self.buf
        # ソフトクリップ + 正規化
        b = np.tanh(b*1.1)
        peak = np.max(np.abs(b)) or 1.0
        b = b/peak*0.92
        pcm = (b*32767).astype('<i2')
        wav = '/tmp/_ct_%s.wav' % os.path.basename(name)   # 一時WAVは /tmp(削除可)へ
        with wave.open(wav,'w') as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
            w.writeframes(pcm.tobytes())
        mp3 = name+'.mp3'
        subprocess.run(['ffmpeg','-y','-loglevel','error','-i',wav,'-ac','1','-ar','44100','-b:a','128k',mp3])
        try: os.remove(wav)
        except Exception: pass
        dur = len(b)/SR
        print('OK %s.mp3  %.1fs' % (name, dur))

# ===== 音楽理論ヘルパー =====
def chord_tones(root, kind):
    if kind=='min':  return [root, root+3, root+7]
    if kind=='maj':  return [root, root+4, root+7]
    if kind=='dim':  return [root, root+3, root+6]
    return [root, root+4, root+7]

def build(s, bar0, prog, bars, lead, *, drums=True, hat=True, arp=True, bass=True, pad=False,
          lead_duty=0.5, arp_duty=0.25, lead_vol=0.30, arp_vol=0.15, bass_vol=0.40,
          bass_pat='drive', arp_oct=12, octave=0, drum_vol=1.0,
          lead_kind='sq', arp_kind='sq', lead_r=0.05, arp_pat='up', swing=0.0):
    """進行 prog をもとに 1セクション(bars小節)を 合成する 汎用ビルダ。
       lead_kind/arp_kind='tri' で 三角波(丸く やわらかい オルゴール風)に。
       arp_pat='bell' で 高音の ベル(2分散)/'soft' で 8分の ゆったり分散。
       lead_r で リードの 余韻(release)を のばせる(癒し系は 長め)。"""
    for i in range(bars):
        bar = bar0 + i
        root, kind = prog[i % len(prog)]
        tones = chord_tones(root, kind)
        base = bar*4
        if bass:
            if bass_pat=='drive':
                pat=[root-12, root, root-12, root-5, root-12, root, root-12, root-5]
                for j,bn in enumerate(pat):
                    s.note(base+j*0.5,0.48,bn,kind='tri',vol=bass_vol,a=0.002,d=0.02,s=0.8,r=0.03)
            elif bass_pat=='walk':
                for j in range(4):
                    bn=(root-12) if j%2==0 else (root-5)
                    s.note(base+j,0.9,bn,kind='tri',vol=bass_vol,a=0.004,d=0.03,s=0.8,r=0.05)
            elif bass_pat=='waltz':  # 1拍ベース + 3拍に和音で ゆれる(3拍子っぽい優しさ)
                s.note(base,0.9,root-12,kind='tri',vol=bass_vol,a=0.006,d=0.04,s=0.8,r=0.08)
                s.note(base+1.5,0.6,root-5,kind='tri',vol=bass_vol*0.8,a=0.006,d=0.04,s=0.8,r=0.06)
                s.note(base+2.5,0.6,root-5,kind='tri',vol=bass_vol*0.8,a=0.006,d=0.04,s=0.8,r=0.06)
            elif bass_pat=='pulse':  # 4分の やわらかい 脈打ち
                for j in range(4):
                    s.note(base+j,0.7,root-12,kind='tri',vol=bass_vol,a=0.006,d=0.05,s=0.78,r=0.08)
            else:  # 'soft' 全音符
                s.note(base,3.9,root-12,kind='tri',vol=bass_vol,a=0.02,d=0.1,s=0.85,r=0.3)
        if pad:
            for tn in tones:
                s.note(base,3.9,tn+12,kind='sq',duty=0.5,vol=0.09,a=0.05,d=0.1,s=0.85,r=0.4)
        if arp:
            seq=tones+[tones[0]+12]
            if arp_pat=='bell':   # 高音で きらきら(8分・余韻長め): オルゴール/鈴
                for k in range(8):
                    s.note(base+k*0.5,0.45,seq[k%len(seq)]+arp_oct,kind=arp_kind,duty=arp_duty,
                           vol=arp_vol,a=0.002,d=0.03,s=0.4,r=0.18)
            elif arp_pat=='soft': # 8分の ゆったり分散
                for k in range(8):
                    s.note(base+k*0.5,0.4,seq[k%len(seq)]+arp_oct,kind=arp_kind,duty=arp_duty,
                           vol=arp_vol,a=0.002,d=0.02,s=0.5,r=0.08)
            else:                 # 'up' 16分の上昇(既存)
                for k in range(16):
                    s.note(base+k*0.25,0.22,seq[k%len(seq)]+arp_oct,kind=arp_kind,duty=arp_duty,
                           vol=arp_vol,a=0.001,d=0.01,s=0.5,r=0.02)
        if drums:
            s.drum(base+0,'kick',0.55*drum_vol); s.drum(base+2,'kick',0.5*drum_vol)
            s.drum(base+1,'snare',0.5*drum_vol); s.drum(base+3,'snare',0.5*drum_vol)
            if hat:
                for h in range(8): s.drum(base+h*0.5,'hat',0.16*drum_vol)
        for (off,dur,mid) in lead[i % len(lead)]:
            if mid is not None:
                s.note(base+off+(swing if (off%1)!=0 else 0),dur,mid+octave,kind=lead_kind,duty=lead_duty,
                       vol=lead_vol,a=0.004,d=0.03,s=0.72,r=lead_r)

# =========================================================
#  灰冠の王 アルゴール ── 本戦曲（悲しい王の決戦）
#  A minor / 128 BPM / 駆動・勇壮・悲哀 / オルガン風square + 速いアルペジオ
# =========================================================
def make_argor_battle(path):
    BPM=128
    # 進行: Am - F - C - E (i - VI - III - V)  ×4サイクル(=16小節) を 2回(A/B)
    prog = [(57,'min'),(53,'maj'),(48,'maj'),(52,'maj')]  # A,F,C,E
    bars = 32
    s = Song(BPM, bars*4 + 1)

    def section(bar0, lead, octave=0, busy=False):
        for i in range(16):
            bar = bar0 + i
            root, kind = prog[i % 4]
            tones = chord_tones(root, kind)
            base = bar*4
            # --- ベース(三角波): 8分の駆動 root/oct ---
            bass_pat = [root-12, root, root-12, root-5, root-12, root, root-12, root-5]
            for j,bn in enumerate(bass_pat):
                s.note(base + j*0.5, 0.48, bn, kind='tri', vol=0.42, a=0.002, d=0.02, s=0.8, r=0.03)
            # --- アルペジオ(square 25%): 16分で上昇 ---
            arp = []
            seq = tones + [tones[0]+12]      # 4音
            for k in range(16):
                arp.append(seq[k % len(seq)] + 12 + octave)
            for k,an in enumerate(arp):
                s.note(base + k*0.25, 0.22, an, kind='sq', duty=0.25, vol=0.16,
                       a=0.001, d=0.01, s=0.5, r=0.02)
            # --- ドラム ---
            s.drum(base+0,'kick',0.55); s.drum(base+2,'kick',0.5)
            s.drum(base+1,'snare',0.5); s.drum(base+3,'snare',0.5)
            for h in range(8):
                s.drum(base+h*0.5,'hat',0.18)
            # --- リード(square 50%, オルガン風): 手書きモチーフ ---
            for (off,dur,mid) in lead[i % len(lead)]:
                if mid is not None:
                    s.note(base+off, dur, mid+octave, kind='sq', duty=0.5, vol=0.30,
                           a=0.004, d=0.03, s=0.72, r=0.05)

    # 8小節モチーフ（A minor。勇ましいが もの悲しい）→ 16小節ぶんに展開
    L = [
        [(0,2,69),(2,1,72),(3,1,71)],     # Am: A . C B
        [(0,2,69),(2,2,65)],              # F:  A . F
        [(0,2,67),(2,1,64),(3,1,67)],     # C:  G . E G
        [(0,2,64),(2,2,68)],              # E:  E . G#
        [(0,2,69),(2,1,76),(3,1,74)],     # Am: A . E5 D5
        [(0,2,72),(2,2,69)],              # F:  C5 . A
        [(0,2,71),(2,2,67)],              # C:  B . G
        [(0,4,64)],                       # E:  E (のばし)
        # 9-16小節（変奏: 少し上に）
        [(0,1,76),(1,1,74),(2,1,72),(3,1,71)],
        [(0,2,69),(2,1,72),(3,1,69)],
        [(0,1,79),(1,1,76),(2,2,72)],
        [(0,2,68),(2,2,64)],
        [(0,1,76),(1,1,77),(2,1,76),(3,1,72)],
        [(0,2,69),(2,2,65)],
        [(0,1,71),(1,1,72),(2,1,74),(3,1,76)],
        [(0,4,69)],
    ]
    section(0,  L, octave=0)    # A セクション(16小節)
    section(16, L, octave=0, busy=True)  # B セクション(くり返し=厚み)
    s.save(path)

# =========================================================
#  アルゴール 導入曲（Bergentrückung的: 静かな儀式・王の覚悟）
#  A minor / 66 BPM / オルガン風 持続音 + 三角パッド / ドラムなし
# =========================================================
def make_argor_intro(path):
    BPM=66
    s = Song(BPM, 8*4 + 2)
    prog = [(57,'min'),(53,'maj'),(48,'maj'),(52,'maj'),
            (57,'min'),(53,'maj'),(52,'maj'),(45,'min')]
    # オルガン風: 和音を 持続（square 50% を 3声 重ねる）+ 低い三角パッド
    lead_motif = [69, 72, 71, 69, 64, 67, 68, 69]  # ゆっくりした 王のテーマ
    for bar,(root,kind) in enumerate(prog):
        base = bar*4
        tones = chord_tones(root, kind)
        # 和音(持続) 3声
        for tn in tones:
            s.note(base, 3.9, tn+12, kind='sq', duty=0.5, vol=0.12, a=0.05, d=0.1, s=0.85, r=0.4)
        # 低い三角パッド(根音)
        s.note(base, 3.9, root-12, kind='tri', vol=0.32, a=0.06, d=0.1, s=0.85, r=0.4)
        # リード(ゆっくり)
        s.note(base, 2.0, lead_motif[bar], kind='sq', duty=0.5, vol=0.26, a=0.02, d=0.08, s=0.8, r=0.3)
        s.note(base+2, 1.8, lead_motif[bar]+ (3 if kind=='min' else 4), kind='sq', duty=0.5,
               vol=0.20, a=0.02, d=0.08, s=0.8, r=0.3)
    s.save(path)

# ===== タイトル: 神秘的な オルゴール (C major / ゆったり) =====
def make_title(path):
    s=Song(92, 8*4+2)
    prog=[(48,'maj'),(57,'min'),(53,'maj'),(55,'maj')]
    lead=[
        [(0,1,72),(1,1,76),(2,2,79)],[(0,1,74),(1,1,72),(2,2,69)],
        [(0,1,77),(1,1,76),(2,2,72)],[(0,2,74),(2,2,71)],
        [(0,1,79),(1,1,76),(2,2,72)],[(0,1,72),(1,1,69),(2,2,76)],
        [(0,1,81),(1,1,77),(2,2,72)],[(0,4,72)],
    ]
    build(s,0,prog,8,lead,drums=False,arp=True,arp_duty=0.5,arp_oct=24,arp_vol=0.09,
          bass_pat='soft',bass_vol=0.30,lead_vol=0.26)
    s.save(path)

# ===== 探索(既定/遺跡): さびしく なつかしく 少し不安。余白おおめ・ドラムなし =====
#   非戦闘曲: 急かさず、シンプルで 染み込む メロディ、音の隙間を 大きく。
def make_explore(path):
    s=Song(82, 8*4+2)
    prog=[(57,'min'),(53,'maj'),(60,'maj'),(55,'maj'),
          (57,'min'),(52,'min'),(53,'maj'),(55,'maj')]
    # 短く 覚えやすい モチーフ（あとで 別アレンジで 思い出せる ように）
    lead=[
        [(0,2,69),(2,2,72)],[(0,2,76),(2,2,72)],[(0,2,71),(2,2,67)],[(0,4,69)],
        [(0,2,72),(2,2,69)],[(0,2,67),(2,2,64)],[(0,2,65),(2,2,69)],[(0,4,69)],
    ]
    build(s,0,prog,8,lead,drums=False,arp=True,arp_kind='tri',arp_pat='soft',arp_duty=0.5,
          arp_oct=24,arp_vol=0.07,pad=True,bass_pat='soft',bass_vol=0.28,
          lead_kind='tri',lead_vol=0.26,lead_r=0.18)
    s.save(path)

# ===== ウォーター: しめった洞・流れる水・神秘と物悲しさ。高音ベルの「しずく」 =====
def make_area_water(path):
    s=Song(72, 8*4+2)
    prog=[(57,'min'),(52,'min'),(55,'maj'),(50,'min'),
          (57,'min'),(53,'maj'),(55,'maj'),(45,'min')]
    lead=[
        [(0,3,76),(3,1,74)],[(0,4,72)],[(0,3,71),(3,1,67)],[(0,4,69)],
        [(0,3,79),(3,1,76)],[(0,4,72)],[(0,2,71),(2,2,74)],[(0,4,69)],
    ]
    build(s,0,prog,8,lead,drums=False,arp=True,arp_kind='tri',arp_pat='bell',arp_duty=0.5,
          arp_oct=24,arp_vol=0.09,pad=True,bass_pat='soft',bass_vol=0.26,
          lead_kind='tri',lead_vol=0.24,lead_r=0.22)
    s.save(path)

# ===== 劇場街: 色あせた華やぎ・なつかしいワルツ・ほろ苦さ。オルゴール =====
def make_area_theater(path):
    s=Song(98, 8*4+2)
    prog=[(48,'maj'),(55,'maj'),(57,'min'),(53,'maj'),
          (50,'min'),(55,'maj'),(52,'min'),(55,'maj')]
    lead=[
        [(0,1,72),(1,1,76),(2,2,79)],[(0,2,74),(2,2,71)],
        [(0,1,72),(1,1,69),(2,2,76)],[(0,4,72)],
        [(0,1,74),(1,1,77),(2,2,74)],[(0,2,71),(2,2,67)],
        [(0,1,72),(1,1,71),(2,1,69),(3,1,67)],[(0,4,67)],
    ]
    build(s,0,prog,8,lead,drums=False,arp=True,arp_kind='tri',arp_pat='soft',arp_duty=0.5,
          arp_oct=12,arp_vol=0.08,pad=True,bass_pat='waltz',bass_vol=0.27,
          lead_kind='tri',lead_vol=0.25,lead_r=0.14)
    s.save(path)

# ===== 工場: 不穏・低音・静かな機械・すこし不気味。不協和を ひとつまみ =====
def make_area_factory(path):
    s=Song(88, 8*4+2)
    prog=[(45,'min'),(45,'min'),(46,'dim'),(44,'min'),
          (45,'min'),(48,'maj'),(46,'dim'),(43,'min')]
    lead=[
        [(0,4,69)],[(0,2,68),(2,2,69)],[(0,4,70)],[(0,4,69)],
        [(0,3,72),(3,1,71)],[(0,4,69)],[(0,2,68),(2,2,65)],[(0,4,64)],
    ]
    build(s,0,prog,8,lead,drums=False,arp=False,pad=True,bass_pat='pulse',bass_vol=0.30,
          lead_kind='tri',lead_vol=0.22,lead_duty=0.25,lead_r=0.16)
    # 静かな機械の つめたい刻み（低い矩形を まばらに）
    for bar in range(8):
        s.note(bar*4+0.0,0.12,33,kind='sq',duty=0.5,vol=0.12,a=0.001,d=0.02,s=0.4,r=0.05)
        s.note(bar*4+2.0,0.12,33,kind='sq',duty=0.5,vol=0.10,a=0.001,d=0.02,s=0.4,r=0.05)
    s.save(path)

# ===== 雑魚戦: 軽快・元気 (D minor / 速い) =====
def make_battle(path):
    s=Song(150, 8*4+1)
    prog=[(50,'min'),(46,'maj'),(48,'maj'),(45,'maj')]
    lead=[
        [(0,1,74),(1,1,74),(2,1,77),(3,1,74)],[(0,1,70),(1,1,70),(2,2,74)],
        [(0,1,72),(1,1,72),(2,1,76),(3,1,72)],[(0,2,69),(2,1,73),(3,1,69)],
        [(0,1,77),(1,1,74),(2,1,77),(3,1,81)],[(0,2,77),(2,2,74)],
        [(0,1,79),(1,1,76),(2,2,72)],[(0,2,69),(2,2,69)],
    ]
    build(s,0,prog,8,lead,drums=True,arp=True,arp_duty=0.25,arp_vol=0.14,
          bass_pat='drive',lead_vol=0.30)
    s.save(path)

# ===== ボス: 緊張・不穏 (E minor / 速い) =====
def make_boss(path):
    s=Song(140, 16*4+1)
    prog=[(52,'min'),(48,'maj'),(50,'maj'),(47,'maj')]
    lead=[
        [(0,2,64),(2,1,67),(3,1,71)],[(0,2,72),(2,2,67)],
        [(0,2,74),(2,1,69),(3,1,66)],[(0,2,71),(2,2,67)],
        [(0,1,76),(1,1,74),(2,1,71),(3,1,67)],[(0,2,72),(2,2,76)],
        [(0,1,78),(1,1,74),(2,2,69)],[(0,4,64)],
    ]
    build(s,0,prog,8,lead,drums=True,arp=True,arp_duty=0.125,arp_vol=0.13,
          bass_pat='drive',lead_vol=0.30)
    build(s,8,prog,8,lead,drums=True,arp=True,arp_duty=0.125,arp_vol=0.13,
          bass_pat='drive',lead_vol=0.30,octave=12)   # 2周目は1オクターブ上
    s.save(path)

# ===== 悲しい: スロー・哀愁 (A minor / 遅い) ドラムなし =====
def make_sad(path):
    s=Song(68, 8*4+2)
    prog=[(57,'min'),(52,'min'),(53,'maj'),(48,'maj')]
    lead=[
        [(0,2,69),(2,2,72)],[(0,2,71),(2,2,67)],
        [(0,2,69),(2,2,65)],[(0,4,64)],
        [(0,2,72),(2,2,69)],[(0,2,67),(2,2,64)],
        [(0,2,65),(2,2,69)],[(0,4,60)],
    ]
    build(s,0,prog,8,lead,drums=False,arp=False,pad=True,bass_pat='soft',
          bass_vol=0.32,lead_vol=0.26)
    s.save(path)

# ===== 希望: あたたかい・上昇 (C major / 中速) =====
def make_hopeful(path):
    s=Song(100, 8*4+1)
    prog=[(48,'maj'),(55,'maj'),(57,'min'),(53,'maj')]
    lead=[
        [(0,1,72),(1,1,76),(2,2,79)],[(0,2,74),(2,2,79)],
        [(0,1,76),(1,1,72),(2,2,69)],[(0,2,72),(2,2,77)],
        [(0,1,79),(1,1,76),(2,2,72)],[(0,2,74),(2,2,71)],
        [(0,2,76),(2,2,72)],[(0,4,72)],
    ]
    build(s,0,prog,8,lead,drums=True,drum_vol=0.55,arp=True,arp_duty=0.5,arp_vol=0.10,
          bass_pat='walk',bass_vol=0.34,lead_vol=0.28)
    s.save(path)

# ===== ゲームオーバー: タイトル曲を もの悲しく 変奏した オマージュ（遅い・短調寄り）=====
def make_gameover(path):
    s=Song(60, 8*4+2)
    # タイトルと 同じ コード進行(C-Am-F-G)だが、おわりを 暗く 沈める
    prog=[(48,'maj'),(57,'min'),(53,'maj'),(55,'maj'),
          (48,'maj'),(57,'min'),(50,'min'),(55,'maj')]
    # タイトルの オルゴール旋律を なぞりつつ ゆっくり・哀しく
    lead=[
        [(0,2,72),(2,2,76)],[(0,2,74),(2,2,72)],
        [(0,2,77),(2,2,72)],[(0,4,74)],
        [(0,2,79),(2,2,76)],[(0,2,72),(2,2,69)],
        [(0,2,71),(2,2,67)],[(0,4,72)],
    ]
    build(s,0,prog,8,lead,drums=False,arp=False,pad=True,bass_pat='soft',
          bass_vol=0.30,lead_vol=0.24)
    s.save(path)

# ===== 不穏: スパース・不気味 (半音階 / 遅い) ドラムなし＋心音 =====
def make_dark(path):
    s=Song(72, 8*4+2)
    prog=[(45,'min'),(44,'min'),(46,'dim'),(45,'min'),
          (48,'maj'),(47,'min'),(44,'min'),(40,'min')]
    lead=[
        [(0,4,69)],[(0,4,68)],[(0,2,70),(2,2,75)],[(0,4,69)],
        [(0,4,72)],[(0,4,71)],[(0,3,68),(3,1,69)],[(0,4,64)],
    ]
    build(s,0,prog,8,lead,drums=False,arp=False,pad=True,bass_pat='soft',
          bass_vol=0.30,lead_vol=0.22,lead_duty=0.25)
    # 心音(低い kick)を まばらに
    for bar in range(8):
        s.drum(bar*4+0,'kick',0.30); s.drum(bar*4+2.5,'kick',0.22)
    s.save(path)

# =========================================================
#  モンスター別 戦闘曲（各モンスターの性格に合わせた 8bit戦闘BGM）
# =========================================================
def _battle(path, bpm, prog, lead, bars=16, **kw):
    s=Song(bpm, bars*4+1)
    build(s,0,prog,bars,lead,**kw)
    s.save(path)

MONSTERS = {
 # そよ風の精: そよ風のように 軽く ふわふわ・オルゴール(癒し)
 'm_soyo': dict(bpm=104, prog=[(60,'maj'),(57,'min'),(53,'maj'),(55,'maj')],
   drums=False, bass_pat='waltz', bass_vol=0.26,
   lead_kind='tri', lead_vol=0.30, lead_r=0.16,
   arp=True, arp_kind='tri', arp_pat='bell', arp_duty=0.5, arp_oct=24, arp_vol=0.10, lead=[
   [(0,1.5,72),(1.5,1,76),(2.5,1.5,79)],[(0,2,77),(2,2,72)],
   [(0,1.5,76),(1.5,1,72),(2.5,1.5,69)],[(0,2,71),(2,2,74)],
   [(0,1.5,79),(1.5,1,76),(2.5,1.5,84)],[(0,2,81),(2,2,76)],
   [(0,1.5,77),(1.5,1,74),(2.5,1.5,72)],[(0,4,72)]]),
 # ねむり虫: ねむい 子守唄・とろける(癒し)
 'm_nemu': dict(bpm=80, prog=[(57,'min'),(53,'maj'),(55,'maj'),(52,'min'),
                              (53,'maj'),(48,'maj'),(55,'maj'),(57,'min')],
   bars=8, drums=False, arp=True, arp_kind='tri', arp_pat='soft', arp_duty=0.5,
   arp_oct=24, arp_vol=0.07, pad=True, bass_pat='soft', bass_vol=0.28,
   lead_kind='tri', lead_vol=0.27, lead_r=0.22, lead=[
   [(0,3,76),(3,1,72)],[(0,2,71),(2,2,67)],[(0,2,69),(2,2,72)],[(0,4,74)],
   [(0,3,72),(3,1,69)],[(0,2,67),(2,2,64)],[(0,2,65),(2,2,69)],[(0,4,57)]]),
 # ホネくん: こわがりだけど 愛らしい・ぴょこぴょこ(癒し寄り)
 'm_hone': dict(bpm=118, prog=[(60,'maj'),(55,'maj'),(57,'min'),(53,'maj')],
   drums=True, hat=True, drum_vol=0.32,
   lead_kind='tri', lead_vol=0.30, lead_r=0.10,
   arp=True, arp_kind='tri', arp_pat='soft', arp_duty=0.5, arp_oct=12, arp_vol=0.09,
   bass_pat='pulse', bass_vol=0.28, lead=[
   [(0,1,72),(1,1,72),(2,1,76),(3,1,72)],[(0,2,69),(2,2,67)],
   [(0,1,71),(1,1,71),(2,1,74),(3,1,71)],[(0,2,72),(2,2,76)],
   [(0,1,76),(1,1,74),(2,1,72),(3,1,71)],[(0,2,69),(2,2,72)],
   [(0,1,72),(1,1,71),(2,1,69),(3,1,67)],[(0,4,72)]]),
 # かさこぞう: あそびたがり・ぴょんぴょん はねる(可愛い)
 'm_kasa': dict(bpm=120, prog=[(55,'maj'),(60,'maj'),(57,'min'),(53,'maj')],
   drums=True, hat=True, drum_vol=0.34,
   lead_kind='tri', lead_vol=0.30, lead_r=0.10,
   arp=True, arp_kind='tri', arp_pat='soft', arp_duty=0.5, arp_oct=12, arp_vol=0.09,
   bass_pat='waltz', bass_vol=0.28, lead=[
   [(0,1,74),(1,1,79),(2,2,76)],[(0,1,72),(1,1,76),(2,2,79)],
   [(0,1,76),(1,1,72),(2,2,69)],[(0,2,71),(2,2,74)],
   [(0,1,79),(1,1,76),(2,2,72)],[(0,1,77),(1,1,74),(2,2,81)],
   [(0,1,76),(1,1,72),(2,1,74),(3,1,76)],[(0,4,72)]]),
 # はくしゅ貝: ほがらか・てとてと拍手(可愛い)
 'm_pachi': dict(bpm=114, prog=[(60,'maj'),(55,'maj'),(53,'maj'),(57,'min')],
   drums=True, hat=True, drum_vol=0.30,
   lead_kind='tri', lead_vol=0.30, lead_r=0.10,
   arp=True, arp_kind='tri', arp_pat='bell', arp_duty=0.5, arp_oct=24, arp_vol=0.09,
   bass_pat='pulse', bass_vol=0.28, lead=[
   [(0,1,72),(1,1,76),(2,2,79)],[(0,1,74),(1,1,79),(2,2,76)],
   [(0,1,72),(1,1,76),(2,2,72)],[(0,2,69),(2,2,76)],
   [(0,1,79),(1,1,76),(2,2,84)],[(0,1,77),(1,1,74),(2,2,79)],
   [(0,1,76),(1,1,74),(2,1,72),(3,1,74)],[(0,4,72)]]),
 # キラリ: 華やか・ショー（高音）
 'm_kirari': dict(bpm=142, prog=[(53,'maj'),(48,'maj'),(50,'min'),(46,'maj')],
   arp_duty=0.25, arp_vol=0.12, lead_vol=0.27, lead=[
   [(0,1,77),(1,1,81),(2,2,84)],[(0,1,79),(1,1,84),(2,2,79)],
   [(0,1,81),(1,1,77),(2,2,74)],[(0,1,82),(1,1,77),(2,2,82)],
   [(0,1,84),(1,1,81),(2,2,77)],[(0,1,79),(1,1,76),(2,2,84)],
   [(0,1,81),(1,1,77),(2,1,74),(3,1,77)],[(0,4,77)]]),
 # みずたま: しずくの ように 流れ おちる・澄んだ(癒し)
 'm_shizuku': dict(bpm=100, prog=[(57,'min'),(53,'maj'),(55,'maj'),(48,'maj')],
   drums=False, bass_pat='pulse', bass_vol=0.26,
   lead_kind='tri', lead_vol=0.29, lead_r=0.16,
   arp=True, arp_kind='tri', arp_pat='bell', arp_duty=0.5, arp_oct=24, arp_vol=0.10, lead=[
   [(0,2,69),(2,1,72),(3,1,76)],[(0,2,74),(2,2,69)],
   [(0,2,71),(2,1,74),(3,1,79)],[(0,2,76),(2,2,72)],
   [(0,1,69),(1,1,72),(2,1,76),(3,1,79)],[(0,2,77),(2,2,72)],
   [(0,2,74),(2,2,69)],[(0,4,69)]]),
 # こだま鳥: やまびこの ように やさしく くり返す(癒し)
 'm_kodama': dict(bpm=108, prog=[(53,'maj'),(57,'min'),(55,'maj'),(60,'maj')],
   drums=True, hat=True, drum_vol=0.26,
   lead_kind='tri', lead_vol=0.29, lead_r=0.14,
   arp=True, arp_kind='tri', arp_pat='soft', arp_duty=0.5, arp_oct=12, arp_vol=0.08,
   bass_pat='waltz', bass_vol=0.27, lead=[
   [(0,1,76),(1,1,72),(2,1,76),(3,1,72)],[(0,2,69),(2,2,74)],
   [(0,1,77),(1,1,74),(2,1,77),(3,1,74)],[(0,2,72),(2,2,76)],
   [(0,1,79),(1,1,76),(2,1,79),(3,1,76)],[(0,2,72),(2,2,69)],
   [(0,1,74),(1,1,72),(2,1,71),(3,1,69)],[(0,4,72)]]),
 # ワンダ: あたたかく 元気
 'm_wanda': dict(bpm=132, prog=[(48,'maj'),(57,'min'),(53,'maj'),(55,'maj')],
   arp_duty=0.5, arp_vol=0.11, bass_pat='walk', lead_vol=0.28, lead=[
   [(0,1,72),(1,1,76),(2,2,79)],[(0,2,69),(2,2,72)],
   [(0,1,77),(1,1,72),(2,2,69)],[(0,2,74),(2,2,79)],
   [(0,1,76),(1,1,79),(2,2,72)],[(0,2,69),(2,2,76)],
   [(0,1,77),(1,1,74),(2,1,72),(3,1,71)],[(0,4,72)]]),
 # ねじまき兵: 機械的な行進
 'm_neji': dict(bpm=120, prog=[(57,'min'),(55,'maj'),(57,'min'),(52,'maj')],
   arp_duty=0.125, arp_vol=0.12, lead_vol=0.28, lead=[
   [(0,1,69),(1,1,69),(2,1,72),(3,1,72)],[(0,1,71),(1,1,71),(2,1,67),(3,1,67)],
   [(0,1,69),(1,1,69),(2,1,76),(3,1,76)],[(0,1,68),(1,1,68),(2,2,64)],
   [(0,1,69),(1,1,69),(2,1,72),(3,1,72)],[(0,1,74),(1,1,74),(2,1,71),(3,1,71)],
   [(0,1,69),(1,1,67),(2,1,65),(3,1,64)],[(0,4,57)]]),
 # ほうき霊: きびきび 掃く
 'm_houki': dict(bpm=138, prog=[(50,'min'),(46,'maj'),(48,'maj'),(45,'maj')],
   arp_duty=0.25, arp_vol=0.12, lead_vol=0.28, lead=[
   [(0,1,74),(1,1,77),(2,1,74),(3,1,69)],[(0,1,70),(1,1,74),(2,2,70)],
   [(0,1,72),(1,1,76),(2,1,72),(3,1,67)],[(0,2,69),(2,2,73)],
   [(0,1,77),(1,1,74),(2,1,77),(3,1,81)],[(0,1,70),(1,1,74),(2,2,77)],
   [(0,1,76),(1,1,72),(2,1,69),(3,1,67)],[(0,4,62)]]),
 # カクト: きちょうめん・冷たく激しい（専用ボス）
 'm_kakuto': dict(bpm=144, prog=[(52,'min'),(48,'maj'),(50,'maj'),(47,'maj')],
   arp_duty=0.125, arp_vol=0.13, lead_vol=0.30, lead=[
   [(0,1,76),(1,1,76),(2,1,71),(3,1,67)],[(0,2,72),(2,2,76)],
   [(0,1,74),(1,1,74),(2,1,69),(3,1,66)],[(0,2,71),(2,2,67)],
   [(0,1,79),(1,1,76),(2,1,74),(3,1,71)],[(0,2,72),(2,2,76)],
   [(0,1,78),(1,1,74),(2,1,71),(3,1,69)],[(0,4,64)]]),
}
def make_monster(name, path):
    c=dict(MONSTERS[name]); bpm=c.pop('bpm'); prog=c.pop('prog'); lead=c.pop('lead')
    _battle(path, bpm, prog, lead, **c)

# ===== ホムラ(保護者): 温かい・母性・少し寂しい =====
def make_homura(path):
    s=Song(76, 8*4+2)
    prog=[(48,'maj'),(57,'min'),(53,'maj'),(55,'maj'),
          (48,'maj'),(52,'min'),(53,'maj'),(55,'maj')]
    lead=[
        [(0,2,72),(2,2,76)],[(0,2,71),(2,2,69)],[(0,2,72),(2,2,77)],[(0,4,74)],
        [(0,2,76),(2,2,79)],[(0,2,72),(2,2,69)],[(0,2,71),(2,2,67)],[(0,4,72)],
    ]
    build(s,0,prog,8,lead,drums=False,arp=True,arp_duty=0.5,arp_oct=24,arp_vol=0.08,
          pad=True,bass_pat='soft',bass_vol=0.30,lead_vol=0.25)
    s.save(path)

# ===== ホムラ戦: 別れの戦い（勇ましさ3:悲しさ7・勝っても嬉しくない）=====
def make_homura_boss(path):
    s=Song(118, 16*4+1)
    prog=[(57,'min'),(53,'maj'),(55,'maj'),(52,'min')]
    lead=[
        [(0,2,69),(2,1,72),(3,1,71)],[(0,2,69),(2,2,65)],
        [(0,2,67),(2,1,74),(3,1,72)],[(0,2,71),(2,2,67)],
        [(0,1,76),(1,1,74),(2,2,69)],[(0,2,72),(2,2,69)],
        [(0,1,74),(1,1,72),(2,1,71),(3,1,67)],[(0,4,69)],
    ]
    build(s,0,prog,8,lead,drums=True,drum_vol=0.5,arp=True,arp_duty=0.5,arp_vol=0.11,
          bass_pat='walk',bass_vol=0.34,lead_vol=0.29)
    build(s,8,prog,8,lead,drums=True,drum_vol=0.5,arp=True,arp_duty=0.25,arp_vol=0.11,
          bass_pat='walk',bass_vol=0.34,lead_vol=0.29,octave=12)
    s.save(path)

# =========================================================
#  ベルゲントリュッケン風 ── 王が 最後の戦いを 受け入れる 17〜21秒の 儀式イントロ
#  D minor / 4/4 / BPM100(重い ハーフテンポ感) / オルガン＋合唱＋低ブラス・音数少なめ
#  進行: Dm Dm Bb C Gm Bb A A（A=V で C#→D の運命感、最後は 未解決で 戦闘へ）
# =========================================================
def make_berg(path):
    BPM=100
    s=Song(BPM, 8*4 + 2)
    prog=[(50,'min'),(50,'min'),(46,'maj'),(48,'maj'),
          (43,'min'),(46,'maj'),(45,'maj'),(45,'maj')]   # Dm Dm Bb C Gm Bb A A
    # 王の旋律: ゆっくり 上がって 重く 下がる（D,F,A中心、最後に C#5→D5 の 解決圧）
    melody=[
        [(0,4,62)],              # 1 Dm: D（のばし）
        [(0,2,65),(2,2,64)],     # 2 Dm: F → E
        [(0,2,62),(2,2,65)],     # 3 Bb: D → F
        [(0,2,67),(2,2,69)],     # 4 C : G → A
        [(0,2,70),(2,2,69)],     # 5 Gm: Bb → A
        [(0,2,65),(2,2,62)],     # 6 Bb: F → D
        [(0,2,69),(2,2,73)],     # 7 A : A → C#5（緊張）
        [(0,4,74)],              # 8 A : D5（未解決の 余韻 → 次の戦闘へ）
    ]
    for bar,(root,kind) in enumerate(prog):
        base=bar*4
        tones=chord_tones(root,kind)
        # パイプオルガン風: square50% の 和音を 2オクターブ 重ねて 厚く
        for tn in tones:
            s.note(base,3.9,tn,    kind='sq',duty=0.5,vol=0.10,a=0.06,d=0.2,s=0.85,r=0.5)
            s.note(base,3.9,tn+12, kind='sq',duty=0.5,vol=0.06,a=0.08,d=0.2,s=0.85,r=0.5)
        # 合唱／ストリングス風パッド: 三角で やわらかく
        for tn in tones:
            s.note(base,3.9,tn+12, kind='tri',vol=0.10,a=0.14,d=0.2,s=0.9,r=0.6)
        # 低いブラス風: 根音を 1オクターブ下で 太く
        s.note(base,3.9,root-12,kind='tri',vol=0.34,a=0.06,d=0.15,s=0.9,r=0.5)
        s.note(base,3.9,root-12,kind='sq',duty=0.5,vol=0.07,a=0.08,d=0.2,s=0.85,r=0.5)
        # 主旋律（オルガン風 square50%）
        for (off,dur,mid) in melody[bar]:
            s.note(base+off,dur*0.96,mid,kind='sq',duty=0.5,vol=0.22,a=0.04,d=0.1,s=0.8,r=0.25)
    # ティンパニ風: 最後の 2小節の 頭に 低い打撃を 1回ずつ だけ
    s.drum((8-2)*4+0,'kick',0.42)
    s.drum((8-1)*4+0,'kick',0.5)
    s.save(path)

ALL={'battle':make_argor_battle,'intro':make_argor_intro,'title':make_title,'berg':make_berg,
     'explore':make_explore,'zako':make_battle,'boss':make_boss,'sad':make_sad,
     'hopeful':make_hopeful,'dark':make_dark,'gameover':make_gameover,
     'homura':make_homura,'homura_boss':make_homura_boss,
     'water':make_area_water,'theater':make_area_theater,'factory':make_area_factory}
PATH={'battle':'music/argor_battle','intro':'music/argor_intro','title':'music/title','berg':'music/berg',
      'explore':'music/explore','zako':'music/battle','boss':'music/boss','sad':'music/sad',
      'hopeful':'music/hopeful','dark':'music/dark','gameover':'music/gameover',
      'homura':'music/homura','homura_boss':'music/homura_boss',
      'water':'music/area_water','theater':'music/area_theater','factory':'music/area_factory'}

if __name__ == '__main__':
    which = sys.argv[1] if len(sys.argv)>1 else 'all'
    os.makedirs('music', exist_ok=True)
    if which == 'monsters':
        for name in MONSTERS:
            make_monster(name, 'music/'+name)
    elif which in MONSTERS:
        make_monster(which, 'music/'+which)
    elif which == 'all':
        for k in ALL: ALL[k](PATH[k])
        for name in MONSTERS: make_monster(name, 'music/'+name)
    else:
        ALL[which](PATH[which])
