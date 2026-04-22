"""
Generate synthetic demo sepsis patients (DEMO_001 ~ DEMO_010).

Mirrors the schema of s3_sample/P-xxx/ (patient_meta.json, vital_ts.parquet,
lab_df.parquet) but values are fully synthesized per scenario — no real-patient
values are copied.

Output: demo/patients/DEMO_00X/{patient_meta.json, vital_ts.parquet, lab_df.parquet}

Usage:
    python scripts/generate_demo_patients.py
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "demo" / "patients"

INTIME = datetime(2024, 5, 1, 6, 0, 0)
SEPSIS_ONSET = datetime(2024, 5, 1, 12, 0, 0)
WINDOW_END = datetime(2024, 5, 3, 6, 0, 0)
N_HOURS = 49  # inclusive both endpoints — matches real samples (49 rows)


VITAL_COLS = [
    "charttime",
    "glucose_vital",
    "heart_rate",
    "mbp",
    "map",
    "sbp",
    "dbp",
    "resp_rate",
    "spo2",
    "temperature",
    "gcs",
    "pao2fio2ratio",
    "ventilation",
]

LAB_COLS = [
    "charttime",
    "lactate",
    "creatinine",
    "bun",
    "sodium",
    "potassium",
    "urine_output",
    "glucose",
    "bicarbonate",
    "ph",
    "albumin",
    "wbc",
    "rdw",
    "aptt",
    "inr",
    "platelet",
    "hemoglobin",
    "bilirubin_total",
    "norepinephrine",
    "dopamine",
    "dobutamine",
    "epinephrine",
    "po2",
    "fio2_bg",
    "peep_feat",
]


@dataclass
class Trend:
    """Time-varying baseline with linear drift + gentle sinusoidal drift + noise."""
    start: float  # value at t=0 (h)
    end: float    # value at t=48 (h)
    noise: float  # gaussian sigma
    floor: float | None = None
    ceiling: float | None = None
    wiggle: float = 0.0  # amplitude of sinusoidal component

    def sample(self, t_hours: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        frac = np.clip(t_hours / 48.0, 0.0, 1.0)
        base = self.start + (self.end - self.start) * frac
        if self.wiggle:
            base = base + self.wiggle * np.sin(2 * math.pi * t_hours / 18.0)
        noise = rng.normal(0.0, self.noise, size=t_hours.shape)
        out = base + noise
        if self.floor is not None:
            out = np.maximum(out, self.floor)
        if self.ceiling is not None:
            out = np.minimum(out, self.ceiling)
        return out


@dataclass
class Scenario:
    demo_id: str
    age: int
    gender: int
    sofa_score: int
    flags: dict = field(default_factory=dict)
    note: str = ""

    # vital trends (start,end over 48h)
    heart_rate: Trend = field(default_factory=lambda: Trend(95, 90, 6.0, 50, 160, wiggle=3))
    sbp: Trend = field(default_factory=lambda: Trend(115, 115, 8.0, 60, 190))
    dbp: Trend = field(default_factory=lambda: Trend(65, 65, 5.0, 30, 110))
    map_: Trend = field(default_factory=lambda: Trend(78, 78, 5.0, 40, 120))
    mbp: Trend = field(default_factory=lambda: Trend(78, 78, 5.0, 40, 120))
    resp_rate: Trend = field(default_factory=lambda: Trend(21, 20, 2.5, 8, 40))
    spo2: Trend = field(default_factory=lambda: Trend(96, 97, 1.2, 70, 100, wiggle=0.3))
    temperature: Trend = field(default_factory=lambda: Trend(38.0, 37.6, 0.35, 34.0, 41.0))
    glucose_vital: Trend = field(default_factory=lambda: Trend(120, 115, 15.0, 60, 400))
    gcs: Trend = field(default_factory=lambda: Trend(14, 14, 0.4, 3, 15))
    pao2fio2ratio: Trend = field(default_factory=lambda: Trend(260, 280, 30.0, 40, 500))
    ventilation_prob: float = 0.25  # probability per hour

    # lab trends (start,end over 48h)
    lactate: Trend = field(default_factory=lambda: Trend(2.5, 2.0, 0.35, 0.4, 15))
    creatinine: Trend = field(default_factory=lambda: Trend(1.3, 1.4, 0.15, 0.3, 10))
    bun: Trend = field(default_factory=lambda: Trend(20, 22, 3.5, 4, 120))
    sodium: Trend = field(default_factory=lambda: Trend(137, 137, 2.0, 120, 160))
    potassium: Trend = field(default_factory=lambda: Trend(4.1, 4.0, 0.3, 2.5, 7.0))
    urine_output: Trend = field(default_factory=lambda: Trend(55, 55, 12.0, 0, 200))
    glucose: Trend = field(default_factory=lambda: Trend(125, 120, 20.0, 60, 400))
    bicarbonate: Trend = field(default_factory=lambda: Trend(22, 22, 1.5, 8, 35))
    ph: Trend = field(default_factory=lambda: Trend(7.40, 7.41, 0.04, 6.90, 7.60))
    albumin: Trend = field(default_factory=lambda: Trend(3.2, 3.1, 0.3, 1.5, 5.0))
    wbc: Trend = field(default_factory=lambda: Trend(12, 11, 2.5, 0.2, 40))
    rdw: Trend = field(default_factory=lambda: Trend(13, 13, 0.6, 10, 20))
    aptt: Trend = field(default_factory=lambda: Trend(32, 30, 4.0, 18, 120))
    inr: Trend = field(default_factory=lambda: Trend(1.2, 1.2, 0.1, 0.8, 6.0))
    platelet: Trend = field(default_factory=lambda: Trend(200, 190, 25.0, 5, 600))
    hemoglobin: Trend = field(default_factory=lambda: Trend(10.5, 10.3, 0.6, 4, 18))
    bilirubin_total: Trend = field(default_factory=lambda: Trend(1.2, 1.2, 0.4, 0.2, 25))
    norepinephrine: Trend = field(default_factory=lambda: Trend(0.0, 0.0, 0.01, 0.0, 2.0))
    dopamine: Trend = field(default_factory=lambda: Trend(0.0, 0.0, 0.05, 0.0, 20.0))
    dobutamine: Trend = field(default_factory=lambda: Trend(0.0, 0.0, 0.05, 0.0, 20.0))
    epinephrine: Trend = field(default_factory=lambda: Trend(0.0, 0.0, 0.005, 0.0, 2.0))
    po2: Trend = field(default_factory=lambda: Trend(85, 90, 6.0, 40, 200))
    fio2_bg: Trend = field(default_factory=lambda: Trend(0.30, 0.28, 0.04, 0.21, 1.0))
    peep_feat: Trend = field(default_factory=lambda: Trend(5.5, 5.0, 1.0, 0.0, 20.0))

    n_labs: int = 12


def build_vitals(sc: Scenario, rng: np.random.Generator) -> pd.DataFrame:
    t = np.arange(N_HOURS, dtype=float)
    timestamps = [INTIME + timedelta(hours=int(h)) for h in t]

    # SBP/DBP drive MAP via rough formula but we also keep our own sampled MAP
    sbp = sc.sbp.sample(t, rng)
    dbp = np.minimum(sc.dbp.sample(t, rng), sbp - 5)  # dbp < sbp
    map_calc = (sbp + 2 * dbp) / 3.0
    map_sampled = sc.map_.sample(t, rng)
    # blend calculated with scenario-sampled MAP so shock scenarios stay low
    map_val = 0.6 * map_sampled + 0.4 * map_calc
    mbp = sc.mbp.sample(t, rng)

    vent = (rng.random(N_HOURS) < sc.ventilation_prob).astype(np.int64)

    df = pd.DataFrame({
        "charttime": pd.to_datetime(timestamps),
        "glucose_vital": sc.glucose_vital.sample(t, rng),
        "heart_rate": sc.heart_rate.sample(t, rng),
        "mbp": mbp,
        "map": map_val,
        "sbp": sbp,
        "dbp": dbp,
        "resp_rate": sc.resp_rate.sample(t, rng),
        "spo2": sc.spo2.sample(t, rng),
        "temperature": sc.temperature.sample(t, rng),
        "gcs": np.round(sc.gcs.sample(t, rng)).clip(3, 15),
        "pao2fio2ratio": sc.pao2fio2ratio.sample(t, rng),
        "ventilation": vent,
    })
    df = df[VITAL_COLS]
    return df


def build_labs(sc: Scenario, rng: np.random.Generator) -> pd.DataFrame:
    # irregular sampling across 48h; first draw near intime
    n = sc.n_labs
    # sort random hours within [0,48]; keep at least one near 0h and one near 48h
    hours = np.sort(np.concatenate([
        np.array([0.0, 48.0]),
        rng.uniform(1.0, 47.5, size=max(0, n - 2)),
    ]))[:n]
    t = hours.astype(float)
    timestamps = [INTIME + timedelta(hours=float(h)) for h in t]

    df = pd.DataFrame({
        "charttime": pd.to_datetime(timestamps),
        "lactate": sc.lactate.sample(t, rng),
        "creatinine": sc.creatinine.sample(t, rng),
        "bun": sc.bun.sample(t, rng),
        "sodium": sc.sodium.sample(t, rng),
        "potassium": sc.potassium.sample(t, rng),
        "urine_output": sc.urine_output.sample(t, rng),
        "glucose": sc.glucose.sample(t, rng),
        "bicarbonate": sc.bicarbonate.sample(t, rng),
        "ph": sc.ph.sample(t, rng),
        "albumin": sc.albumin.sample(t, rng),
        "wbc": sc.wbc.sample(t, rng),
        "rdw": sc.rdw.sample(t, rng),
        "aptt": sc.aptt.sample(t, rng),
        "inr": sc.inr.sample(t, rng),
        "platelet": sc.platelet.sample(t, rng),
        "hemoglobin": sc.hemoglobin.sample(t, rng),
        "bilirubin_total": sc.bilirubin_total.sample(t, rng),
        "norepinephrine": sc.norepinephrine.sample(t, rng),
        "dopamine": sc.dopamine.sample(t, rng),
        "dobutamine": sc.dobutamine.sample(t, rng),
        "epinephrine": sc.epinephrine.sample(t, rng),
        "po2": sc.po2.sample(t, rng),
        "fio2_bg": sc.fio2_bg.sample(t, rng),
        "peep_feat": sc.peep_feat.sample(t, rng),
    })
    df = df[LAB_COLS]
    return df


DEMO_NAMES = {
    "DEMO_001": "김영호",
    "DEMO_002": "이정숙",
    "DEMO_003": "박철수",
    "DEMO_004": "최미경",
    "DEMO_005": "정성훈",
    "DEMO_006": "강혜진",
    "DEMO_007": "윤기태",
    "DEMO_008": "임지현",
    "DEMO_009": "한승우",
    "DEMO_010": "송만식",
}


def build_meta(sc: Scenario) -> dict:
    meta = {
        "patient_id": sc.demo_id,
        "name": DEMO_NAMES.get(sc.demo_id, ""),
        "age": sc.age,
        "gender": sc.gender,
        "label": 1,  # all scenarios are confirmed sepsis
        "intime": INTIME.isoformat(),
        "sofa_score": sc.sofa_score,
        "sepsis_onset_time": SEPSIS_ONSET.isoformat(),
        "window_start_vital": INTIME.isoformat(),
        "window_start_lab": INTIME.isoformat(),
        "window_end": WINDOW_END.isoformat(),
        "flag_liver_failure": sc.flags.get("liver_failure", 0),
        "flag_ckd": sc.flags.get("ckd", 0),
        "flag_coagulopathy": sc.flags.get("coagulopathy", 0),
        "flag_diabetes": sc.flags.get("diabetes", 0),
        "flag_immunosuppression": sc.flags.get("immunosuppression", 0),
        "flag_chf": sc.flags.get("chf", 0),
        "flag_septic_shock_hx": sc.flags.get("septic_shock_hx", 0),
    }
    return meta


# --- Scenario factory ---------------------------------------------------------

def scenarios() -> list[Scenario]:
    scs: list[Scenario] = []

    # DEMO_001: septic shock + elderly (75)
    sc = Scenario(
        demo_id="DEMO_001", age=75, gender=1, sofa_score=11,
        flags={"septic_shock_hx": 1},
        note="septic shock, elderly — hypotension on pressors, high lactate",
    )
    sc.heart_rate = Trend(118, 125, 7.0, 50, 170)
    sc.sbp = Trend(88, 82, 6.0, 60, 140)
    sc.dbp = Trend(48, 45, 4.0, 30, 90)
    sc.map_ = Trend(58, 55, 4.0, 40, 100)
    sc.mbp = Trend(60, 57, 4.0, 40, 100)
    sc.resp_rate = Trend(26, 28, 3.0, 10, 40)
    sc.spo2 = Trend(93, 93, 1.5, 80, 100)
    sc.temperature = Trend(38.9, 38.5, 0.4, 35, 41)
    sc.pao2fio2ratio = Trend(210, 195, 25.0, 80, 400)
    sc.ventilation_prob = 0.55
    sc.lactate = Trend(5.8, 5.2, 0.6, 2.0, 12)
    sc.norepinephrine = Trend(0.35, 0.42, 0.05, 0.0, 2.0)
    sc.epinephrine = Trend(0.04, 0.06, 0.015, 0.0, 1.5)
    sc.urine_output = Trend(22, 18, 6.0, 0, 120)
    sc.creatinine = Trend(1.8, 2.1, 0.2, 0.3, 6)
    sc.bicarbonate = Trend(17, 16, 1.2, 8, 30)
    sc.ph = Trend(7.28, 7.27, 0.03, 6.9, 7.5)
    sc.wbc = Trend(19, 21, 3.0, 1, 40)
    scs.append(sc)

    # DEMO_002: sepsis + ARDS
    sc = Scenario(
        demo_id="DEMO_002", age=61, gender=0, sofa_score=9,
        flags={},
        note="sepsis + ARDS — low P/F, high FiO2, high PEEP",
    )
    sc.heart_rate = Trend(108, 112, 6.0, 60, 150)
    sc.spo2 = Trend(89, 91, 1.8, 75, 98)
    sc.resp_rate = Trend(28, 30, 3.0, 12, 40)
    sc.pao2fio2ratio = Trend(110, 95, 15.0, 50, 250)
    sc.ventilation_prob = 0.95
    sc.temperature = Trend(38.7, 38.4, 0.4, 35, 41)
    sc.po2 = Trend(62, 60, 6.0, 40, 120)
    sc.fio2_bg = Trend(0.75, 0.80, 0.08, 0.4, 1.0)
    sc.peep_feat = Trend(12, 13, 1.5, 5, 20)
    sc.lactate = Trend(3.2, 2.8, 0.4, 1.0, 10)
    sc.norepinephrine = Trend(0.10, 0.12, 0.03, 0.0, 1.0)
    scs.append(sc)

    # DEMO_003: sepsis + AKI
    sc = Scenario(
        demo_id="DEMO_003", age=68, gender=1, sofa_score=8,
        flags={"ckd": 0},
        note="sepsis + AKI — creatinine rising, low urine output",
    )
    sc.heart_rate = Trend(104, 108, 6.0, 60, 150)
    sc.creatinine = Trend(1.4, 3.2, 0.3, 0.5, 8)
    sc.bun = Trend(32, 55, 5.0, 10, 120)
    sc.potassium = Trend(4.8, 5.6, 0.3, 3, 7)
    sc.urine_output = Trend(18, 12, 6.0, 0, 80)
    sc.lactate = Trend(3.0, 2.6, 0.4, 1.0, 8)
    sc.bicarbonate = Trend(18, 17, 1.3, 8, 30)
    sc.norepinephrine = Trend(0.05, 0.06, 0.02, 0.0, 1.0)
    sc.ph = Trend(7.31, 7.30, 0.03, 6.9, 7.5)
    sc.temperature = Trend(38.5, 38.2, 0.4, 35, 41)
    scs.append(sc)

    # DEMO_004: sepsis + SIC (coagulopathy)
    sc = Scenario(
        demo_id="DEMO_004", age=58, gender=0, sofa_score=8,
        flags={"coagulopathy": 1},
        note="sepsis + SIC — platelet drop, INR elevated",
    )
    sc.platelet = Trend(90, 55, 10.0, 5, 300)
    sc.inr = Trend(1.9, 2.4, 0.2, 0.8, 6)
    sc.aptt = Trend(52, 62, 6.0, 20, 150)
    sc.bilirubin_total = Trend(2.2, 2.8, 0.4, 0.3, 15)
    sc.hemoglobin = Trend(9.2, 8.6, 0.5, 4, 16)
    sc.heart_rate = Trend(106, 108, 6.0, 60, 150)
    sc.lactate = Trend(2.8, 2.6, 0.4, 1.0, 8)
    sc.norepinephrine = Trend(0.08, 0.09, 0.02, 0.0, 1.0)
    sc.temperature = Trend(38.6, 38.3, 0.4, 35, 41)
    scs.append(sc)

    # DEMO_005: complex — sepsis + ARDS + AKI
    sc = Scenario(
        demo_id="DEMO_005", age=71, gender=1, sofa_score=14,
        flags={"septic_shock_hx": 1},
        note="sepsis + ARDS + AKI — all bad",
    )
    sc.heart_rate = Trend(120, 124, 7.0, 60, 170)
    sc.sbp = Trend(92, 88, 7.0, 60, 140)
    sc.dbp = Trend(50, 47, 4.0, 30, 90)
    sc.map_ = Trend(62, 58, 4.0, 40, 100)
    sc.mbp = Trend(63, 60, 4.0, 40, 100)
    sc.spo2 = Trend(88, 87, 2.0, 75, 98)
    sc.resp_rate = Trend(30, 32, 3.0, 12, 45)
    sc.pao2fio2ratio = Trend(95, 80, 12.0, 40, 200)
    sc.ventilation_prob = 1.0
    sc.po2 = Trend(58, 55, 6.0, 35, 100)
    sc.fio2_bg = Trend(0.85, 0.90, 0.05, 0.5, 1.0)
    sc.peep_feat = Trend(14, 15, 1.5, 5, 22)
    sc.lactate = Trend(6.2, 6.8, 0.5, 2, 15)
    sc.creatinine = Trend(2.4, 3.6, 0.3, 0.5, 9)
    sc.bun = Trend(48, 68, 6.0, 10, 140)
    sc.urine_output = Trend(10, 6, 4.0, 0, 50)
    sc.norepinephrine = Trend(0.55, 0.70, 0.08, 0.0, 2.5)
    sc.epinephrine = Trend(0.08, 0.12, 0.02, 0.0, 1.5)
    sc.bicarbonate = Trend(15, 13, 1.5, 6, 30)
    sc.ph = Trend(7.22, 7.20, 0.03, 6.9, 7.5)
    sc.platelet = Trend(95, 70, 12.0, 5, 300)
    sc.inr = Trend(1.7, 2.0, 0.2, 0.8, 6)
    sc.temperature = Trend(39.2, 38.8, 0.4, 35, 41.5)
    scs.append(sc)

    # DEMO_006: moderate sepsis, stabilizing
    sc = Scenario(
        demo_id="DEMO_006", age=55, gender=0, sofa_score=5,
        flags={},
        note="moderate sepsis, stabilizing — trending toward normal",
    )
    sc.heart_rate = Trend(110, 88, 5.0, 60, 150)
    sc.sbp = Trend(105, 122, 7.0, 70, 160)
    sc.map_ = Trend(72, 85, 5.0, 50, 110)
    sc.mbp = Trend(72, 84, 5.0, 50, 110)
    sc.temperature = Trend(38.7, 37.2, 0.35, 35, 40)
    sc.resp_rate = Trend(24, 17, 2.5, 10, 36)
    sc.spo2 = Trend(94, 98, 1.0, 85, 100)
    sc.lactate = Trend(3.2, 1.4, 0.35, 0.5, 8)
    sc.wbc = Trend(15, 10, 2.0, 1, 30)
    sc.creatinine = Trend(1.4, 1.0, 0.12, 0.4, 4)
    sc.urine_output = Trend(42, 68, 10.0, 10, 200)
    sc.norepinephrine = Trend(0.08, 0.0, 0.015, 0.0, 1.0)
    scs.append(sc)

    # DEMO_007: moderate sepsis + diabetes, slow recovery
    sc = Scenario(
        demo_id="DEMO_007", age=63, gender=1, sofa_score=6,
        flags={"diabetes": 1},
        note="moderate sepsis + diabetes — glucose unstable, slow recovery",
    )
    sc.heart_rate = Trend(108, 102, 7.0, 60, 150)
    sc.temperature = Trend(38.5, 38.0, 0.35, 35, 40)
    sc.glucose_vital = Trend(235, 205, 45.0, 70, 420)
    sc.glucose = Trend(250, 215, 50.0, 70, 450)
    sc.lactate = Trend(2.8, 2.4, 0.4, 0.5, 8)
    sc.wbc = Trend(14, 12, 2.0, 1, 30)
    sc.creatinine = Trend(1.5, 1.4, 0.15, 0.4, 5)
    sc.urine_output = Trend(40, 45, 10.0, 10, 180)
    sc.norepinephrine = Trend(0.05, 0.03, 0.015, 0.0, 1.0)
    scs.append(sc)

    # DEMO_008: mild sepsis, young (38)
    sc = Scenario(
        demo_id="DEMO_008", age=38, gender=0, sofa_score=3,
        flags={},
        note="mild sepsis, young — borderline values",
    )
    sc.heart_rate = Trend(98, 90, 5.0, 55, 140)
    sc.sbp = Trend(118, 122, 7.0, 80, 160)
    sc.map_ = Trend(82, 85, 5.0, 55, 110)
    sc.mbp = Trend(82, 85, 5.0, 55, 110)
    sc.temperature = Trend(38.2, 37.6, 0.3, 35, 40)
    sc.resp_rate = Trend(20, 17, 2.0, 10, 32)
    sc.spo2 = Trend(96, 98, 1.0, 88, 100)
    sc.pao2fio2ratio = Trend(320, 360, 30.0, 150, 500)
    sc.ventilation_prob = 0.05
    sc.lactate = Trend(2.1, 1.6, 0.3, 0.5, 6)
    sc.wbc = Trend(13, 10, 2.0, 1, 25)
    sc.creatinine = Trend(1.0, 0.9, 0.1, 0.4, 3)
    sc.urine_output = Trend(70, 85, 12.0, 20, 220)
    scs.append(sc)

    # DEMO_009: early sepsis, rapid improvement
    sc = Scenario(
        demo_id="DEMO_009", age=47, gender=1, sofa_score=5,
        flags={},
        note="early sepsis, rapid improvement after intervention",
    )
    sc.heart_rate = Trend(120, 80, 5.0, 55, 160)
    sc.sbp = Trend(100, 125, 7.0, 70, 160)
    sc.map_ = Trend(70, 88, 5.0, 50, 110)
    sc.mbp = Trend(70, 87, 5.0, 50, 110)
    sc.temperature = Trend(39.1, 37.0, 0.35, 35, 41)
    sc.resp_rate = Trend(28, 15, 2.5, 10, 40)
    sc.spo2 = Trend(93, 98, 1.0, 80, 100)
    sc.lactate = Trend(4.5, 1.3, 0.4, 0.5, 10)
    sc.wbc = Trend(18, 9, 2.5, 1, 30)
    sc.creatinine = Trend(1.6, 0.95, 0.15, 0.4, 5)
    sc.urine_output = Trend(30, 75, 12.0, 10, 220)
    sc.norepinephrine = Trend(0.15, 0.0, 0.02, 0.0, 1.5)
    sc.ph = Trend(7.30, 7.42, 0.03, 7.0, 7.55)
    sc.bicarbonate = Trend(17, 23, 1.5, 8, 32)
    scs.append(sc)

    # DEMO_010: worst case, elderly (82) + multi-complication
    sc = Scenario(
        demo_id="DEMO_010", age=82, gender=1, sofa_score=16,
        flags={"ckd": 1, "chf": 1, "septic_shock_hx": 1, "coagulopathy": 1},
        note="worst case — elderly, multi-organ failure, trending down",
    )
    sc.heart_rate = Trend(128, 135, 8.0, 60, 180)
    sc.sbp = Trend(82, 74, 7.0, 50, 130)
    sc.dbp = Trend(44, 40, 4.0, 25, 80)
    sc.map_ = Trend(56, 50, 4.0, 35, 95)
    sc.mbp = Trend(57, 51, 4.0, 35, 95)
    sc.spo2 = Trend(86, 84, 2.0, 72, 98)
    sc.resp_rate = Trend(32, 34, 3.0, 12, 45)
    sc.temperature = Trend(35.6, 35.2, 0.45, 33, 40)  # hypothermia
    sc.pao2fio2ratio = Trend(80, 65, 12.0, 40, 200)
    sc.ventilation_prob = 1.0
    sc.gcs = Trend(9, 7, 0.6, 3, 14)
    sc.lactate = Trend(7.8, 9.2, 0.6, 2, 18)
    sc.creatinine = Trend(3.2, 4.1, 0.3, 0.5, 10)
    sc.bun = Trend(72, 95, 6.0, 15, 160)
    sc.urine_output = Trend(6, 3, 3.0, 0, 40)
    sc.norepinephrine = Trend(0.85, 1.10, 0.1, 0.0, 3.0)
    sc.epinephrine = Trend(0.15, 0.22, 0.03, 0.0, 2.0)
    sc.dopamine = Trend(5.0, 6.5, 1.0, 0.0, 20.0)
    sc.bicarbonate = Trend(13, 11, 1.5, 6, 28)
    sc.ph = Trend(7.16, 7.12, 0.04, 6.85, 7.45)
    sc.platelet = Trend(65, 35, 10.0, 5, 250)
    sc.inr = Trend(2.4, 2.9, 0.25, 0.8, 7)
    sc.aptt = Trend(68, 82, 8.0, 20, 180)
    sc.bilirubin_total = Trend(3.5, 4.8, 0.5, 0.3, 25)
    sc.wbc = Trend(22, 26, 3.5, 0.5, 50)
    sc.hemoglobin = Trend(8.4, 7.6, 0.5, 4, 15)
    sc.po2 = Trend(55, 50, 6.0, 35, 100)
    sc.fio2_bg = Trend(0.90, 0.95, 0.05, 0.5, 1.0)
    sc.peep_feat = Trend(14, 16, 1.5, 5, 22)
    sc.n_labs = 16
    scs.append(sc)

    return scs


# --- Main ---------------------------------------------------------------------

def generate_all(seed: int = 20240501) -> list[tuple[Scenario, pd.DataFrame, pd.DataFrame, dict]]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    outputs = []
    for i, sc in enumerate(scenarios()):
        rng = np.random.default_rng(seed + i)
        vitals = build_vitals(sc, rng)
        labs = build_labs(sc, rng)
        meta = build_meta(sc)

        pdir = OUT_DIR / sc.demo_id
        pdir.mkdir(parents=True, exist_ok=True)
        vitals.to_parquet(pdir / "vital_ts.parquet", index=False)
        labs.to_parquet(pdir / "lab_df.parquet", index=False)
        with open(pdir / "patient_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f)
        outputs.append((sc, vitals, labs, meta))
    return outputs


def summarize(outputs):
    print("\n" + "=" * 78)
    print("SUMMARY - synthetic demo patients")
    print("=" * 78)
    for sc, vitals, labs, meta in outputs:
        print(f"\n[{sc.demo_id}] age={sc.age} gender={sc.gender} sofa={sc.sofa_score}")
        print(f"  scenario: {sc.note}")
        print(f"  vitals shape={vitals.shape}  labs shape={labs.shape}")
        # key vital stats
        v_min = {c: float(vitals[c].min()) for c in ["map", "sbp", "spo2", "pao2fio2ratio"]}
        v_max = {c: float(vitals[c].max()) for c in ["heart_rate", "temperature", "resp_rate"]}
        print(f"  vital min: map={v_min['map']:.1f} sbp={v_min['sbp']:.1f} "
              f"spo2={v_min['spo2']:.1f} P/F={v_min['pao2fio2ratio']:.0f}")
        print(f"  vital max: hr={v_max['heart_rate']:.0f} temp={v_max['temperature']:.1f} "
              f"rr={v_max['resp_rate']:.0f}")
        # key labs
        l_max = {c: float(labs[c].max()) for c in ["lactate", "creatinine", "inr", "norepinephrine"]}
        l_min = {c: float(labs[c].min()) for c in ["platelet", "ph", "urine_output"]}
        print(f"  lab max: lactate={l_max['lactate']:.2f} creat={l_max['creatinine']:.2f} "
              f"inr={l_max['inr']:.2f} norepi={l_max['norepinephrine']:.3f}")
        print(f"  lab min: plt={l_min['platelet']:.0f} pH={l_min['ph']:.2f} "
              f"uo={l_min['urine_output']:.1f}")
        flags_on = [k for k, v in meta.items() if k.startswith("flag_") and v]
        if flags_on:
            print(f"  flags: {', '.join(flags_on)}")


def main():
    outputs = generate_all()
    print(f"Wrote {len(outputs)} patients into {OUT_DIR}")
    summarize(outputs)


if __name__ == "__main__":
    main()
