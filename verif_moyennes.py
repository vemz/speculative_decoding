DOMAINES = {"maths": ["GSM8K", "MATH-500", "AIME25"],
            "code": ["HumanEval", "MBPP", "LiveCodeBench"],
            "chat": ["MT-Bench", "Alpaca", "Arena-Hard"]}

N = None
PAPIER = {   # benchmark : (gamma, speedup)
    "EAGLE-3": {b: (N, N) for d in DOMAINES.values() for b in d},
    "DFlash": {"GSM8K": (7.43, 5.77), "MATH-500": (N, N), "AIME25": (N, N),
               "HumanEval": (5.74, 4.50), "MBPP": (N, N), "LiveCodeBench": (N, N),
               "MT-Bench": (3.14, 2.48), "Alpaca": (N, N), "Arena-Hard": (N, N)},
    "FlashMap 1 pas": {"GSM8K": (7.69, 6.14), "MATH-500": (N, N), "AIME25": (N, N),
                       "HumanEval": (5.82, 4.73), "MBPP": (N, N), "LiveCodeBench": (N, N),
                       "MT-Bench": (3.34, 2.73), "Alpaca": (N, N), "Arena-Hard": (N, N)},
    "FlashMap 2 pas": {"GSM8K": (8.31, 5.89), "MATH-500": (7.69, N), "AIME25": (6.36, N),
                       "HumanEval": (6.26, 4.53), "MBPP": (5.24, N), "LiveCodeBench": (7.19, N),
                       "MT-Bench": (3.40, 2.47), "Alpaca": (3.20, N), "Arena-Hard": (3.80, N)},
}

ECHELLE = {"gamma": 2.6 / 8, "speedup": 2.6 / 6}
BARRES = {   # méthode : {domaine : (hauteur gamma, hauteur speedup)}
    "EAGLE-3":        {"maths": (2.074, 1.317), "code": (1.905, 1.205), "chat": (1.238, 0.793)},
    "DFlash":         {"maths": (2.213, 2.314), "code": (1.827, 1.894), "chat": (1.027, 1.083)},
    "FlashMap 1 pas": {"maths": (2.269, 2.444), "code": (1.888, 2.024), "chat": (1.105, 1.205)},
    "FlashMap 2 pas": {"maths": (2.421, 2.323), "code": (2.025, 1.933), "chat": (1.128, 1.092)},
}
SLIDE_GAINS = {"FlashMap 1 pas": (4, 7), "FlashMap 2 pas": (10, 1)} 

def moyenne(xs):
    return sum(xs) / len(xs) if xs and None not in xs else None


manquants = [(m, b) for m, t in PAPIER.items() for b, v in t.items() if None in v]
if manquants:
    for m, b in manquants:
        print(f"  {m:15s} {b}")
    print()

for m, t in PAPIER.items():
    for d, bs in DOMAINES.items():
        for k, nom in enumerate(("gamma", "speedup")):
            calc = moyenne([t[b][k] for b in bs])
            slide = BARRES[m][d][k] / ECHELLE[nom]
            ok = "" if calc is None else ("ok" if abs(calc - slide) < 0.02 else "ÉCART")
            c = "?" if calc is None else f"{calc:.2f}"
            print(f"  {m:15s} {d:6s} {nom:8s} {c:>5s} | {slide:.2f}  {ok}")

for m, (sg, ss) in SLIDE_GAINS.items():
    for k, (nom, s) in enumerate((("gamma", sg), ("speedup", ss))):
        r = [PAPIER[m][b][k] / PAPIER["DFlash"][b][k] if None not in (PAPIER[m][b][k], PAPIER["DFlash"][b][k]) else None
             for bs in DOMAINES.values() for b in bs]
        g = moyenne(r)
        c = "?" if g is None else f"{100 * (g - 1):+.0f} %"
        print(f"  {m:15s} {nom:8s} {c:>6s} | {s:+d} %")
