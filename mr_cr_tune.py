import random
import math
import time
import sys
import csv
import os
import platform
import argparse
from typing import List, Tuple, Dict, Set
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed


# =====================================================
# SİSTEM AYARLARI
# =====================================================
def prevent_sleep():
    """İşlem sırasında bilgisayarın uyumasını engeller."""
    try:
        system = platform.system()
        if system == "Windows":
            import ctypes
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000001 | 0x00000040)
        elif system == "Darwin":
            os.system('caffeinate -i &')
    except Exception as e:
        print(f"Uyarı: Uyku engelleme aktif edilemedi: {e}")


# =====================================================
# UALBP ÇÖZÜCÜ SINIFI
# =====================================================
class UALBPSolver:
    def __init__(self, instance_path: str, m_stations: int):
        self.path = instance_path
        self.m = m_stations
        self.n, self.times, self.precedences = self._read_in2()
        self.preds, self.succs = self._build_graphs()

    def _read_in2(self):
        if not os.path.exists(self.path):
            print(f"Hata: {self.path} dosyası bulunamadı!")
            print("Lütfen .IN2 dosyasını aynı klasöre koyun veya --path ile yolu belirtin.")
            sys.exit(1)

        with open(self.path, "r") as f:
            lines = [line.strip() for line in f if line.strip()]

        n = int(lines[0])
        times = [int(lines[i]) for i in range(1, n + 1)]
        precedences = []
        for line in lines[n + 1:]:
            parts = line.replace(",", " ").split()
            if len(parts) < 2: continue
            i, j = int(parts[0]), int(parts[1])
            if i == -1 and j == -1: break
            precedences.append((i, j))
        return n, times, precedences

    def _build_graphs(self):
        preds = {i: set() for i in range(1, self.n + 1)}
        succs = {i: set() for i in range(1, self.n + 1)}
        for i, j in self.precedences:
            preds[j].add(i)
            succs[i].add(j)
        return preds, succs

    @staticmethod
    def decode_u_shape(perm, times, preds, succs, m, c):
        assigned, n = set(), len(perm)
        loads = [0] * m

        for s_idx in range(m):
            progress = True
            while progress and len(assigned) < n:
                progress = False
                for task in perm:
                    if task in assigned: continue
                    # U-Tipi kuralı: Önceller VEYA ardıllar atanmış olmalı
                    if preds[task].issubset(assigned) or succs[task].issubset(assigned):
                        if loads[s_idx] + times[task - 1] <= c:
                            loads[s_idx] += times[task - 1]
                            assigned.add(task)
                            progress = True
            if len(assigned) == n: return True
        return False


# =====================================================
# GENETİK ALGORİTMA FONKSİYONLARI
# =====================================================

def evaluate_cycle_time(perm, solver, c_min, c_max):
    lb, ub = c_min, c_max
    best_c = c_max
    while lb <= ub:
        mid = (lb + ub) // 2
        if solver.decode_u_shape(perm, solver.times, solver.preds, solver.succs, solver.m, mid):
            best_c, ub = mid, mid - 1
        else:
            lb = mid + 1
    return best_c


def run_ga_instance(args):
    """Paralel çalıştırma için sarmalayıcı fonksiyon."""
    n, times, preds, succs, m, pop_size, gens, c_rate, m_rate, seed = args
    rng = random.Random(seed)

    # Yardımcı nesne oluştur (decode metodu için)
    c_min = max(max(times), math.ceil(sum(times) / m))
    c_max = sum(times)

    # Başlangıç Popülasyonu (Topolojik)
    population = []
    for _ in range(pop_size):
        rem, res, asgn = set(range(1, n + 1)), [], set()
        while rem:
            elig = [i for i in rem if preds[i].issubset(asgn)]
            c = rng.choice(elig)
            res.append(c);
            asgn.add(c);
            rem.remove(c)
        population.append(res)

    best_cycle = float('inf')
    best_p = None

    for _ in range(gens):
        cycles = [evaluate_cycle_time(p, UALBPSolver, c_min, c_max) for p in population]

        for i, c in enumerate(cycles):
            if c < best_cycle:
                best_cycle = c
                best_p = population[i][:]

        # Seçim ve Üretim
        fitness = [1.0 / c for c in cycles]
        new_pop = [best_p]  # Elitizm

        while len(new_pop) < pop_size:
            # Turnuva
            idx1, idx2 = rng.sample(range(pop_size), 2)
            p1 = population[idx1] if fitness[idx1] > fitness[idx2] else population[idx2]
            idx3, idx4 = rng.sample(range(pop_size), 2)
            p2 = population[idx3] if fitness[idx3] > fitness[idx4] else population[idx4]

            # Crossover (POX)
            if rng.random() < c_rate:
                subset = {g for g in p1 if rng.random() < 0.5} or {rng.choice(p1)}
                child = [g for g in p1 if g in subset]
                child.extend([g for g in p2 if g not in subset])
            else:
                child = p1[:]

            # Mutation (Swap)
            if rng.random() < m_rate:
                i, j = rng.sample(range(n), 2)
                child[i], child[j] = child[j], child[i]

            # Repair
            rank = {t: idx for idx, t in enumerate(child)}
            rem, res, placed = set(child), [], set()
            while rem:
                elig = [t for t in rem if preds[t].issubset(placed)]
                if not elig: elig = list(rem)
                chosen = min(elig, key=lambda x: rank[x])
                res.append(chosen);
                placed.add(chosen);
                rem.remove(chosen)
            new_pop.append(res)

        population = new_pop

    return best_cycle


# =====================================================
# ANA ÇALIŞTIRICI
# =====================================================
def main():
    prevent_sleep()

    parser = argparse.ArgumentParser(description="UALBP-GA Hyperparameter Tuning")
    parser.add_argument("--path", type=str, default="ARC83.IN2", help="Veri seti yolu")
    parser.add_argument("--stations", type=int, default=12, help="İstasyon sayısı")
    parser.add_argument("--runs", type=int, default=5, help="DOE için tekrar sayısı")
    args = parser.parse_args()

    solver = UALBPSolver(args.path, args.stations)

    # Parametre Grid (Örnek Set)
    MUT_VALUES = [0.1, 0.3, 0.5]
    CROS_VALUES = [0.4, 0.6, 0.8]

    results_file = f"results_tuning_{datetime.now().strftime('%m%d_%H%M')}.csv"

    print(f"\n{'=' * 50}")
    print(f"UALBP GENETİK ALGORİTMA OPTİMİZASYONU")
    print(f"{'=' * 50}")
    print(f"Veri Seti: {args.path} | İş Sayısı: {solver.n}")
    print(f"Hücre Sayısı (m): {args.stations}")
    print(f"Kombinasyonlar: {len(MUT_VALUES) * len(CROS_VALUES)}")
    print(f"{'=' * 50}\n")

    with open(results_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Mutation", "Crossover", "Avg_Cycle", "Min_Cycle", "Time_Sec"])

        for m_rate in MUT_VALUES:
            for c_rate in CROS_VALUES:
                start_t = time.time()

                # Paralel İş Yükü Hazırlama
                ga_tasks = [
                    (solver.n, solver.times, solver.preds, solver.succs, solver.m,
                     50, 200, c_rate, m_rate, 42 + r)
                    for r in range(args.runs)
                ]

                with ProcessPoolExecutor() as executor:
                    run_cycles = list(executor.map(run_ga_instance, ga_tasks))

                avg_c = sum(run_cycles) / args.runs
                min_c = min(run_cycles)
                elapsed = time.time() - start_t

                print(f"Mut: {m_rate:.2f} | Cros: {c_rate:.2f} -> Ort: {avg_c:.1f} | En İyi: {min_c} | {elapsed:.1f}s")
                writer.writerow([m_rate, c_rate, avg_c, min_c, f"{elapsed:.2f}"])
                f.flush()

    print(f"\nSüreç Tamamlandı. Çıktı: {results_file}")


if __name__ == "__main__":
    main()