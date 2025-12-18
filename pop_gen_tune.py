import random
import math
import time
import statistics
import argparse
import os
import csv
from typing import List, Tuple, Dict, Set

# =====================================================
# GLOBAL CACHE & AYARLAR
# =====================================================
evaluation_cache = {}


class ULineBalancerGA:
    def __init__(self, instance_path, m_stations, pop_size, gens, crossover_rate=0.2, mutation_rate=0.5, seed=42):
        self.instance_path = instance_path
        self.m_stations = m_stations
        self.pop_size = pop_size
        self.gens = gens
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.seed = seed
        self.n, self.times, self.precedences = self._read_in2(instance_path)
        self.preds, self.succs = self._build_pred_succ()

    def _read_in2(self, path: str) -> Tuple[int, List[int], List[Tuple[int, int]]]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Hata: {path} dosyası bulunamadı!")

        with open(path, "r") as f:
            lines = [line.strip() for line in f if line.strip()]

        n = int(lines[0])
        times = [int(lines[i]) for i in range(1, n + 1)]
        precedences = []
        for line in lines[n + 1:]:
            parts = line.replace(",", " ").split()
            if not parts: continue
            a, b = int(parts[0]), int(parts[1])
            if a == -1: break
            precedences.append((a, b))
        return n, times, precedences

    def _build_pred_succ(self):
        preds = {i: set() for i in range(1, self.n + 1)}
        succs = {i: set() for i in range(1, self.n + 1)}
        for i, j in self.precedences:
            preds[j].add(i)
            succs[i].add(j)
        return preds, succs

    def repair_to_topological(self, perm):
        rank = {task: idx for idx, task in enumerate(perm)}
        remaining = set(perm)
        result, placed = [], set()
        while remaining:
            eligible = [i for i in remaining if self.preds[i].issubset(placed)]
            if not eligible: eligible = list(remaining)
            chosen = min(eligible, key=lambda x: rank[x])
            result.append(chosen)
            placed.add(chosen)
            remaining.remove(chosen)
        return result

    def decode_with_cycle_limit(self, perm, c):
        assigned = set()
        stations_load = [0] * self.m_stations
        assigned_count = 0

        for s_idx in range(self.m_stations):
            progress = True
            while progress and assigned_count < self.n:
                progress = False
                for task in perm:
                    if task in assigned: continue
                    if self.preds[task].issubset(assigned) or self.succs[task].issubset(assigned):
                        t_time = self.times[task - 1]
                        if stations_load[s_idx] + t_time <= c:
                            stations_load[s_idx] += t_time
                            assigned.add(task)
                            assigned_count += 1
                            progress = True
            if assigned_count == self.n: return True
        return False

    def evaluate_cycle_time(self, perm):
        p_tuple = tuple(perm)
        if p_tuple in evaluation_cache: return evaluation_cache[p_tuple]

        lb = max(max(self.times), math.ceil(sum(self.times) / self.m_stations))
        ub = sum(self.times)
        best_c = ub

        while lb <= ub:
            mid = (lb + ub) // 2
            if self.decode_with_cycle_limit(perm, mid):
                best_c = mid
                ub = mid - 1
            else:
                lb = mid + 1

        evaluation_cache[p_tuple] = float(best_c)
        return float(best_c)

    def run(self):
        rng = random.Random(self.seed)
        population = []
        for _ in range(self.pop_size):
            rem, res, asgn = set(range(1, self.n + 1)), [], set()
            while rem:
                elig = [i for i in rem if self.preds[i].issubset(asgn)]
                c = rng.choice(elig)
                res.append(c);
                asgn.add(c);
                rem.remove(c)
            population.append(res)

        best_overall = float('inf')
        for _ in range(self.gens):
            cycles = [self.evaluate_cycle_time(p) for p in population]
            min_c = min(cycles)
            best_overall = min(min_c, best_overall)

            fitness = [1.0 / c for c in cycles]
            new_pop = [population[cycles.index(min_c)]]  # Elitism

            while len(new_pop) < self.pop_size:
                # Tournament Selection
                p_idx = []
                for _ in range(2):
                    i1, i2 = rng.randint(0, self.pop_size - 1), rng.randint(0, self.pop_size - 1)
                    p_idx.append(i1 if fitness[i1] > fitness[i2] else i2)

                p1, p2 = population[p_idx[0]], population[p_idx[1]]

                # POX Crossover
                if rng.random() < self.crossover_rate:
                    subset = {g for g in p1 if rng.random() < 0.5} or {rng.choice(p1)}
                    child = [g for g in p1 if g in subset]
                    child.extend([g for g in p2 if g not in subset])
                else:
                    child = p1[:]

                # Mutation
                if rng.random() < self.mutation_rate:
                    idx1, idx2 = rng.sample(range(self.n), 2)
                    child[idx1], child[idx2] = child[idx2], child[idx1]

                new_pop.append(self.repair_to_topological(child))
            population = new_pop

        return best_overall


# =====================================================
# EXECUTION LAYER (CLI)
# =====================================================

def main():
    parser = argparse.ArgumentParser(description="U-Type Assembly Line Balancing Genetic Algorithm")
    parser.add_argument("--file", type=str, default="ARC83.IN2", help="Input .IN2 file path")
    parser.add_argument("--stations", type=int, default=12, help="Number of stations (m)")
    parser.add_argument("--runs", type=int, default=5, help="Runs per configuration for DOE")
    args = parser.parse_args()

    # Deney Parametreleri
    POP_SIZES = [50, 100]
    GEN_LIST = [250, 500]

    results_log = []
    header = f"{'Pop':<5} | {'Gen':<5} | {'Avg Cycle':<12} | {'Best':<8} | {'Time/Run'}"
    print(f"\nAnaliz Başlatılıyor: {args.file} ({args.stations} İstasyon)")
    print("-" * len(header))
    print(header)

    for p_size in POP_SIZES:
        for g_count in GEN_LIST:
            start_t = time.time()
            config_results = []

            for r in range(args.runs):
                solver = ULineBalancerGA(args.file, args.stations, p_size, g_count, seed=42 + r)
                config_results.append(solver.run())

            duration = (time.time() - start_t) / args.runs
            avg_c = statistics.mean(config_results)
            best_c = min(config_results)

            print(f"{p_size:<5} | {g_count:<5} | {avg_c:<12.2f} | {best_c:<8.0f} | {duration:.2f}s")
            results_log.append([p_size, g_count, avg_c, best_c, round(duration, 4)])

    # CSV Olarak Kaydet (GitHub için raporlama kolaylığı)
    with open("doe_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Pop_Size", "Gens", "Avg_Cycle", "Best_Cycle", "Avg_Time"])
        writer.writerows(results_log)
    print(f"\nSonuçlar 'doe_results.csv' dosyasına kaydedildi.")


if __name__ == "__main__":
    main()