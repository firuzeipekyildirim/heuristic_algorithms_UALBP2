import random
import math
import time
from typing import List, Tuple, Dict, Set
import csv
import os
from datetime import datetime

# =====================================================
# KULLANICI AYARLARI
# =====================================================
INSTANCE_PATH     = "ARC111.IN2"   # IN2 dosyanın yolu
M_STATIONS        = 12            # istasyon sayısı
POP_SIZE          = 40
GENERATIONS       = 300 #gorsellestirme
CROSSOVER_RATE    = 0.7
MUTATION_RATE     = 0.08
RUNS              = 10         # GA kaç kez çalışacak
BASE_SEED         = 0

OPTIMAL_CYCLE     = 12534         # Excel'den bildiğimiz optimum (Arcus1, m=12)

# =====================================================
# POX DEBUG AYARLARI
# =====================================================
DEBUG_POX         = False         # POX örneği görmek istemezsen False yap
POX_DEBUG_LIMIT   = 5            # En fazla kaç crossover örneği yazdırılsın
pox_debug_counter = 0


# =====================================================
# 1) IN2 DOSYASINI OKUMA
# =====================================================

def read_in2(path: str) -> Tuple[int, List[int], List[Tuple[int, int]]]:
    with open(path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    n = int(lines[0])

    times = []
    for i in range(1, n + 1):
        times.append(int(lines[i]))

    precedences = []
    for line in lines[n + 1:]:
        if "," in line:
            a, b = line.split(",")
        else:
            a, b = line.split()
        i = int(a)
        j = int(b)
        if i == -1 and j == -1:
            break
        precedences.append((i, j))

    return n, times, precedences


# =====================================================
# 2) ÖNCÜL / ARDIL KÜMELERİ
# =====================================================

def build_pred_succ(
    n: int,
    precedences: List[Tuple[int, int]]
) -> Tuple[Dict[int, Set[int]], Dict[int, Set[int]]]:
    preds = {i: set() for i in range(1, n + 1)}
    succs = {i: set() for i in range(1, n + 1)}

    for i, j in precedences:
        preds[j].add(i)
        succs[i].add(j)

    return preds, succs


# =====================================================
# 3) TOPOLOJİK SIRALAMA + REPAIR
# =====================================================

def random_topological_sort(
    n: int,
    preds: Dict[int, Set[int]],
    rng: random.Random
) -> List[int]:
    remaining = set(range(1, n + 1))
    result = []
    assigned = set()

    while remaining:
        eligible = [i for i in remaining if preds[i].issubset(assigned)]
        if not eligible:
            raise RuntimeError("Precedence graph contains a cycle?")
        chosen = rng.choice(eligible)
        result.append(chosen)
        assigned.add(chosen)
        remaining.remove(chosen)

    return result


def repair_to_topological(
    perm: List[int],
    preds: Dict[int, Set[int]]
) -> List[int]:
    rank = {task: idx for idx, task in enumerate(perm)}
    remaining = set(perm)
    result = []
    placed = set()

    while remaining:
        eligible = [i for i in remaining if preds[i].issubset(placed)]
        if not eligible:
            eligible = list(remaining)
        chosen = min(eligible, key=lambda x: rank[x])
        result.append(chosen)
        placed.add(chosen)
        remaining.remove(chosen)

    return result


# =====================================================
# 4) U-SHAPED DECODE + CYCLE TIME
# =====================================================

def is_task_eligible(
    task: int,
    assigned: Set[int],
    preds: Dict[int, Set[int]],
    succs: Dict[int, Set[int]]
) -> bool:
    return preds[task].issubset(assigned) or succs[task].issubset(assigned)


def decode_with_cycle_limit(
    perm: List[int],
    times: List[int],
    preds: Dict[int, Set[int]],
    succs: Dict[int, Set[int]],
    m: int,
    c: int
) -> Tuple[bool, List[List[int]], List[int]]:
    n = len(perm)
    assigned: Set[int] = set()
    stations: List[List[int]] = [[] for _ in range(m)]
    loads: List[int] = [0 for _ in range(m)]

    station_idx = 0

    while station_idx < m and len(assigned) < n:
        progress = True
        while progress and len(assigned) < n:
            progress = False
            for task in perm:
                if task in assigned:
                    continue
                if not is_task_eligible(task, assigned, preds, succs):
                    continue

                duration = times[task - 1]
                if loads[station_idx] + duration <= c:
                    stations[station_idx].append(task)
                    loads[station_idx] += duration
                    assigned.add(task)
                    progress = True
        station_idx += 1

    feasible = (len(assigned) == n)
    return feasible, stations, loads


def evaluate_permutation_cycle_time(
    perm: List[int],
    times: List[int],
    preds: Dict[int, Set[int]],
    succs: Dict[int, Set[int]],
    m: int
) -> Tuple[float, List[List[int]], List[int]]:
    total_time = sum(times)
    max_time = max(times)

    lb = max(max_time, math.ceil(total_time / m))
    ub = total_time

    best_c = None
    best_stations = None
    best_loads = None

    while lb <= ub:
        mid = (lb + ub) // 2
        feasible, stations, loads = decode_with_cycle_limit(
            perm, times, preds, succs, m, mid
        )
        if feasible:
            best_c = mid
            best_stations = stations
            best_loads = loads
            ub = mid - 1
        else:
            lb = mid + 1

    if best_c is None:
        return float("inf"), [[] for _ in range(m)], [0 for _ in range(m)]

    return float(best_c), best_stations, best_loads


# =====================================================
# 5) GA OPERATÖRLERİ
# =====================================================

def tournament_selection(
    population: List[List[int]],
    fitness: List[float],
    rng: random.Random,
    k: int = 2
) -> List[int]:
    n = len(population)
    best_idx = None
    best_fit = -float("inf")
    for _ in range(k):
        i = rng.randrange(n)
        if fitness[i] > best_fit:
            best_fit = fitness[i]
            best_idx = i
    return population[best_idx][:]


def pox_crossover(
    p1: List[int],
    p2: List[int],
    rng: random.Random
) -> List[int]:
    """
    POX crossover + isteğe bağlı debug çıktısı.
    """
    global pox_debug_counter

    chosen = set()
    for gene in p1:
        if rng.random() < 0.5:
            chosen.add(gene)
    if not chosen:
        chosen.add(rng.choice(p1))

    child = []
    for gene in p1:
        if gene in chosen:
            child.append(gene)
    for gene in p2:
        if gene not in chosen:
            child.append(gene)

    # ---- Debug output ----
    if DEBUG_POX and pox_debug_counter < POX_DEBUG_LIMIT:
        pox_debug_counter += 1
        print("\n=== POX CROSSOVER SAMPLE ===")
        print("Parent1 (first 15):", p1[:15])
        print("Parent2 (first 15):", p2[:15])
        print("Chosen subset:", sorted(chosen))
        print("Child   (first 15):", child[:15])
        print("============================\n")

    return child


def swap_mutation(perm: List[int], rng: random.Random) -> List[int]:
    n = len(perm)
    i, j = rng.sample(range(n), 2)
    perm[i], perm[j] = perm[j], perm[i]
    return perm


# =====================================================
# 6) GA: TEK RUN + İSTATİSTİK
# =====================================================

def genetic_algorithm_ualbp2(
    n: int,
    times: List[int],
    preds: Dict[int, Set[int]],
    succs: Dict[int, Set[int]],
    m: int,
    pop_size: int,
    generations: int,
    crossover_rate: float,
    mutation_rate: float,
    seed: int,
    run_index: int
):
    rng = random.Random(seed)

    population: List[List[int]] = []
    for _ in range(pop_size):
        perm = random_topological_sort(n, preds, rng)
        population.append(perm)

    best_perm = None
    best_cycle = float("inf")
    best_stations = None
    best_loads = None

    for gen in range(generations):
        fitness_vals = []
        eval_cache = {}

        for perm in population:
            key = tuple(perm)
            if key in eval_cache:
                cycle_time = eval_cache[key]
            else:
                cycle_time, stations, loads = evaluate_permutation_cycle_time(
                    perm, times, preds, succs, m
                )
                eval_cache[key] = cycle_time
                if cycle_time < best_cycle:
                    best_cycle = cycle_time
                    best_perm = perm[:]
                    best_stations = stations
                    best_loads = loads

            if math.isinf(eval_cache[key]):
                fit = 0.0
            else:
                fit = 1.0 / eval_cache[key]
            fitness_vals.append(fit)

        new_population: List[List[int]] = []
        if best_perm is not None:
            new_population.append(best_perm[:])

        while len(new_population) < pop_size:
            p1 = tournament_selection(population, fitness_vals, rng, k=2)
            p2 = tournament_selection(population, fitness_vals, rng, k=2)

            if rng.random() < crossover_rate:
                child = pox_crossover(p1, p2, rng)
            else:
                child = p1[:]

            if rng.random() < mutation_rate:
                child = swap_mutation(child, rng)

            child = repair_to_topological(child, preds)
            new_population.append(child)

        population = new_population

    # ----- RUN SONU: final popülasyon istatistikleri -----
    final_stats = []   # (fitness, cycle, perm)
    cycles = []
    fits = []

    for perm in population:
        cycle_time, _, _ = evaluate_permutation_cycle_time(
            perm, times, preds, succs, m
        )
        fit = 0.0 if math.isinf(cycle_time) else 1.0 / cycle_time
        cycles.append(cycle_time)
        fits.append(fit)
        final_stats.append((fit, cycle_time, perm))

    best_cycle_final = min(cycles)
    worst_cycle_final = max(cycles)
    mean_cycle_final = sum(cycles) / len(cycles)

    best_fit_final = max(fits)
    worst_fit_final = min(fits)
    mean_fit_final = sum(fits) / len(fits)

    final_stats.sort(key=lambda x: x[0], reverse=True)
    top10 = final_stats[:10]

    # ek bilgi için döndür
    run_stats = {
        "best_cycle": best_cycle,
        "final_best_cycle": best_cycle_final,
        "final_mean_cycle": mean_cycle_final,
        "final_worst_cycle": worst_cycle_final,
        "final_best_fit": best_fit_final,
        "final_mean_fit": mean_fit_final,
        "final_worst_fit": worst_fit_final,
        "top10": top10
    }

    return best_perm, best_cycle, best_stations, best_loads, run_stats


# =====================================================
# 7) ÇOKLU RUN + ÖZET
# =====================================================

def main():
    start_time = time.time()

    # veri oku
    n, times, precedences = read_in2(INSTANCE_PATH)
    preds, succs = build_pred_succ(n, precedences)

    total_time = sum(times)
    max_time = max(times)
    lb1 = math.ceil(total_time / M_STATIONS)
    lb2 = max_time
    lb = max(lb1, lb2)

    print(f"Instance: {INSTANCE_PATH}")
    print(f"Tasks (n)              : {n}")
    print(f"Stations (m)           : {M_STATIONS}")
    print(f"Total work content sum : {total_time}")
    print(f"Max task time          : {max_time}")
    print(f"Lower bound LB1=ceil(sum/m) = {lb1}")
    print(f"Lower bound LB2=max(t_i)    = {lb2}")
    print(f"Overall theoretical LB       = {lb}")
    print()

    run_best_cycles = []
    run_runtimes = []
    best_overall_cycle = float("inf")
    best_overall_solution = None
    best_overall_top10 = None

    for r in range(1, RUNS + 1):
        seed = BASE_SEED + r
        print(f"\n============================")
        print(f"        RUN {r}/{RUNS}")
        print(f"============================")
        run_start = time.time()

        best_perm, best_cycle, best_stations, best_loads, stats = genetic_algorithm_ualbp2(
            n=n,
            times=times,
            preds=preds,
            succs=succs,
            m=M_STATIONS,
            pop_size=POP_SIZE,
            generations=GENERATIONS,
            crossover_rate=CROSSOVER_RATE,
            mutation_rate=MUTATION_RATE,
            seed=seed,
            run_index=r
        )

        run_end = time.time()                 # bu run'ın bitiş zamanı
        run_runtime = run_end - run_start     # süre (saniye)

        run_best_cycles.append(best_cycle)
        run_runtimes.append(run_runtime)

        print(f"\nRun {r} summary:")
        print(f"  Best cycle time over generations      : {best_cycle}")
        print(f"  Final population best / mean / worst c: "
              f"{stats['final_best_cycle']} / {stats['final_mean_cycle']:.2f} / {stats['final_worst_cycle']}")
        print(f"  Final population best / mean / worst fitness: "
              f"{stats['final_best_fit']:.6f} / {stats['final_mean_fit']:.6f} / {stats['final_worst_fit']:.6f}")

        print("\n  Top 10 individuals in final population:")
        for rank, (fit, cyc, perm) in enumerate(stats["top10"], start=1):
            print(f"    {rank:2d}) fitness={fit:.6f}, cycle={cyc}")
            # İlgili istasyonlara dağılımı yazdır
            feasible, stations, loads = decode_with_cycle_limit(
                perm, times, preds, succs, M_STATIONS, int(cyc)
            )
            for s_idx, tasks in enumerate(stations, start=1):
                print(f"        Station {s_idx}: {tasks}")

        if best_cycle < best_overall_cycle:
            best_overall_cycle = best_cycle
            best_overall_solution = (best_perm, best_stations, best_loads)
            best_overall_top10 = stats["top10"]

    # 10 RUN ÖZET İSTATİSTİKLER
    mean_cycle_runs = sum(run_best_cycles) / len(run_best_cycles)
    worst_cycle_runs = max(run_best_cycles)

    best_gap_abs = best_overall_cycle - OPTIMAL_CYCLE
    best_gap_pct = 100.0 * best_gap_abs / OPTIMAL_CYCLE

    mean_gap_abs = mean_cycle_runs - OPTIMAL_CYCLE
    mean_gap_pct = 100.0 * mean_gap_abs / OPTIMAL_CYCLE

    total_runtime = time.time() - start_time

    print("\n====================================")
    print("        10 RUN GLOBAL SUMMARY")
    print("====================================")
    print(f"GA parameters:")
    print(f"  Population size   : {POP_SIZE}")
    print(f"  Generations       : {GENERATIONS}")
    print(f"  Crossover rate    : {CROSSOVER_RATE}")
    print(f"  Mutation rate     : {MUTATION_RATE}")
    print(f"  Parent selection  : Tournament (k=2)")
    print(f"  Crossover method  : POX-benzeri precedence preserving")
    print(f"  Mutation method   : Swap mutation")
    print(f"  Iteration number  : {GENERATIONS} generations per run")
    print(f"  Total runs        : {RUNS}")
    print()

    print(f"Run best cycle times: {run_best_cycles}")
    print(f"Best  cycle over runs : {best_overall_cycle}")
    print(f"Mean  cycle over runs : {mean_cycle_runs:.2f}")
    print(f"Worst cycle over runs : {worst_cycle_runs}")
    print()

    print(f"Known optimal C*      : {OPTIMAL_CYCLE}")
    print(f"Best-of-10 gap        : {best_gap_abs}  ({best_gap_pct:.3f} % above optimum)")
    print(f"Mean-of-10 gap        : {mean_gap_abs:.2f}  ({mean_gap_pct:.3f} % above optimum)")
    print()

    # En iyi çözümün istasyon yükleri ve görevleri
    best_perm, best_stations, best_loads = best_overall_solution
    print("Best-of-10 solution station loads:")
    for i, load in enumerate(best_loads, start=1):
        print(f"  Station {i}: load = {load}")

    print("\nBest-of-10 solution assignments:")
    for i, tasks in enumerate(best_stations, start=1):
        print(f"  Station {i}: {tasks}")

    print("\nTop 10 individuals of best run (by fitness):")
    for rank, (fit, cyc, perm) in enumerate(best_overall_top10, start=1):
        print(f"  {rank:2d}) fitness={fit:.6f}, cycle={cyc}")
        feasible, stations, loads = decode_with_cycle_limit(
            perm, times, preds, succs, M_STATIONS, int(cyc)
        )
        for s_idx, tasks in enumerate(stations, start=1):
            print(f"        Station {s_idx}: {tasks}")

    print(f"\nTotal runtime for {RUNS} runs: {total_runtime:.2f} seconds")

    # =====================================================
    # CSV OUTPUT: instance + timestamp ile isimlendirilmiş dosyalar
    # =====================================================

    instance_name = os.path.splitext(os.path.basename(INSTANCE_PATH))[0]

    # Tarih-saat etiketi (YYYY-MM-DD_HH-MM)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")

    # Otomatik dosya adları
    summary_file      = f"{instance_name}_summary_{timestamp}.csv"
    runs_file         = f"{instance_name}_run_details_{timestamp}.csv"
    best_details_file = f"{instance_name}_best_details_{timestamp}.csv"
    top10_file        = f"{instance_name}-top-10-individuals-{timestamp}.csv"

    print("\nGenerated output filenames:")
    print("  Summary file      :", summary_file)
    print("  Run details file  :", runs_file)
    print("  Best details file :", best_details_file)
    print("  Top10 file        :", top10_file)
    print()

    # =====================================================
    # CSV OUTPUT 1: genel özet
    # =====================================================
    with open(summary_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow([
            "Instance",
            "Stations(m)",
            "TotalWork(sum_ti)",
            "Population",
            "Generations",
            "CrossoverRate",
            "MutationRate",
            "ParentSelection",
            "CrossoverMethod",
            "MutationMethod",
            "Runs",
            "BestCycle",
            "MeanCycle",
            "WorstCycle",
            "OptimalC",
            "BestGapAbs",
            "BestGapPct",
            "MeanGapAbs",
            "MeanGapPct",
            "TotalRuntime(sec)"
        ])

        writer.writerow([
            instance_name,
            M_STATIONS,
            total_time,
            POP_SIZE,
            GENERATIONS,
            CROSSOVER_RATE,
            MUTATION_RATE,
            "Tournament(k=2)",
            "POX",
            "Swap",
            RUNS,
            best_overall_cycle,
            f"{mean_cycle_runs:.2f}",
            worst_cycle_runs,
            OPTIMAL_CYCLE,
            best_gap_abs,
            f"{best_gap_pct:.3f}",
            f"{mean_gap_abs:.2f}",
            f"{mean_gap_pct:.3f}",
            f"{total_runtime:.2f}"
        ])

    print(f"CSV summary saved to: {summary_file}\n")

    # =====================================================
    # CSV OUTPUT 2: run bazında cycle & runtime
    # =====================================================
    with open(runs_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow([
            "Run",
            "BestCycle",
            "RunRuntime(sec)"
        ])

        for r in range(len(run_best_cycles)):
            cyc = run_best_cycles[r]
            rt  = run_runtimes[r]
            writer.writerow([r + 1, cyc, f"{rt:.2f}"])

    print(f"Run details saved to: {runs_file}")

    # =====================================================
    # CSV OUTPUT 3: en iyi çözümün istasyon detayı
    # =====================================================

    total_line_eff = 100.0 * total_time / (M_STATIONS * best_overall_cycle)

    with open(best_details_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow([
            "Instance",
            "Stations(m)",
            "BestCycle",
            "TotalWork(sum_ti)",
            "GlobalLineEfficiency(%)",
            "StationIndex",
            "StationLoad",
            "StationUtilization(Load/BestCycle)",
            "TasksSequence"
        ])

        for s_idx, (load, tasks) in enumerate(zip(best_loads, best_stations), start=1):
            utilization = load / best_overall_cycle
            tasks_str = " ".join(str(t) for t in tasks)

            writer.writerow([
                instance_name,
                M_STATIONS,
                best_overall_cycle,
                total_time,
                f"{total_line_eff:.2f}",
                s_idx,
                load,
                f"{utilization:.4f}",
                tasks_str
            ])

    print(f"Best solution details saved to: {best_details_file}")

    # =====================================================
    # CSV OUTPUT 4: Top 10 individuals of best run (her biri için istasyonlar)
    # =====================================================
    with open(top10_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow([
            "Instance",
            "Stations(m)",
            "RankInBestRun",
            "Fitness",
            "Cycle",
            "StationIndex",
            "StationLoad",
            "StationUtilization(Load/Cycle)",
            "TasksSequence"
        ])

        for rank, (fit, cyc, perm) in enumerate(best_overall_top10, start=1):
            feasible, stations, loads = decode_with_cycle_limit(
                perm, times, preds, succs, M_STATIONS, int(cyc)
            )
            for s_idx, (load, tasks) in enumerate(zip(loads, stations), start=1):
                tasks_str = " ".join(str(t) for t in tasks)
                utilization = 0.0 if cyc == 0 else load / cyc

                writer.writerow([
                    instance_name,
                    M_STATIONS,
                    rank,
                    f"{fit:.6f}",
                    cyc,
                    s_idx,
                    load,
                    f"{utilization:.4f}",
                    tasks_str
                ])

    print(f"Top 10 individuals (best run) saved to: {top10_file}")


if __name__ == "__main__":
    main()
