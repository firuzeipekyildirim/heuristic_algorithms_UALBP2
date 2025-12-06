import csv

# genetic.py ile aynı klasörde olduğumuzu varsayıyoruz
from genetic import (
    read_in2,
    build_pred_succ,
    genetic_algorithm_ualbp2,
    INSTANCE_PATH,
    M_STATIONS,
    POP_SIZE,
    GENERATIONS,
    RUNS,
    BASE_SEED,
)

def run_experiment(mut_rate: float, cross_rate: float) -> float:
    """
    Verilen mutasyon ve crossover oranı için,
    RUNS defa GA çalıştırır ve bu 10 run içindeki
    EN İYİ cycle time’ı döndürür.
    """
    # Instance verisini bir kere yükle
    n, times, precedences = read_in2(INSTANCE_PATH)
    preds, succs = build_pred_succ(n, precedences)

    best_over_runs = float("inf")

    for r in range(1, RUNS + 1):
        seed = BASE_SEED + r

        best_perm, best_cycle, best_stations, best_loads, stats = genetic_algorithm_ualbp2(
            n=n,
            times=times,
            preds=preds,
            succs=succs,
            m=M_STATIONS,
            pop_size=POP_SIZE,
            generations=GENERATIONS,
            crossover_rate=cross_rate,
            mutation_rate=mut_rate,
            seed=seed,
            run_index=r
        )

        if best_cycle < best_over_runs:
            best_over_runs = best_cycle

    return best_over_runs


def main():
    # farklı değerleri dene
    mut_values = [0.01, 0.05, 0.10, 0.13]
    cros_values = [0.6, 0.7, 0.8, 0.95]

    out_file = "GA_param_tuning_cycles.csv"

    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # başlık
        writer.writerow(["deneme", "mut", "cros", "best_cycle_over_10_runs"])

        deneme_no = 1
        for mut in mut_values:
            for cros in cros_values:
                print(f"\nDeneme {deneme_no}: mut={mut}, cros={cros} için GA çalışıyor...")

                best_cycle = run_experiment(mut_rate=mut, cross_rate=cros)

                print(f"  -> 2 run içindeki en iyi cycle: {best_cycle}")

                writer.writerow([deneme_no, mut, cros, best_cycle])
                deneme_no += 1

    print(f"\nTuning tamamlandı. Sonuçlar: {out_file}")


if __name__ == "__main__":
    main()
