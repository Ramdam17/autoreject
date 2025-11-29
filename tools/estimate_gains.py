#!/usr/bin/env python
"""
Estimation realiste du gain pour les donnees utilisateur.
"""

# Vos temps reels
time_ica = 275.3  # minutes
time_final = 630.5  # minutes

print("=" * 70)
print("ESTIMATION REALISTE DU GAIN POUR VOS DONNEES")
print("=" * 70)
print()
print("Vos temps actuels:")
print(f"  AutoReject ICA:   {time_ica:.0f} min ({time_ica/60:.1f}h)")
print(f"  AutoReject Final: {time_final:.0f} min ({time_final/60:.1f}h)")
print(f"  Total:            {time_ica + time_final:.0f} min ({(time_ica + time_final)/60:.1f}h)")
print()

# Decomposition estimee du temps AutoReject:
# Basee sur le profiling typique de AutoReject.fit()
# 
# Pour chaque combinaison (n_interpolate, consensus, cv_fold):
#   1. Interpolation des mauvaises epoques (~60% du temps)
#   2. Calcul des seuils ptp + vote (~25% du temps) 
#   3. RANSAC si utilise (~10% du temps)
#   4. Overhead CV, copies, etc (~5% du temps)

print("Decomposition estimee du temps:")
print("  - Interpolation MNE: 60%")
print("  - Peak-to-peak + vote: 25%")
print("  - RANSAC (si utilise): 10%")
print("  - Overhead: 5%")
print()

# Gains mesures par composant
interp_speedup = 4.4  # n_jobs=-1 sur M3 Pro 16 cores
ptp_speedup = 0.4     # GPU plus lent pour ptp
median_speedup = 7.0  # GPU beaucoup plus rapide pour median
ransac_speedup = median_speedup  # RANSAC utilise principalement median

print("Gains mesures (benchmarks):")
print(f"  - Interpolation (n_jobs=-1): {interp_speedup:.1f}x")
print(f"  - Peak-to-peak (GPU): {ptp_speedup:.1f}x (GPU plus lent car overhead transfert)")
print(f"  - Median/RANSAC (GPU): {median_speedup:.1f}x")
print()

# Scenario 1: Parallelisation CPU seulement (n_jobs=-1)
# Seule l'interpolation est acceleree
speedup_cpu = 1 / (0.60 / interp_speedup + 0.25 + 0.10 + 0.05)
new_time_ica_cpu = time_ica / speedup_cpu
new_time_final_cpu = time_final / speedup_cpu

print("SCENARIO 1: CPU multicore seulement (n_jobs=-1)")
print("=" * 50)
print(f"  Gain global: {speedup_cpu:.1f}x")
print(f"  AutoReject ICA:   {time_ica:.0f} min -> {new_time_ica_cpu:.0f} min ({new_time_ica_cpu/60:.1f}h)")
print(f"  AutoReject Final: {time_final:.0f} min -> {new_time_final_cpu:.0f} min ({new_time_final_cpu/60:.1f}h)")
savings_cpu = (time_ica + time_final - new_time_ica_cpu - new_time_final_cpu)
print(f"  Economie: {savings_cpu:.0f} min ({savings_cpu/60:.1f}h)")
print()

# Scenario 2: GPU seulement (AUTOREJECT_BACKEND=torch)
# ptp plus lent, RANSAC plus rapide
speedup_gpu = 1 / (0.60 + 0.25 / ptp_speedup + 0.10 / ransac_speedup + 0.05)
new_time_ica_gpu = time_ica / speedup_gpu
new_time_final_gpu = time_final / speedup_gpu

print("SCENARIO 2: GPU seulement (AUTOREJECT_BACKEND=torch)")
print("=" * 50)
print(f"  Gain global: {speedup_gpu:.1f}x (PLUS LENT)")
print(f"  AutoReject ICA:   {time_ica:.0f} min -> {new_time_ica_gpu:.0f} min")
print(f"  AutoReject Final: {time_final:.0f} min -> {new_time_final_gpu:.0f} min")
print("  NOTE: Plus lent car ptp sur GPU est 2.5x plus lent que CPU")
print()

# Scenario 3: CPU multicore + GPU intelligent
# Utiliser NumPy pour ptp, GPU pour median, CPU parallele pour interpolation
# Cela necessite de modifier le code pour choisir dynamiquement
speedup_optimal = 1 / (
    0.60 / interp_speedup +  # Interpolation parallelisee
    0.25 / 1.0 +             # ptp reste sur CPU (NumPy)
    0.10 / median_speedup +  # RANSAC sur GPU
    0.05
)
new_time_ica_opt = time_ica / speedup_optimal
new_time_final_opt = time_final / speedup_optimal

print("SCENARIO 3: CPU multicore + GPU selectif (optimal theorique)")
print("=" * 50)
print(f"  Gain global: {speedup_optimal:.1f}x")
print(f"  AutoReject ICA:   {time_ica:.0f} min -> {new_time_ica_opt:.0f} min ({new_time_ica_opt/60:.1f}h)")
print(f"  AutoReject Final: {time_final:.0f} min -> {new_time_final_opt:.0f} min ({new_time_final_opt/60:.1f}h)")
savings_opt = (time_ica + time_final - new_time_ica_opt - new_time_final_opt)
print(f"  Economie: {savings_opt:.0f} min ({savings_opt/60:.1f}h)")
print()

print("=" * 70)
print("RECOMMANDATION")
print("=" * 70)
print()
print("Pour votre cas, utilisez SEULEMENT la parallelisation CPU:")
print()
print("  ar = AutoReject(n_jobs=-1, ...)")
print()
print("Le GPU MPS (Apple Silicon) n'est pas avantageux ici car:")
print("  1. ptp() est 2.5x plus lent sur GPU (overhead transfert CPU<->GPU)")
print("  2. Seul median() beneficie du GPU (7x plus rapide)")
print("  3. L'interpolation (60% du temps) n'utilise pas notre backend GPU")
print()
print(f"Gain attendu avec n_jobs=-1: {speedup_cpu:.1f}x")
print(f"Temps total: {(time_ica + time_final)/60:.1f}h -> {(new_time_ica_cpu + new_time_final_cpu)/60:.1f}h")
print()
print("=" * 70)
print("POUR ALLER PLUS LOIN")
print("=" * 70)
print()
print("Pour vraiment accelerer AutoReject, il faudrait:")
print("  1. Paralleliser l'interpolation MNE elle-meme (complexe)")
print("  2. Reduire les parametres de recherche:")
print("     - n_interpolate=[1, 4, 16] au lieu de [1, 4, 32]")
print("     - consensus=linspace(0, 1, 5) au lieu de 11 valeurs")
print("     - cv=5 au lieu de 10")
print("  3. Utiliser get_rejection_threshold() pour un seuil global rapide")
print()
print("Avec des parametres reduits, vous pourriez gagner 2-4x supplementaires")
