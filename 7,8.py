import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


file_path = "III задание данные.xlsx"
df = pd.read_excel(file_path, sheet_name=0, header=None)
data = df.iloc[:, 0].dropna().astype(float).values

print("МАССИВ ДАННЫХ:")
for i in range(0, len(data), 15):
    print("  ".join(f"{val:8.3f}" for val in data[i:i + 15]))
print(f"\nВсего значений: {len(data)}\n")

# Обозначение переменных
print("=" * 70)
print("ОБОЗНАЧЕНИЯ ПЕРЕМЕННЫХ")
print("=" * 70)
print("n     - количество значений в выборке")
print("x_min - минимальное значение")
print("x_max - максимальное значение")
print("k     - количество интервалов (формула Стёржеса)")
print("l     - длина (ширина) каждого интервала")
print("n_i   - частота (количество значений в интервале)")
print("h_i   - высота столбца гистограммы = n_i / (n × l)")
print("=" * 70)

# Параметры
n = len(data)
x_min = min(data)
x_max = max(data)

print(f"\nn     = {n}")
print(f"x_min = {x_min:.3f}")
print(f"x_max = {x_max:.3f}")

# Расчёт k
log2_n = np.log2(n)
k_calculated = 1 + log2_n
k = int(np.ceil(k_calculated))

print(f"\nk = 1 + log₂({n}) = 1 + {log2_n:.3f} = {k_calculated:.3f} ≈ {k}")

# Расчёт l (длина интервала)
l = (x_max - x_min) / k

print(f"l = (x_max - x_min) / k = ({x_max:.3f} - {x_min:.3f}) / {k} = {l:.3f}")
print("\n" + "=" * 70)

# Таблица
print("\n{:<3} {:<20} {:<10} {:<15}".format("№", "Интервал", "n_i", "h_i"))
print("-" * 70)

intervals = []
left = x_min
for i in range(k):
    right = left + l
    intervals.append((left, right))
    left = right

for idx, (left, right) in enumerate(intervals, 1):
    if idx == len(intervals):
        n_i = sum(1 for x in data if left <= x <= right)
    else:
        n_i = sum(1 for x in data if left <= x < right)

    h_i = n_i / (n * l)
    interval_str = f"[{left:.3f}; {right:.3f})"

    print("{:<3} {:<20} {:<10} {:<15.6f}".format(idx, interval_str, n_i, h_i))

print("-" * 70)


# Собираем данные для гистограммы
heights = []
for idx, (left, right) in enumerate(intervals, 1):
    if idx == len(intervals):
        n_i = sum(1 for x in data if left <= x <= right)
    else:
        n_i = sum(1 for x in data if left <= x < right)
    h_i = n_i / (n * l)
    heights.append(h_i)

# Строим гистограмму
labels = [str(i) for i in range(1, k + 1)]
plt.figure(figsize=(12, 6))
plt.bar(labels, heights, edgecolor='black', color='skyblue', width=0.8)
plt.xlabel('Интервал')
plt.ylabel('Высота столбца (h_i)')
plt.title('Гистограмма относительных частот')
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()


data_sorted = sorted(data)

print("\n" + "=" * 90)
print("ВЫБОРОЧНЫЕ ХАРАКТЕРИСТИКИ")
print("=" * 90)

# 1. СРЕДНЕЕ
print("\n1. СРЕДНЕЕ (выборочное математическое ожидание)")
print("   Формула: X̄ = (1/n) × Σ(X_i)")
sum_xi = sum(data)
print(f"\n   Вычисляем Σ(X_i):")
print(f"   Σ(X_i) = X₁ + X₂ + X₃ + ... + X₁₉₉")
print(f"   Σ(X_i) = {' + '.join([f'{data[i]:.3f}' for i in range(min(5, len(data)))])} + ... + {data[-1]:.3f}")
print(f"   Σ(X_i) = {sum_xi:.3f}")
mean = sum_xi / n
print(f"\n   X̄ = (1/{n}) × {sum_xi:.3f}")
print(f"   X̄ = {mean:.6f}\n")

# 2. ДИСПЕРСИЯ
print("2. ДИСПЕРСИЯ (выборочная)")
print("   Формула: S² = (1/n) × Σ(X_i²) - (X̄)²")
sum_xi_squared = sum(x**2 for x in data)
print(f"\n   Вычисляем Σ(X_i²):")
print(f"   Σ(X_i²) = X₁² + X₂² + X₃² + ... + X₁₉₉²")
print(f"   Σ(X_i²) = {data[0]:.3f}² + {data[1]:.3f}² + {data[2]:.3f}² + ... + {data[-1]:.3f}²")
print(f"   Σ(X_i²) = {data[0]**2:.3f} + {data[1]**2:.3f} + {data[2]**2:.3f} + ... + {data[-1]**2:.3f}")
print(f"   Σ(X_i²) = {sum_xi_squared:.3f}")
mean_squared = mean**2
print(f"\n   (X̄)² = ({mean:.6f})² = {mean_squared:.6f}")
variance = (sum_xi_squared / n) - mean_squared
print(f"\n   S² = (1/{n}) × {sum_xi_squared:.3f} - {mean_squared:.6f}")
print(f"   S² = {sum_xi_squared/n:.6f} - {mean_squared:.6f}")
print(f"   S² = {variance:.6f}")
std_dev = np.sqrt(variance)
print(f"\n   σ = √(S²) = √({variance:.6f}) = {std_dev:.6f}\n")

# 3. МЕДИАНА (ПО ФОРМУЛЕ ИЗ ФОТО)
print("3. МЕДИАНА (выборочная квантиль порядка 0,5)")
print("   Формула: x*_p = { X_{[np]+1}         если np - дробное")
print("                   { (X_{np} + X_{np+1})/2  если np - целое")
np_value = n * 0.5
print(f"\n   np = {n} × 0,5 = {np_value}")

if np_value == int(np_value):  # если ЦЕЛОЕ
    print(f"   np = {int(np_value)} - ЦЕЛОЕ число")
    np_int = int(np_value)
    x_np = data_sorted[np_int - 1]
    x_np_plus_1 = data_sorted[np_int]
    median = (x_np + x_np_plus_1) / 2
    print(f"   x*_med = (X_{{{np_int}}} + X_{{{np_int+1}}}) / 2 = ({x_np:.6f} + {x_np_plus_1:.6f}) / 2 = {median:.6f}\n")
else:  # если ДРОБНОЕ
    print(f"   np = {np_value} - ДРОБНОЕ число")
    floor_np = int(np.floor(np_value))
    ceil_np = floor_np + 1
    print(f"   [np] = {floor_np}")
    print(f"   [np] + 1 = {floor_np} + 1 = {ceil_np}")
    median = data_sorted[ceil_np - 1]
    print(f"   x*_med = X_{{{ceil_np}}} = data_sorted[{ceil_np - 1}] = {median:.6f}\n")


# 4. КОЭФФИЦИЕНТ АСИММЕТРИИ (KAS)
print("4. КОЭФФИЦИЕНТ АСИММЕТРИИ")
print("   Формула: Kas = μ₃* / (√Sx²)³")
print("   где μ₃* = (1/n) × Σ(X_i - X̄)³ — центральный момент 3-го порядка\n")

# Вычисляем отклонения один раз
deviations = [x - mean for x in data]

# Выводим первые 3 отклонения
for i in range(min(3, len(deviations))):
    print(f"   (X_{i+1} - X̄) = {data[i]:.6f} - {mean:.6f} = {deviations[i]:.6f}")
print(f"   ... (всего {n} отклонений) ...\n")

# Вычисляем Σ(X_i - X̄)³
deviations_cubed = sum(dev**3 for dev in deviations)
print(f"   Вычисляем Σ(X_i - X̄)³:")
print(f"   (X₁ - X̄)³ = ({deviations[0]:.6f})³ = {deviations[0]**3:.6f}")
print(f"   (X₂ - X̄)³ = ({deviations[1]:.6f})³ = {deviations[1]**3:.6f}")
print(f"   (X₃ - X̄)³ = ({deviations[2]:.6f})³ = {deviations[2]**3:.6f}")
print(f"   ... (всего {n} членов) ...")
print(f"   Σ(X_i - X̄)³ = {deviations_cubed:.6f}\n")

# Вычисляем центральный момент 3-го порядка
mu3_star = deviations_cubed / n
print(f"   μ₃* = (1/{n}) × {deviations_cubed:.6f} = {mu3_star:.6f}\n")

# Используем уже вычисленные σ = √Sx²
sqrt_sx2_cubed = std_dev**3
print(f"   Из раздела 2 (ДИСПЕРСИЯ): σ = √(S²) = {std_dev:.6f}")
print(f"   (√Sx²)³ = ({std_dev:.6f})³ = {sqrt_sx2_cubed:.6f}\n")

# Вычисляем коэффициент асимметрии
kas = mu3_star / sqrt_sx2_cubed
print(f"   Kas = {mu3_star:.6f} / {sqrt_sx2_cubed:.6f}")
print(f"   Kas = {kas:.6f}\n")

# 5. КОЭФФИЦИЕНТ ЭКСЦЕССА (KEX)
print("5. КОЭФФИЦИЕНТ ЭКСЦЕССА (КУРТОЗИС)")
print("   Формула: Kex = μ₄* / (√Sx²)⁴ - 3")
print("   где μ₄* = (1/n) × Σ(X_i - X̄)⁴ — центральный момент 4-го порядка\n")

print(f"   Используем вычисленные отклонения (X_i - X̄):\n")

# Вычисляем Σ(X_i - X̄)⁴
deviations_fourth = sum(dev**4 for dev in deviations)
print(f"   Вычисляем Σ(X_i - X̄)⁴:")
print(f"   (X₁ - X̄)⁴ = ({deviations[0]:.6f})⁴ = {deviations[0]**4:.6f}")
print(f"   (X₂ - X̄)⁴ = ({deviations[1]:.6f})⁴ = {deviations[1]**4:.6f}")
print(f"   (X₃ - X̄)⁴ = ({deviations[2]:.6f})⁴ = {deviations[2]**4:.6f}")
print(f"   ... (всего {n} членов) ...")
print(f"   Σ(X_i - X̄)⁴ = {deviations_fourth:.6f}\n")

# Вычисляем центральный момент 4-го порядка
mu4_star = deviations_fourth / n
print(f"   μ₄* = (1/{n}) × {deviations_fourth:.6f} = {mu4_star:.6f}\n")

# Используем уже вычисленные σ = √Sx²
sqrt_sx2_fourth = std_dev**4
print(f"   Из раздела 2 (ДИСПЕРСИЯ): σ = √(S²) = {std_dev:.6f}")
print(f"   (√Sx²)⁴ = ({std_dev:.6f})⁴ = {sqrt_sx2_fourth:.6f}\n")

# Вычисляем куртозис
excess_before_subtract = mu4_star / sqrt_sx2_fourth
kex = excess_before_subtract - 3
print(f"   Kex = {mu4_star:.6f} / {sqrt_sx2_fourth:.6f} - 3")
print(f"   Kex = {excess_before_subtract:.6f} - 3")
print(f"   Kex = {kex:.6f}\n")

print("=" * 80)
print("выборочные характеристики".center(80))
print("=" * 80)
print(f"\n   Параметры распределения:")
print(f"   • Среднее (X̄)               = {mean:.6f}")
print(f"   • Дисперсия (S²)            = {variance:.6f}")
print(f"   • Медиана (x*_med)          = {median:.6f}")
print(f"\n   Коэффициенты формы распределения:")
print(f"   • Асимметрия (Kas)          = {kas:.6f}")
print(f"   • Эксцесс (Kex)             = {kex:.6f}")
print("\n" + "=" * 80)


# === ЗАДАНИЕ 8: КРИТЕРИЙ СЕРИЙ ===

print("\n" + "="*80)
print("ЗАДАНИЕ 8. КРИТЕРИЙ СЕРИЙ (ПРОВЕРКА СЛУЧАЙНОСТИ)".center(80))
print("="*80)

# 1. КОДИРОВАНИЕ (+ и -)
print("\n1. КОДИРОВАНИЕ ВЫБОРКИ")

print("\n1. Вариационный ряд:")
for i in range(0, len(data_sorted), 15):
    start = i + 1
    end = min(i + 15, len(data_sorted))
    values_str = "  ".join(f"{val:8.3f}" for val in data_sorted[i:i+15])
    print(f"X_{start:3d} - X_{end:3d}:  {values_str}")

median_7 = np.median(data_sorted)
print(f"\nМедиана из задания 7 = {median_7:.3f}\n")

signs = []
for x in data:
    if x > median_7:
        signs.append('+')
    elif x < median_7:
        signs.append('-')

print(f"   Знаковая последовательность (по 50 в строке):")
for i in range(0, len(signs), 50):
    start = i + 1
    end = min(i + 50, len(signs))
    signs_str = ''.join(signs[i:i+50])
    print(f"   {start:3d}-{end:3d}: {signs_str}")

print(f"\n2. ПОДСЧЕТ ХАРАКТЕРИСТИК")
n1 = signs.count('+')
n2 = signs.count('-')
print(f"   n₁ (количество плюсов) = {n1}")
print(f"   n₂ (количество минусов) = {n2}")

KS = 1
for i in range(1, len(signs)):
    if signs[i] != signs[i-1]:
        KS += 1
print(f"   KS (количество серий) = {KS}")
