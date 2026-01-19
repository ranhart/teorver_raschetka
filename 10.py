import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import os
from datetime import datetime

# Создание папки plots, если она не существует
plots_dir = "ad/TV/plots"
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)
    print(f"Создана папка: {plots_dir}")

# Чтение данных из файла
def read_data(filename):
    data = []
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            value = float(line.strip().replace(',', '.'))
            data.append(value)
    return np.array(data)

# Функция для проверки гипотезы о гамма-распределении с помощью критерия χ²
def gamma_chi_square_test(data, alpha=0.1, save_plot=True, plot_filename=None):
    n = len(data)
    
    print("=" * 70)
    print("ПРОВЕРКА ГИПОТЕЗЫ Ho: X ~ Г(k,Θ)")
    print("=" * 70)
    
    # Шаг 1: Оценка параметров гамма-распределения
    print("\n1. ОЦЕНКА ПАРАМЕТРОВ ГАММА-РАСПРЕДЕЛЕНИЯ")
    print("-" * 50)
    
    mean_val = np.mean(data)
    var_val = np.var(data, ddof=1)  # несмещенная оценка дисперсии
    
    # Оценка параметров методом моментов
    shape_hat = mean_val**2 / var_val      # параметр формы (k)
    scale_hat = var_val / mean_val         # параметр масштаба (θ)
    
    print(f"Выборочное среднее: x̄ = {mean_val:.4f}")
    print(f"Выборочная дисперсия: D² = (1/(n-1)) * Σ(x_i - x̄)² = {var_val:.4f}")
    print(f"Оценка параметра формы: k̂ = x̄²/D² = {mean_val:.4f}²/{var_val:.4f} = {shape_hat:.4f}")
    print(f"Оценка параметра масштаба: θ̂ = D²/x̄ = {var_val:.4f}/{mean_val:.4f} = {scale_hat:.4f}")
    
    # Шаг 2: Определение числа интервалов по формуле Старджесса
    print("\n2. ОПРЕДЕЛЕНИЕ ЧИСЛА ИНТЕРВАЛОВ")
    print("-" * 50)
    
    # Формула Старджесса: k = 1 + log₂(n)
    k_initial_float = 1 + np.log2(n)
    k = int(np.round(k_initial_float))  # ПРАВИЛЬНОЕ ОКРУГЛЕНИЕ!
    
    print(f"Формула Старджесса: k = 1 + log₂(n) = 1 + log₂({n})")
    print(f"log₂({n}) = np.log2({n}) = {np.log2(n):.3f}")
    print(f"k = 1 + {np.log2(n):.3f} = {k_initial_float:.3f}")
    print(f"Округление: {k_initial_float:.3f} → {k}")
    
    # Шаг 3: Создание равношироких интервалов
    print("\n3. СОЗДАНИЕ РАВНОШИРОКИХ ИНТЕРВАЛОВ")
    print("-" * 50)
    
    x_min = np.min(data)
    x_max = np.max(data)
    h = (x_max - x_min) / k  # ширина интервала
    
    print(f"Минимум выборки: X(1) = {x_min:.3f}")
    print(f"Максимум выборки: X(n) = {x_max:.3f}")
    print(f"Размах выборки: R = X(n) - X(1) = {x_max:.3f} - {x_min:.3f} = {x_max - x_min:.3f}")
    print(f"Ширина интервала: h = R/k = ({x_max:.3f} - {x_min:.3f}) / {k} = {h:.3f}")
    
    # Создание границ интервалов
    bounds = np.zeros(k + 1)
    for i in range(k + 1):
        bounds[i] = x_min + i * h
    
    # Корректировка последней границы для включения максимума
    bounds[-1] = x_max + 1e-10  # добавляем небольшую погрешность
    
    print(f"\nГраницы интервалов:")
    for i in range(k + 1):
        print(f"  a_{i} = {bounds[i]:.3f}")
    
    # Шаг 4: Расчет наблюдаемых частот
    observed_freq, _ = np.histogram(data, bins=bounds)
    
    # Шаг 5: Расчет теоретических вероятностей и частот для гамма-распределения
    expected_probs = np.zeros(k)
    for i in range(k):
        # Вероятность попадания в i-й интервал для гамма-распределения
        prob = (stats.gamma.cdf(bounds[i+1], a=shape_hat, scale=scale_hat) - 
                stats.gamma.cdf(bounds[i], a=shape_hat, scale=scale_hat))
        expected_probs[i] = prob
    
    expected_freq = expected_probs * n
    
    print(f"\nРаспределение по интервалам (БЕЗ ОБЪЕДИНЕНИЯ):")
    print(f"{'Интервал':<30} {'n_i':<8} {'np_i':<10}")
    print("-" * 50)
    
    for i in range(k):
        lower = bounds[i]
        upper = bounds[i+1]
        interval_str = f"[{lower:.2f}, {upper:.2f})"
        print(f"{interval_str:<30} {observed_freq[i]:<8.0f} {expected_freq[i]:<10.2f}")
    
    # Проверка условия np_i ≥ 5
    print(f"\nПроверка условия np_i ≥ 5:")
    min_expected = np.min(expected_freq)
    if min_expected < 5:
        print(f"  ВНИМАНИЕ: min(np_i) = {min_expected:.2f} < 5")
        print(f"  Это может нарушить условия применимости критерия χ²")
    else:
        print(f"  ✓ Все np_i ≥ 5 (min = {min_expected:.2f})")
    
    # Шаг 6: Расчет статистики χ²
    print("\n4. РАСЧЕТ СТАТИСТИКИ χ²")
    print("-" * 50)
    
    chi2_components = (observed_freq - expected_freq)**2 / expected_freq
    chi2_observed = np.sum(chi2_components)
    
    print(f"\nРасчетная таблица:")
    print(f"{'Интервал':<30} {'n_i':<8} {'np_i':<10} {'(n_i-np_i)²/np_i':<15}")
    print("-" * 70)
    
    for i in range(k):
        lower = bounds[i]
        upper = bounds[i+1]
        interval_str = f"[{lower:.2f}, {upper:.2f})"
        component = chi2_components[i]
        
        print(f"{interval_str:<30} {observed_freq[i]:<8.0f} "
              f"{expected_freq[i]:<10.2f} {component:<15.4f}")
    
    print("-" * 70)
    print(f"{'СУММА (χ²_набл)':<50} {chi2_observed:.4f}")
    
    # Шаг 7: Определение числа степеней свободы
    print("\n5. ОПРЕДЕЛЕНИЕ ЧИСЛА СТЕПЕНЕЙ СВОБОДЫ")
    print("-" * 50)
    
    m = 2  # оценено 2 параметра (k и θ)
    df = k - 1 - m
    
    print(f"Число интервалов: k = {k}")
    print(f"Число оцененных параметров: = {m} (k̂ и θ̂)")
    print(f"Число степеней свободы: v = k - 1 - ЧОП = {k} - 1 - {m} = {df}")
    
    # Шаг 8: Критическое значение χ²
    print("\n6. КРИТИЧЕСКОЕ ЗНАЧЕНИЕ")
    print("-" * 50)
    
    chi2_critical = stats.chi2.ppf(1 - alpha, df)
    print(f"Уровень значимости: α = {alpha}")
    print(f"Критическое значение: χ²_{1-alpha:.3f}[χ²(v)] = {chi2_critical:.4f}")
    
    # Шаг 9: P-value
    p_value = 1 - stats.chi2.cdf(chi2_observed, df)
    print(f"P-value: P(χ² ≥ {chi2_observed:.4f}) = {p_value:.6f}")
    
    # Шаг 10: Принятие решения
    print("\n7. ПРИНЯТИЕ РЕШЕНИЯ")
    print("-" * 50)
    
    if chi2_observed < chi2_critical:
        print(f"χ²_набл ({chi2_observed:.4f}) < χ²_крит ({chi2_critical:.4f})")
        print("✓ НЕТ оснований отвергнуть нулевую гипотезу H₀")
        print("  Выборка соответствует гамма-распределению")
        accepted = True
    else:
        print(f"χ²_набл ({chi2_observed:.4f}) ≥ χ²_крит ({chi2_critical:.4f})")
        print("✗ ОТВЕРГАЕМ нулевую гипотезу H₀")
        print("  Выборка НЕ соответствует гамма-распределению")
        accepted = False
    
    # Визуализация
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # 1. ГИСТОГРАММА И ГРАФИК ПЛОТНОСТИ ГАММА-РАСПРЕДЕЛЕНИЯ
    # Строим гистограмму с вычисленными интервалами
    ax1.hist(data, bins=bounds, density=True, alpha=0.6, color='gray', 
             edgecolor='black', label='Гистограмма выборки')
    
    # Добавляем график плотности гамма-распределения
    x_vals = np.linspace(x_min, x_max, 1000)
    gamma_pdf = stats.gamma.pdf(x_vals, a=shape_hat, scale=scale_hat)
    ax1.plot(x_vals, gamma_pdf, 'r-', linewidth=2, 
             label=f'Гамма-распределение\n(k̂={shape_hat:.2f}, θ̂={scale_hat:.2f})')
    
    # Отмечаем границы интервалов
    for bound in bounds:
        ax1.axvline(bound, color='black', linestyle=':', alpha=0.3, linewidth=0.5)
    
    ax1.set_xlabel('Значение', fontsize=12)
    ax1.set_ylabel('Плотность вероятности', fontsize=12)
    ax1.set_title('Гистограмма выборки и плотность гамма-распределения', 
                 fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Добавляем вертикальные линии для среднего и медианы
    ax1.axvline(mean_val, color='blue', linestyle='--', alpha=0.7, 
                label=f'Среднее = {mean_val:.2f}')
    ax1.axvline(np.median(data), color='red', linestyle='--', alpha=0.7, 
                label=f'Медиана = {np.median(data):.2f}')
    ax1.legend(loc='upper right', fontsize=9)
    
    # 2. ГРАФИК РАСПРЕДЕЛЕНИЯ χ²
    x_chi2 = np.linspace(0, max(chi2_observed * 1.5, chi2_critical * 1.5, df * 2), 1000)
    y_chi2 = stats.chi2.pdf(x_chi2, df)
    
    ax2.plot(x_chi2, y_chi2, 'b-', linewidth=2, label=f'χ² распределение (df={df})')
    
    # Критическое значение
    ax2.axvline(chi2_critical, color='r', linestyle='--', linewidth=2,
                label=f'χ² крит. = {chi2_critical:.2f}')
    
    # Наблюдаемое значение
    ax2.axvline(chi2_observed, color='g', linestyle='-', linewidth=2,
                label=f'χ² набл. = {chi2_observed:.2f}')
    
    # Закрашиваем область отвержения
    x_fill = np.linspace(chi2_critical, max(x_chi2), 100)
    y_fill = stats.chi2.pdf(x_fill, df)
    ax2.fill_between(x_fill, y_fill, alpha=0.3, color='red',
                    label=f'Область отвержения (α={alpha})')
    
    ax2.set_xlabel('χ² значение', fontsize=12)
    ax2.set_ylabel('Плотность вероятности', fontsize=12)
    ax2.set_title(f'Критерий χ²: проверка гамма-распределения\n'
                 f'χ²_набл = {chi2_observed:.2f}, χ²_крит = {chi2_critical:.2f}', 
                 fontsize=14, fontweight='bold')
    
    # Добавляем надпись о результате
    if accepted:
        result_text = "✓ ГИПОТЕЗА ПРИНЯТА"
        result_color = "green"
    else:
        result_text = "✗ ГИПОТЕЗА ОТВЕРГНУТА"
        result_color = "red"
    
    ax2.text(0.05, 0.95, result_text, transform=ax2.transAxes,
             fontsize=12, fontweight='bold', color=result_color,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # Общий заголовок
    fig.suptitle(f'Проверка гипотезы о гамма-распределении (n={n}, α={alpha})', 
                fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    # Сохранение графика вместо показа
    if save_plot:
        if plot_filename is None:
            # Генерируем имя файла с временной меткой
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_filename = f"gamma_test_{timestamp}.png"
        
        plot_path = os.path.join(plots_dir, plot_filename)
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"\nГрафик сохранен в файл: {plot_path}")
        plt.close(fig)  # Закрываем фигуру после сохранения
    else:
        plt.show()
    
    # Итоговое заключение
    print("\n" + "=" * 70)
    print("ИТОГОВОЕ ЗАКЛЮЧЕНИЕ")
    print("=" * 70)
    
    print(f"\nГипотеза H₀: Выборка имеет гамма-распределение")
    print(f"Гипотеза H₁: Выборка не имеет гамма-распределение")
    print(f"\nОценки параметров гамма-распределения:")
    print(f"  - Параметр формы: k̂ = {shape_hat:.4f}")
    print(f"  - Параметр масштаба: θ̂ = {scale_hat:.4f}")
    print(f"\nРезультаты критерия χ²:")
    print(f"  - Число интервалов: k = {k} (по формуле Старджесса)")
    print(f"  - Число степеней свободы: df = k - 1 - m = {k} - 1 - 2 = {df}")
    print(f"  - Наблюдаемое значение: χ²_набл = {chi2_observed:.4f}")
    print(f"  - Критическое значение: χ²_крит = χ²_{1-alpha:.3f}({df}) = {chi2_critical:.4f}")
    print(f"  - P-value: {p_value:.6f}")
    
    if accepted:
        print(f"\nВЫВОД: При уровне значимости α={alpha} НЕТ оснований отвергнуть гипотезу H₀.")
        print("Выборка соответствует гамма-распределению с параметрами:")
        print(f"k̂ = {shape_hat:.4f}, θ̂ = {scale_hat:.4f}")
    else:
        print(f"\nВЫВОД: При уровне значимости α={alpha} ОТВЕРГАЕМ гипотезу H₀.")
        print("Выборка НЕ соответствует гамма-распределению.")
    
    # Предупреждение о нарушении условия np_i ≥ 5
    if min_expected < 5:
        print(f"\n⚠  ПРЕДУПРЕЖДЕНИЕ: min(np_i) = {min_expected:.2f} < 5")
        print("   Это может нарушать условия применимости критерия χ²")
        print("   Рекомендуется объединить интервалы с малыми ожидаемыми частотами")
    
    print("=" * 70)
    
    return {
        'shape_hat': shape_hat,
        'scale_hat': scale_hat,
        'chi2_observed': chi2_observed,
        'chi2_critical': chi2_critical,
        'p_value': p_value,
        'df': df,
        'accepted': accepted,
        'bounds': bounds,
        'observed_freq': observed_freq,
        'expected_freq': expected_freq,
        'h': h,
        'k': k,
        'x_min': x_min,
        'x_max': x_max,
        'min_expected': min_expected,
        'plot_path': plot_path if save_plot else None
    }

# Основная программа
if __name__ == "__main__":
    # Чтение данных
    data = read_data("ad/TV/ras1.txt")
    n = len(data)
    
    print("=" * 70)
    print("АНАЛИЗ РАСПРЕДЕЛЕНИЯ ВЫБОРКИ")
    print("=" * 70)
    
    print(f"\nОСНОВНЫЕ ХАРАКТЕРИСТИКИ ВЫБОРКИ:")
    print(f"Объем выборки: n = {n}")
    print(f"Минимум: X(1) = {np.min(data):.3f}")
    print(f"Максимум: X(n) = {np.max(data):.3f}")
    print(f"Среднее: x̄ = {np.mean(data):.3f}")
    print(f"Дисперсия: D² = {np.var(data, ddof=1):.3f}")
    print(f"Стандартное отклонение: s = {np.std(data, ddof=1):.3f}")
    print(f"Медиана: {np.median(data):.3f}")
    print(f"Коэффициент вариации: {np.std(data, ddof=1)/np.mean(data)*100:.1f}%")
    
    # Проверка гипотезы о гамма-распределении
    results = gamma_chi_square_test(data, alpha=0.1, save_plot=True)