import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import os

# Чтение данных из файла
def read_data(filename):
    data = []
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            value = float(line.strip().replace(',', '.'))
            data.append(value)
    return np.array(data)

# Основная программа
if __name__ == "__main__":
    # Путь к файлу с данными
    data_filename = "ad/TV/ras1.txt"
    
    # Получаем текущую директорию
    current_dir = os.getcwd()
    
    # Создаем папку plots, если она не существует
    plots_dir = os.path.join(current_dir, "plots")
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
        print(f"Создана папка: {plots_dir}")
    
    print(f"Файл данных: {data_filename}")
    print(f"Текущая директория: {current_dir}")
    print(f"Папка для графиков: {plots_dir}")
    
    # Чтение данных
    data = read_data(data_filename)
    n = len(data)
    
    # 1. Оценка параметров гамма-распределения
    mean_val = np.mean(data)
    var_val = np.var(data, ddof=1)
    
    # Оценка параметров методом моментов
    shape_hat = mean_val**2 / var_val      # параметр формы (k)
    scale_hat = var_val / mean_val         # параметр масштаба (θ)
    
    print("=" * 70)
    print("ОЦЕНКА ПАРАМЕТРОВ ГАММА-РАСПРЕДЕЛЕНИЯ")
    print("=" * 70)
    print(f"Объем выборки: n = {n}")
    print(f"Выборочное среднее: x̄ = {mean_val:.4f}")
    print(f"Выборочная дисперсия: D² = {var_val:.4f}")
    print(f"Оценка параметра формы: k̂ = x̄²/D² = {shape_hat:.4f}")
    print(f"Оценка параметра масштаба: θ̂ = D²/x̄ = {scale_hat:.4f}")
    
    # 2. Построение графика гистограммы
    plt.figure(figsize=(12, 8))
    
    # 2.1 Гистограмма выборки
    plt.hist(data, bins='auto', density=True, alpha=0.6, 
             color='skyblue', edgecolor='black', 
             label=f'Гистограмма выборки (n={n})')
    
    # 2.2 Теоретическая плотность гамма-распределения с оцененными параметрами
    x = np.linspace(0, np.max(data) * 1.2, 1000)
    y_gamma = stats.gamma.pdf(x, a=shape_hat, scale=scale_hat)
    
    plt.plot(x, y_gamma, 'r-', linewidth=3, 
             label=f'Гамма-распределение\nk̂={shape_hat:.2f}, θ̂={scale_hat:.2f}')
    
    # 3. Настройка графика
    plt.xlabel('Значение', fontsize=14)
    plt.ylabel('Плотность вероятности', fontsize=14)
    plt.title('Гистограмма выборки и плотность гамма-распределения\nс оцененными параметрами', 
              fontsize=16, fontweight='bold', pad=20)
    
    # 4. Добавление дополнительной информации на график
    plt.axvline(mean_val, color='blue', linestyle='--', alpha=0.7, 
                label=f'Среднее (x̄={mean_val:.2f})')
    plt.axvline(np.median(data), color='green', linestyle='--', alpha=0.7, 
                label=f'Медиана={np.median(data):.2f}')
    
    # 5. Добавление параметров распределения в текстовое поле
    textstr = '\n'.join((
        f'Параметры гамма-распределения:',
        f'k̂ (форма) = {shape_hat:.4f}',
        f'θ̂ (масштаб) = {scale_hat:.4f}',
        f'',
        f'Характеристики выборки:',
        f'n = {n}',
        f'x̄ = {mean_val:.4f}',
        f's² = {var_val:.4f}',
        f'min = {np.min(data):.3f}',
        f'max = {np.max(data):.3f}'))
    
    # Помещаем текст в правом верхнем углу
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    plt.text(0.95, 0.95, textstr, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top', horizontalalignment='right',
             bbox=props)
    
    # 6. Настройка легенды и сетки
    plt.legend(loc='upper right', fontsize=11)
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # 7. Настройка осей
    plt.xlim([0, np.max(data) * 1.1])
    
    # 8. Сохранение графика в папку plots
    output_filename1 = os.path.join(plots_dir, 'histogram_gamma_fit.png')
    plt.tight_layout()
    plt.savefig(output_filename1, dpi=300, bbox_inches='tight')
    print(f"\nГрафик 1 сохранен: {output_filename1}")
    plt.show()
    
    # 9. Дополнительный анализ: сравнение эмпирической и теоретической ФР
    print("\n" + "=" * 70)
    print("СРАВНЕНИЕ ЭМПИРИЧЕСКОЙ И ТЕОРЕТИЧЕСКОЙ ФУНКЦИЙ РАСПРЕДЕЛЕНИЯ")
    print("=" * 70)
    
    # Эмпирическая функция распределения
    sorted_data = np.sort(data)
    ecdf = np.arange(1, n + 1) / n
    
    # Теоретическая функция распределения гамма
    theor_cdf = stats.gamma.cdf(sorted_data, a=shape_hat, scale=scale_hat)
    
    # Разность (для оценки близости)
    diff = np.abs(ecdf - theor_cdf)
    max_diff = np.max(diff)
    
    print(f"Максимальная разность между эмпирической и теоретической ФР: {max_diff:.4f}")
    
    # Построение графика ФР
    plt.figure(figsize=(10, 6))
    
    plt.step(sorted_data, ecdf, where='post', linewidth=2, 
             label='Эмпирическая ФР', color='blue', alpha=0.7)
    plt.plot(sorted_data, theor_cdf, 'r-', linewidth=2, 
             label=f'Теоретическая ФР гамма-распределения\nk̂={shape_hat:.2f}, θ̂={scale_hat:.2f}')
    
    plt.xlabel('Значение', fontsize=14)
    plt.ylabel('Функция распределения', fontsize=14)
    plt.title('Эмпирическая и теоретическая функции распределения', 
              fontsize=16, fontweight='bold')
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Сохранение второго графика в папку plots
    output_filename2 = os.path.join(plots_dir, 'ecdf_theor_cdf.png')
    plt.tight_layout()
    plt.savefig(output_filename2, dpi=300, bbox_inches='tight')
    print(f"График 2 сохранен: {output_filename2}")
    plt.show()
    
    print("\n" + "=" * 70)
    print("АНАЛИЗ ЗАВЕРШЕН")
    print(f"Все графики сохранены в папке: {plots_dir}")
    print("=" * 70)