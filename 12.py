import numpy as np
import scipy.stats as stats

def read_data(filename):
    data = []
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            value = float(line.strip().replace(',', '.'))
            data.append(value)
    return np.array(data)

# Загрузка данных
data = read_data("ad/TV/ras1.txt")

print(f"\nРазделение выборки на первые 100 значение и оставшиеся 99")
print(f"H0 - выборки однородны")
# Разделение на выборки
n = 100
m = 99
x = data[:n]   # первая выборка
y = data[n:]   # вторая выборка

print()
print(f"Размер X: {len(x)}")
print(f"Размер Y: {len(y)}")

x = np.sort(x)
y = np.sort(y)

x_new = x.tolist()
y_new = y.tolist()


print(f'\n Вариационный ряд для x')
print(x)
print(f'\n Вариационный ряд для y')
print(y)

# Вычисление U
print()
print("Вычисление U: 1, если x[i] < y[j]; 0.5, если x[i] == y[j]; 0, если x[i] > y[j]")
U_new = 0 
for i in x_new:
    for j in y_new:
        if (i < j):
            U_new += 1
        if (i == j):
            U_new += 0.5
print("U = ", U_new)


# Вычисление Z
print()
print("Вычисление Z:")
print(f'Z = (U - n*m/2)/√(n*m(n+m+1)/12)')
Z = (U_new - (n*m/2)) / np.sqrt(n*m*(n+m+1)/12)  # исправил: было 200, теперь (n+m+1)=200
print(f"Z = {Z}")

# Вычисление Z_крит для разных уровней значимости
print('Z_крит = Z_(1-α/2) для N(0,1)')

# Уровни значимости
alpha_values = [0.01]

for alpha in alpha_values:
    Z_crit = stats.norm.ppf(1 - alpha/2)
    print(f"\nα = {alpha}:")
    print(f"x(0.995)[N(0,1)] = Ф(?) = 0.995")
    print(f"  Z_крит = {Z_crit:.2f}")
    
    # Проверка гипотезы
    if abs(Z) > Z_crit:
        print(f"  |{Z:.2f}| > {Z_crit:.2f} → отвергаем H0")
    else:
        print(f"  |{Z:.2f}| ≤ {Z_crit:.2f} → нет оснований отвергать H0")

