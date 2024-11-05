#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from prettytable import PrettyTable
import matplotlib.pyplot as plt
import math


def arrange_temperature(any_list):
    """
    Функция упорядочивания списка по возрастанию (создание непрерывного вариационного ряда).

    :param any_list: изначальная выборка.
    :return: непрерывный вариационный ряд.
    """
    return sorted(any_list)


def scope(any_list):
    """
    Функция нахождения размаха варьирования признака.

    :param any_list: непрерывный вариационный ряд.
    :return: размах варьирования признака.
    """
    return any_list[-1] - any_list[0]


def number_intervals(any_list):
    """
    Функция нахождения количества интервалов вариационного ряда.

    :param any_list: изначальная выборка.
    :return: округленное до целого количество частичных интервалов.
    """
    k_intervals = 1 + (3.2 * math.log(len(any_list)))
    return round(k_intervals)


def length_intervals(a, b):
    """
    Функция нахождения длины частичных интервалов.

    :param a: размах варьирования признака;
    :param b: количество частичных интервалов.
    :return: длина частичного интервала.
    """
    return round(a / b, 1)


def interval_table(any_list, k_num, h_num):
    """
    Функция постоения интервального вариационного ряда
    и получения списка вариант интервалов и списка частот для него.

    :param any_list: вариационный ряд;
    :param k_num: число интервалов;
    :param h_num: длина одного интервала;
    :return: содержимое таблицы: список вариант интервалов (field_list) и список частот (row_list).
    """
    table = PrettyTable()
    field_list = ["Варианты интервалы"]
    row_list = ["Частоты"]
    a = min(any_list) - (0.5 * h_num)
    c = " - "
    count_control = 0
    for i in range(0, k_num - 1):
        b = round(a + h_num, 2)
        meaning = f"{a}{c}{b}"
        field_list.append(meaning)
        count = 0
        for item in any_list:
            if a <= item <= b:
                count += 1
        row_list.append(count)
        count_control += count
        a = b

    b = max(any_list) + (0.5 * h_num)

    count = 0
    for item in any_list:
        if a <= item <= b:
            count += 1
    row_list.append(count)
    count_control += count

    field_list.append(f"{a}{c}{b}")

    table.field_names = field_list

    table.add_row(row_list)

    print(table)

    # Проведем контроль для полученных частот.
    print("Проведем контроль для полученных частот:")
    if count_control == len(any_list):
        print("Контроль пройден: Сумма ni (частот) = n (объему выборки)")
    else:
        print("Контроль не пройден: Сумма ni (частот) != n (объему выборки)")

    field_list.pop(0)  # удаляем ненужные первые элементы
    row_list.pop(0)

    return field_list, row_list


def discrete_print(field_list, row_list):
    """
    Функция вывода дискретного вариационного ряда.

    :param field_list: середины интервалов
    :param row_list: частоты
    :return: ничего
    """
    # Копируем списки, чтобы не менять оригинальные
    field_list_copy = field_list.copy()
    row_list_copy = row_list.copy()

    # Создаем таблицу
    table = PrettyTable()

    # Вставляем заголовки
    field_list_copy.insert(0, 'Варианты, xi')
    row_list_copy.insert(0, 'Частоты, ni')

    # Добавляем строки в таблицу
    table.field_names = field_list_copy
    table.add_row(row_list_copy)

    # Печатаем таблицу
    print(table)


def find_moda(midpoints, frequency):
    """
    Функция нахождения моды среди вариант.

    :param midpoints: список вариантов (середины интервалов);
    :param frequency: список их частот.
    :return mode: мода.
    """
    # Находим индекс максимальной частоты
    max_frequency_index = frequency.index(max(frequency))

    # Мода — это вариант с максимальной частотой
    mode = midpoints[max_frequency_index]

    return mode


def calculation_table(num_k, frequency, midpoints, mode, num_h):
    """
    Функция создания расчетной таблицы.

    :param num_k: число интервалов или условных вариант;
    :param frequency: список частот вариант;
    :param midpoints: список середин интервалов;
    :param mode: мода среди всех вариант в списке середин интервалов;
    :param num_h: длина одного интервала.
    :return compute_table: рассчетная таблица (посчитанная).
    """

    # Создаем двумерный пустой массив.
    compute_table = [[0 for _ in range(8)] for _ in range(num_k + 1)]

    # Заполним созданный список расчетными значениями
    for i in range(0, num_k):
        compute_table[i][0] = midpoints[i]

    for i in range(0, num_k):
        compute_table[i][1] = frequency[i]

    for i in frequency:
        compute_table[num_k][1] += i

    # Заполним третий столбец ui, создадим условные варианты
    for i in range(0, num_k):
        compute_table[i][2] = round((compute_table[i][0] - mode) / num_h)

    # Заполним четвертый столбец таблицы, ni * ui
    for i in range(0, num_k):
        compute_table[i][3] = compute_table[i][1] * compute_table[i][2]
        compute_table[num_k][3] += compute_table[i][3]

    # Заполним пятый столбец таблицы, ni * ui^2
    for i in range(0, num_k):
        compute_table[i][4] = compute_table[i][1] * compute_table[i][2] * compute_table[i][2]
        compute_table[num_k][4] += compute_table[i][4]

    # Заполним шестой столбец таблицы, ni * ui^3
    for i in range(0, num_k):
        compute_table[i][5] = compute_table[i][1] * compute_table[i][2] * compute_table[i][2] * compute_table[i][2]
        compute_table[num_k][5] += compute_table[i][5]

    # Заполним седьмой столбец таблицы, ni * ui^4
    for i in range(0, num_k):
        compute_table[i][6] =\
            compute_table[i][1] * compute_table[i][2] * compute_table[i][2] * compute_table[i][2] * compute_table[i][2]
        compute_table[num_k][6] += compute_table[i][6]

    # Заполним контрольный столбец таблицы, ni * (ui+1)^2
    for i in range(0, num_k):
        compute_table[i][7] = compute_table[i][1] * (compute_table[i][2] + 1) * (compute_table[i][2] + 1)
        compute_table[num_k][7] += compute_table[i][7]

    # Создание таблицы
    table = PrettyTable()

    # Установка заголовков столбцов (например, Col1, Col2, ..., Col8)
    columns = ['xi', 'ni', 'ui', 'ni * ui', 'ni * ui^2', 'ni * ui^3', 'ni * ui^4', 'контрольный столбец, ni * (ui+1)^2']
    table.field_names = columns

    # Добавляем строки из списка в таблицу
    for row in compute_table:
        table.add_row(row)

    # Вывод таблицы
    print('Расчетная таблица:')
    print(table)

    # Произведем контроль вычислений
    print("Проведем контроль вычислений табличных данных:")
    if compute_table[num_k][1] + (2 * compute_table[num_k][3]) + compute_table[num_k][4] == compute_table[num_k][7]:
        print("Контроль пройден, таблица рассчитана верно: "
              "Сумма(ni) + 2*Сумма(ni*ui) + Сумма(ni * ui^2) = Контрольной сумме")
    else:
        print("Контроль не пройден, таблица рассчитана неверно: "
              "Сумма(ni) + 2*Сумма(ni*ui) + Сумма(ni * ui^2) != Контрольной сумме")

    # Вернем табличные значения для дальнейших вычислений
    return compute_table


def calculation_moments(compute_table, any_list, num_k):
    """
    Функция нахождения начальных условных моментов различных порядков по расчетной таблице.

    :param compute_table: расчетная таблица;
    :param any_list: изначальная выборка;
    :param num_k: число условных вариант.
    :return m1, m2, m3, m4: начальные моменты.
    """

    # Вычислим количество элементов в выборке
    num_n = len(any_list)

    # Найдем условный начальный момент первого порядка
    m1 = compute_table[num_k][3] / num_n

    # Найдем условный начальный момент второго порядка
    m2 = compute_table[num_k][4] / num_n

    # Найдем условный начальный момент третьего порядка
    m3 = compute_table[num_k][5] / num_n

    # Найдем условный начальный момент четверного порядка
    m4 = compute_table[num_k][6] / num_n

    # Выведем все моменты на экран
    print("Начальный условный момент первого порядка М1 = ", m1)
    print("Начальный условный момент второго порядка М2 = ", m2)
    print("Начальный условный момент третьего порядка М3 = ", m3)
    print("Начальный условный момент четвертого порядка М4 = ", m4)

    return m1, m2, m3, m4


def calculate_sample_variance(m1, m2, num_h):
    """
    Функция нахождения выборочной дисперсии.

    :param m1: условный момент первого порядка;
    :param m2: условный момент второго порядка;
    :param num_h: размер одного интервала.
    :return s: выборочная дисперсия.
    """

    s = (m2 - (m1 * m1)) * num_h * num_h
    print("Выборочная дисперсия S_2 = ", s)

    return s


def calculate_rms(s):
    """
    Функция нахождения среднего квадратичного отклонения.

    :param s: дисперсия.
    :return s_rms: среднее квадратичное отклонение.
    """

    s_rms = math.sqrt(s)
    print("Среднее квадратичное отклонение S = ", s_rms)

    return s_rms


def get_middle(any_list):
    """
    Функция получения из списка интервалов - список середин интервалов.
    Возвращает список середин интервалов для дискретного вариационного ряда.

    :param any_list: список интервалов в виде строк.
    :return: midpoints: получившийся список середин интервалов
    """
    midpoints = []
    for interval in any_list:
        # Разбиваем строку по " - " и преобразуем в числа
        start, end = map(float, interval.split(' - '))
        # Вычисляем середину интервала
        midpoint = (start + end) / 2
        # Округляем значение
        midpoint = round(midpoint, 2)
        midpoints.append(midpoint)

    return midpoints


def find_middle(any_list, frequency):
    """
    Функция нахождения среднеарифметического значения.

    :param any_list: дискретный вариационный ряд.
    :param frequency: частоты вариант.
    :return: среднеарифметическое значение.
    """

    sum_x = 0
    for i in range(0, len(any_list)):
        sum_x += any_list[i] * frequency[i]

    return sum_x / sum(frequency)


def calculation_theoretical_table(any_list, frequency, x_mid, rms, num_h):
    """
    Функция нахождения значений и построения расчетной таблицы.

    :param any_list: Дискретный вариационный ряд;
    :param frequency: частоты дискретного вариационного ряда;
    :param x_mid: среднее значение исходя из дискретного вариационного ряда;
    :param rms: среднеквадратическое отклонение;
    :param num_h: длина частичного интервала.
    :return: расчетная таблица и список теоретических частот.
    """

    # Создаем двумерный пустой массив.
    compute_table = [[0 for _ in range(7)] for _ in range(len(any_list))]

    # Заполним созданный список расчетными значениями
    for i in range(0, len(any_list)):
        compute_table[i][0] = any_list[i]

    for i in range(0, len(any_list)):
        compute_table[i][1] = frequency[i]

    for i in range(0, len(any_list)):
        compute_table[i][2] = any_list[i] - x_mid

    for i in range(0, len(any_list)):
        compute_table[i][3] = round((compute_table[i][2] / rms), 1)

    f_ui = [
        0.0175, 0.044, 0.079, 0.1295, 0.1942, 0.2661, 0.3332, 0.3814,
        0.3989, 0.3814, 0.3332, 0.2661, 0.1942, 0.1295, 0.054
    ]

    for i in range(0, len(any_list)):
        compute_table[i][4] = f_ui[i]

    for i in range(0, len(any_list)):
        compute_table[i][5] = (90 * num_h * compute_table[i][4]) / rms

    for i in range(0, len(any_list)):
        compute_table[i][6] = round(compute_table[i][5])

    # Создание таблицы
    table = PrettyTable()

    # Установка заголовков столбцов (например, Col1, Col2, ..., Col8)
    columns = ['xi', 'ni', 'xi-xср', 'ui=(xi-xср)/S', 'ф(ui)', 'yi', 'ni`']
    table.field_names = columns

    # Добавляем строки из списка в таблицу
    for row in compute_table:
        table.add_row(row)

    # Вывод таблицы
    print('Расчетная таблица:')
    print(table)

    theoretical_frequency = []
    for i in range(0, len(any_list)):
        theoretical_frequency.append(compute_table[i][6])

    # Вернем табличные значения для дальнейших вычислений и список теоретических частот
    return compute_table, theoretical_frequency


def plot_frequency_polygon(x_values, frequency, theoretical_frequency):
    """
    Функция для построения полигона частот и теоретической кривой нормального распределения.
    :param x_values: список значений дискретного вариационного ряда
    :param frequency: список эмпирических частот
    :param theoretical_frequency: список теоретических частот нормального распределения
    """
    plt.figure(figsize=(10, 6))

    # Построение эмпирического полигона частот
    plt.plot(x_values, frequency, marker='o', linestyle='-', color='blue',
             label='Эмпирическая кривая')

    # Построение теоретической кривой нормального распределения
    plt.plot(x_values, theoretical_frequency, marker='o', linestyle='--', color='red',
             label='Нормальное распределение')

    # Настройки графика
    plt.xlabel('Значения вариационного ряда')
    plt.ylabel('Частоты')
    plt.title('Полигон частот и теоретическая кривая нормального распределения')
    plt.legend()
    plt.grid(True)

    # Отображение графика
    plt.show()


def pirson(compute_table, any_list):
    """
    Функция проверки согласованности распределения по критерию Пирсона.

    :param compute_table: Расчетная таблица
    :param any_list: дискретный вариационный ряд.
    :return: хи квадрат, коэффициент.
    """

    # Вычислим значение величины Хи квадрат
    xi_2 = 0
    for i in range(len(any_list)):
        xi_2 += ((compute_table[i][1] - compute_table[i][6]) *
                 (compute_table[i][1] - compute_table[i][6])) / compute_table[i][5]
    print("Величина Xи^2 = ", xi_2)

    # Пусть уровень значимости a = 0,95, тогда
    num_k = len(any_list) - 3
    a = 0.95
    print("Число степеней свободы k = ", num_k)
    print("Уровень значисмости а = ", a)

    # Исходя из приложения 1:
    xi_kr_2 = 5.23
    print("Критическое значение Хи^2", xi_kr_2)

    if xi_2 < xi_kr_2:
        print("Хи^2 < Xкр^2, распределение соотвествует нормальному согласно критерию Пирсона.")
    else:
        print("Хи^2 > Xкр^2, распределение не соотвествует нормальному согласно критерию Пирсона.")

    return xi_2


def kolmagorov(any_list, frequency, theoretical_frequency):
    """
    Функция проверки согласованности распределения нормальному по критерию Колмагорова.

    :param any_list: выборка;
    :param frequency: частоты дискретного вариационного ряда;
    :param theoretical_frequency: частоты теоретического нормального распределения.
    :return: значение функции Колмагорова.
    """

    # Вычислим объем выборки
    sample_size = len(any_list)

    # Находим общие суммы для нормализации частот
    sum_empirical = sum(frequency)
    sum_theoretical = sum(theoretical_frequency)

    # Вычисляем накопленные частоты вручную
    cumulative_empirical = []
    cumulative_theoretical = []

    cumulative_sum_empirical = 0
    cumulative_sum_theoretical = 0

    for i in frequency:
        cumulative_sum_empirical += i
        # Нормализуем накопленную сумму и добавляем в список накопленной частоты
        cumulative_empirical.append(cumulative_sum_empirical / sum_empirical)

    for i in theoretical_frequency:
        cumulative_sum_theoretical += i
        # Нормализуем накопленную сумму и добавляем в список накопленной частоты
        cumulative_theoretical.append(cumulative_sum_theoretical / sum_theoretical)

    # Вычисляем абсолютные разности и находим максимальную разность
    max_difference = max(abs(emp - theo) for emp, theo in zip(cumulative_empirical, cumulative_theoretical))

    # Делим на корень из объема выборки для получения статистики Колмогорова
    lambda_k = max_difference / math.sqrt(sample_size)

    print("Статистика Колмагорова = ", lambda_k)

    # Исходя из статистики Колмагорова найдем значение функции K(lambda_k)
    summation = 0
    for num_k in range(1, 11):
        summation += (-1) ** (num_k - 1) * math.exp(-2 * (num_k ** 2) * (lambda_k ** 2))

    k_lambda_k = 1 - 2 * summation

    print("Значение функции K(lambda_k) = ", k_lambda_k)

    if k_lambda_k < 0.05:
        print("Распределение соответствует нормальному согласно критерию Колмагорова.")
    else:
        print("Распределение не соответствует нормальному согласно критерию Колмагорова.")

    return k_lambda_k


def calculate_central_moments(m1, m2, m3, m4, num_h):
    """
    Функция расчета центральных моментов третьего и четвертого порядков.

    :param m1: условный начальный момент первого порядка;
    :param m2: условный начальный момент второго порядка;
    :param m3: условный начальный момент третьего порядка;
    :param m4: условный начальный момент четвертого порядка.
    :param num_h: длина одного интервала.
    :return m_3, m_4: центральные моменты.
    """

    m_3 = (m3 - (3 * m2 * m1) + (2 * m1 * m1 * m1)) * num_h * num_h * num_h
    print("Центральный момент третьего порядка: m3 = ", m_3)
    m_4 = (m4 - (4 * m3 * m1) + (6 * m2 * m1 * m1) - (3 * m1 * m1 * m1 * m1)) * num_h * num_h * num_h * num_h
    print("Центральный момент четвертого порядка: m4 = ", m_4)

    return m_3, m_4


def asymmetry_excess_calculate(m_3, m_4, s):
    """
    Функция нахождения коэффициентов асимметрии и эксцесса.

    :param m_3: центральный момент третьего порядка;
    :param m_4: центральный момент четвертого порядка;
    :param s: среднеквадратичное отклонение.
    :return a_s, e_x: коэффициенты асимметрии и эксцесса.
    """

    a_s = m_3 / (s * s * s)
    print("Коэффициент асимметрии: As = ", a_s)
    e_x = (m_4 / (s * s * s * s)) - 3
    print("Коэффициент эксцесса: Ex = ", e_x)

    return a_s, e_x


def rsm_asymmetry_excess(any_list):
    """
    Функция нахождения среднеквадратического отклонения асимметрии и эксцесса.

    :param any_list: выборка.
    :return: среднеквадратическое отклонение асимметрии и среднеквадратическое отклонение эксцесса.
    """

    # Найдем объем выборки
    num_n = len(any_list)

    # Вычислим среднеквадратическое отклонение асимметрии
    s_as = math.sqrt((6 * (num_n - 1)) / ((num_n + 1) * (num_n + 3)))

    # Вычислим среднеквадратическое отклонение эксцесса
    s_ex = math.sqrt(((24 * num_n) * (num_n - 2) * (num_n - 3) /
                      ((num_n - 1) * (num_n - 1) * (num_n + 3) * (num_n + 5))))

    print("Среднеквадратическое отклонение асимметрии S_As = ", s_as)
    print("Среднеквадратическое отклонение эксцесса S_Ex = ", s_ex)

    return s_as, s_ex


def approximate_criterion(a_s, e_x, s_as, s_ex):
    """
    Функция проверки согласованности нормальному закону распределения согласно приближенному критерию.

    :param a_s: коэффициент асимметрии;
    :param e_x: коэффициент эксцесса;
    :param s_as: среднеквадратическое отклонение асимметрии;
    :param s_ex: среднеквадратическое отклонение эксцесса.
    :return: ничего.
    """
    
    if abs(a_s) <= s_as and abs(e_x) <= s_ex:
        print("Выборочная совокупность подчиняется нормальному закону распределения согласно приближенному критерию.")
    else:
        print("Выборочная совокупность не подчиняется нормальному "
              "закону распределения согласно приближенному критерию.")


if __name__ == '__main__':
    # Создадим список, в который входят данные замеров согласно Варианту 25
    temperature = [
        19, 29, 21, 39, 25, 26, 32, 25, 28, 26, 36, 30, 31,
        29, 35, 23, 32, 27, 27, 26, 26, 30, 27, 25, 28, 28,
        36, 29, 35, 26, 32, 29, 38, 28, 25, 29, 34, 28, 29,
        32, 34, 28, 28, 29, 33, 27, 34, 25, 28, 26, 30, 38,
        39, 32, 29, 29, 34, 35, 32, 27, 26, 25, 26, 35, 36,
        30, 28, 33, 26, 28, 26, 28, 27, 33, 33, 29, 32, 25,
        38, 26, 36, 23, 24, 27, 26, 30, 34, 25, 24, 33
    ]

    # Упорядочим ряд по возрастанию (создадим непрерывный вариационный ряд)
    variation_temperature = arrange_temperature(temperature)
    print("Непрерывный вариационный ряд: ", variation_temperature)

    # Найдем размах варьирования (вычтем первый элемент вариационного ряда из последнего).
    R = scope(variation_temperature)
    print("Размах варьирования признака R =", R)

    # Найдем число интервалов вариационного ряда по одному из соотношений
    k = number_intervals(temperature)
    print("Число интервалов вариационного ряда k = ", k)

    # Найдем длину частичных интервалов и округлим ее до одной цифры после запятой.
    h = length_intervals(R, k)
    print("Длина частичных интервалов h = ", h)

    # Запишем полученный интервальный вариационный ряд и получим списки частот и вариант.
    options_intervals, frequencies = interval_table(variation_temperature, k, h)

    # Получим из списка Вариант интервалов список середин интервалов для дискретного вариационного ряда
    middle_intervals = get_middle(options_intervals)

    # Выведем дискретный вариационный ряд, в качестве xi берем середины интервалов интервального вариационного ряда
    discrete_print(middle_intervals, frequencies)

    # Найдем моду среди списка середин интевалов
    moda = find_moda(middle_intervals, frequencies)

    # Составим в виде двумерного списка расчетную таблицу
    desired_table = calculation_table(k, frequencies, middle_intervals, moda, h)

    # После создания расчетной таблицы, по ней найдем условные начальные моменты от М1 до М4
    M1, M2, M3, M4 = calculation_moments(desired_table, temperature, k)

    # Найдем выборочную дисперсию
    S_2 = calculate_sample_variance(M1, M2, h)

    # Найдем выборочное среднее квадратичное отклонение
    S = calculate_rms(S_2)

    # Найдем среднеарифметическое в дискретном вариационном ряду
    x_middle = find_middle(middle_intervals, frequencies)
    print("Среднеарифметическое дискретного вариационного ряда: Хср = ", x_middle)

    # Создадим расчетную таблицу для теоретических частот
    theoretical_table, theoretical_frequencies = calculation_theoretical_table(middle_intervals, frequencies,
                                                                               x_middle, S, h)

    # Постоим эмпирическую и теоретическую кривые
    plot_frequency_polygon(middle_intervals, frequencies, theoretical_frequencies)

    # Проведем критерий Пирсона
    Xi_2 = pirson(theoretical_table, middle_intervals)

    # Проведем проверку распределения согласно критерию Колмагорова
    kolmagorov_k = kolmagorov(temperature, frequencies, theoretical_frequencies)

    # Проведем проверку близости распределения к нормальному согласно приближенному критерию

    # Найдем центральные моменты третьего и четвертого порядков для нахождения ассимметрии и эксцесса
    M_3, M_4 = calculate_central_moments(M1, M2, M3, M4, h)

    # Найдем ассимметрию и эксцесс, а также их среднеквадратические отклонения
    # Найдем коэффициент ассимметрии и эксцесса
    As, Ex = asymmetry_excess_calculate(M_3, M4, S)

    # Найдем среднеквадратические отклонения для асимметрии и эксцесса
    S_As, S_Ex = rsm_asymmetry_excess(temperature)

    # Проверим нормальность распределения приближенным критерием
    approximate_criterion(As, Ex, S_As, S_Ex)
