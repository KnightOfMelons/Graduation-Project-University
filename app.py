from flask import Flask, render_template, request
import matplotlib.pyplot as plt
import io
import base64
import numpy as np
from scipy.signal import lti, step, find_peaks
from scipy.interpolate import interp1d
import control as ctl
import math

app = Flask(__name__)


@app.route("/")
def main_page():
    return render_template('index.html')


def transient_process_using_a_regulator(T1, T2, K, C0, C1, C2):
    # Параметры системы
    numerator = [1]
    denominator = [T1, T2, 1]
    plant = ctl.TransferFunction(numerator, denominator)

    # Коэффициент усиления
    gain = K
    gain_block = ctl.TransferFunction([gain], [1])

    # ПИД-регулятор с заданными значениями
    Kp = C0
    Ki = C1
    Kd = C2
    N = 1  # Коэффициент фильтра

    # Идеальная форма ПИД-регулятора с фильтрацией деривационного члена
    pid = Kp * (1 + ctl.TransferFunction([Ki], [1, 0]) + ctl.TransferFunction([Kd * N, 0], [1, N]))

    # Добавление транспортной задержки с аппроксимацией Pade
    time_delay = 60
    num, den = ctl.pade(time_delay, 10)  # Используем более высокую аппроксимацию Pade
    delay_block = ctl.TransferFunction(num, den)

    # Формирование открытой системы с учетом ПИД-регулятора и усилителя
    open_loop = ctl.series(gain_block, plant, delay_block)

    # Замкнутая система с отрицательной обратной связью
    closed_loop = ctl.feedback(open_loop, pid, sign=-1)

    # Входное воздействие
    t = np.linspace(0, 2500, 10000, dtype=np.float64)  # Временной диапазон для лучшего соответствия
    u = np.ones_like(t, dtype=np.float64) * 30  # Ступенчатое воздействие с конечным значением 30

    # Ответ системы
    t, y = ctl.forced_response(closed_loop, t, u)

    # Вычисление интеграла от квадрата сигнала управления
    u_response = np.abs(y) ** 2
    integral = np.trapz(u_response, t)

    # Возвращение результатов в экспоненциальной нотации
    integral_formatted = "{:.2e}".format(integral)

    return t, y, integral_formatted


def calculate_integrals(T1, T2, K, C0_list, C1_list, C2):
    integrals = []
    for C0, C1 in zip(C0_list, C1_list):
        _, _, integral_formatted = transient_process_using_a_regulator(T1, T2, K, C0, C1, C2)
        integrals.append(integral_formatted)
    return integrals


def plot_system_response(t, y):
    # Create a new figure and axis
    fig, ax = plt.subplots()

    # Plot the graph
    ax.plot(t, y, 'y')
    plt.title('Переходной процесс с использованием регулятора')
    plt.grid(True)

    # Find peaks
    peaks, _ = find_peaks(y)

    # Identify the highest peak (A6)
    A6_index = np.argmax(y)
    # A6_time = t[A6_index]
    A6_value = y[A6_index]

    # Find the local minima
    minima_indices = (np.diff(np.sign(np.diff(y))) > 0).nonzero()[0] + 1

    # Identify the first significant local minimum after the highest peak (A12)
    minima_after_A6 = [i for i in minima_indices if i > A6_index]
    if minima_after_A6:
        A12_index = minima_after_A6[0]
        A12_value = y[A12_index]
    else:
        A12_index = None
        A12_value = None

    # Identify the next peak after this local minimum (A7)
    peaks_after_A12 = [i for i in peaks if i > A12_index]
    if peaks_after_A12:
        A7_index = peaks_after_A12[0]
        A7_value = y[A7_index]
    else:
        A7_index = None
        A7_value = None

    # Annotate the points A1 (A6), A2 (A12), and A3 (A7)
    if A6_index is not None:
        ax.plot(t[A6_index], y[A6_index], "ro")
        ax.annotate('A1', (t[A6_index], y[A6_index]), textcoords="offset points", xytext=(0, 10), ha='center',
                    color='brown')
    if A12_index is not None:
        ax.plot(t[A12_index], y[A12_index], "ro")
        ax.annotate('A2', (t[A12_index], y[A12_index]), textcoords="offset points", xytext=(0, 10), ha='center',
                    color='brown')
    if A7_index is not None:
        ax.plot(t[A7_index], y[A7_index], "ro")
        ax.annotate('A3', (t[A7_index], y[A7_index]), textcoords="offset points", xytext=(0, 10), ha='center',
                    color='brown')

    # Find the value of the first maximum y_M
    y_M = max(y)
    # t_M_index = np.argmax(y)
    # t_M = t[t_M_index]

    # Find the settling time t_p
    threshold = 0.02 * y_M  # Threshold for determining the settled value
    t_p_index = next(i for i in range(len(y) - 1, 0, -1) if y[i] > threshold)
    t_p = t[t_p_index]

    # Save the plot to a buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    # Encode the buffer to a base64 string
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    # Generate HTML code to display the image
    image_html = '<img src="data:image/png;base64,{}">'.format(image_base64)

    # Return the HTML code and data
    return image_html, A6_value, A12_value, A7_value, t_p


def compute_and_plot(T1, T2, K, C2):
    w = np.arange(0, 0.02, 0.0001)
    m = 0.3
    s = (1j - m) * w
    Wo = (K * np.exp(-60 * s)) / (T1 * (s ** 2) + T2 * s + 1)
    Wr = 1 / Wo
    R = np.real(Wr)
    J = np.imag(Wr)

    Kp = m * J - R + 2 * m * w * C2
    Ki = w * m * (m ** 2 + 1) * (J + w * C2)

    # Найти максимальное значение Kp и соответствующее значение Ki
    max_Kp_index = np.argmax(Kp)
    max_Kp = Kp[max_Kp_index]
    max_Ki = Ki[max_Kp_index]

    # Определить точки C0 и C1 на определенных процентных уровнях от максимального значения Kp
    percentage_levels = [0.95, 0.9, 0.85, 0.8]  # Процентные уровни для нахождения точек
    C0_values = []
    C1_values = []

    for level in percentage_levels:
        target_Kp = max_Kp * level
        # Найти индекс ближайшего значения Kp к target_Kp
        closest_index = (np.abs(Kp - target_Kp)).argmin()
        C0_values.append(Kp[closest_index])
        C1_values.append(Ki[closest_index])

    # Plot the results
    fig, ax = plt.subplots()
    ax.plot(Kp, Ki, label=f'C2 = {C2}')
    plt.xlabel('axis Kp')
    plt.ylabel('axis Ki')
    plt.title(f'Плоскость параметров настройки ПИД регулятора при С_2 = {C2}')
    plt.legend()
    plt.grid(True)

    # Save plot to a buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    # Encode buffer to base64 string
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    # Generate HTML code to display the image
    image_result = '<img src="data:image/png;base64,{}">'.format(image_base64)

    return image_result, C0_values, C1_values


# Это функция для определения тремя методами передаточных функций
def transfer_functions_determined_by_three_methods(t1_kos=40,
                                                   t2_kos=240,
                                                   t1_aldenb=199,
                                                   t2_aldenb=166,
                                                   t1_simou=131.71,
                                                   t2_simou=7436.27,
                                                   k=3.5,
                                                   stop_time=2500,
                                                   y_values=None):
    if y_values is None:
        y_values = [0, 1, 3, 8, 18, 30, 44, 59, 73, 85, 95, 101, 104, 105, 105]

    def find_values_at_points(t, y, x_points):
        y_values = np.interp(x_points, t, y)
        return y_values

    # Define the transfer functions
    G1 = ctl.TransferFunction([1], [int(t1_kos / 5), 1])
    G2 = ctl.TransferFunction([1], [int(t2_kos / 5), 1])
    G3 = ctl.TransferFunction([1], [int(t1_aldenb / 5), 1])
    G4 = ctl.TransferFunction([1], [int(t2_aldenb / 5), 1])
    G5 = ctl.TransferFunction([1], [int(t2_simou), int(t1_simou), 1])

    # Connect the blocks
    sys1 = k * G1 * G2
    sys2 = k * G3 * G4
    sys3 = k * G5

    # Define the time vector
    t = np.linspace(0, stop_time, num=5000)

    # Define the step input with time delay
    step_input = np.zeros_like(t)
    step_input[t >= 60] = 30  # Apply step input of 30 after 60 seconds

    # Simulate the step responses
    t1, y1 = ctl.forced_response(sys1, T=t, U=step_input)
    t2, y2 = ctl.forced_response(sys2, T=t, U=step_input)
    t3, y3 = ctl.forced_response(sys3, T=t, U=step_input)

    # Define the points where we want to find the first values
    x_points = [0, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320]

    # Find values at the specified points
    y1_values = find_values_at_points(t1, y1, x_points)
    y2_values = find_values_at_points(t2, y2, x_points)
    y3_values = find_values_at_points(t3, y3, x_points)

    # Calculate the sum of squared differences (dispersion) for each method
    dispersion_y1 = sum((yv - y1v) ** 2 for yv, y1v in zip(y_values, y1_values))
    dispersion_y2 = sum((yv - y2v) ** 2 for yv, y2v in zip(y_values, y2_values))
    dispersion_y3 = sum((yv - y3v) ** 2 for yv, y3v in zip(y_values, y3_values))

    answer_1_kasatelnaia = 'метода касательной.'
    answer_2_oldenburg_sartorius = 'метода Ольденбурга-Сарториуса.'
    answer_3_simou = 'метода Симою.'

    # Determine which dispersion is the largest
    if dispersion_y1 < dispersion_y2 and dispersion_y1 < dispersion_y3:
        chosen_answer = answer_1_kasatelnaia
        nums_for_answer = [t1_kos, t2_kos]
    elif dispersion_y2 < dispersion_y1 and dispersion_y2 < dispersion_y3:
        chosen_answer = answer_2_oldenburg_sartorius
        nums_for_answer = [t1_aldenb, t2_aldenb]
    else:
        chosen_answer = answer_3_simou
        nums_for_answer = [t1_simou, t2_simou]

    # Plot the results
    fig, ax = plt.subplots()
    ax.plot(t1, y1, label='Касательной')
    ax.plot(t2, y2, label='Ольденбурга-Сарториуса')
    ax.plot(t3, y3, label='Симою')
    plt.title('Переходные характеристики трёх моделей')
    plt.legend()
    plt.grid(True)

    # Сохраняем график в буфер
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    # Кодируем буфер в base64 строку
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    # Генерируем HTML-код для отображения картинки
    image_result = '<img src="data:image/png;base64,{}">'.format(image_base64)

    return image_result, y1_values, y2_values, y3_values, dispersion_y1, dispersion_y2, dispersion_y3, chosen_answer, nums_for_answer


# Функция для вычисления трёх методов Симою
def simou_method(y, t, input_K=1):
    # # Number of data points
    # a = len(y)

    # Calculation of e
    e = 1 - y / y[-1]

    # Calculation of t1
    t1 = (t[1] - t[0]) * (np.sum(e) - 0.5 * (1 - y[0] / y[-1]))

    # Calculation of f, g, h, j
    f = t / t1
    g = 1 - f
    h = e * g
    j = e * (1 - 2 * f + f ** 2 / 2)

    # Calculation of t2 and t3
    t2 = t1 ** 2 * (g[0] - g[1]) * (np.sum(h) - 0.5 * (1 - y[0] / y[-1]))
    t3 = t1 ** 3 * (g[0] - g[1]) * (np.sum(j) - 0.5 * (1 - y[0] / y[-1]))

    time_points = np.linspace(0, 260, 500)

    # Interpolate the original curve to extend it
    interpolator = interp1d(t, y, kind='linear', fill_value='extrapolate')
    extended_y = interpolator(time_points)

    # Define the transfer function for 1st, 2nd, and 3rd order systems
    num = [y[-1]]  # Numerator for all systems
    input_K = num[0] / input_K

    # Denominators based on the orders
    den_1st = [t1, 1]
    den_2nd = [t2, t1, 1]
    den_3rd = [t3, t2, t1, 1]

    # Calculate the step responses
    system_1st = lti(num, den_1st)
    system_2nd = lti(num, den_2nd)
    system_3rd = lti(num, den_3rd)

    _, response_1st = step(system_1st, T=time_points)
    _, response_2nd = step(system_2nd, T=time_points)
    _, response_3rd = step(system_3rd, T=time_points)

    # Создаем график
    plt.figure(figsize=(8, 4))
    plt.plot(np.linspace(0, 260, 500), extended_y, 'r', label='Исходная кривая')
    plt.plot(np.linspace(0, 260, 500), response_1st, 'g--', label='Симою 1го порядка')
    plt.plot(np.linspace(0, 260, 500), response_2nd, 'b--', label='Симою 2го порядка')
    plt.plot(np.linspace(0, 260, 500), response_3rd, 'k--', label='Симою 3го порядка')

    plt.xlabel('Время')
    plt.ylabel('Переменная')
    plt.title('Метод Симою')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.xticks(np.arange(0, max(t) + 1, 20))
    plt.yticks(np.arange(0, max(y) + 1, 10))
    plt.minorticks_on()
    plt.grid(which='minor', linestyle=':', linewidth=0.5)

    # Сохраняем график в буфер
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    # Кодируем буфер в base64 строку
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    # Генерируем HTML-код для отображения картинки
    image_simou = '<img src="data:image/png;base64,{}">'.format(image_base64)

    return t1, t2, t3, input_K, extended_y, response_1st, response_2nd, response_3rd, image_simou


# Эта функция для построения разгонной характеристики
def acceleration_characteristic(x_values, y_values):
    z_values = []
    cumulative_sum = 0
    for y in y_values:
        cumulative_sum += y
        z_values.append(cumulative_sum)

    # Создаем новую фигуру и оси
    fig, ax = plt.subplots()

    # Строим график
    ax.plot(x_values, z_values, marker='o')
    plt.xlabel('T')
    plt.ylabel('Ø')
    plt.title('Разгонная характеристика объекта')
    plt.grid(True)

    # Сохраняем график в буфер
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    # Кодируем буфер в base64 строку
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    # Генерируем HTML-код для отображения картинки
    image_acceleration_charact = '<img src="data:image/png;base64,{}">'.format(image_base64)

    # Возвращаем HTML-код и данные
    return image_acceleration_charact, z_values


# Функция для вычисления площади
def calculate_total_area(x_values, y_values):
    total_area = 0

    for i in range(len(x_values) - 1):
        width = x_values[i + 1] - x_values[i]
        height1 = y_values[i]
        height2 = y_values[i + 1]
        area = 0.5 * (height1 + height2) * width
        total_area += area

    return total_area


def find_first_positive_index(y_values, x_values):
    for index, value in enumerate(y_values):
        if value > 0:
            return x_values[index]
    return None


# Фукнция построения импульсной переходной характеристики
def impulse_transient_plot(x_values, y_values):
    # Создаем новую фигуру и оси
    fig, ax = plt.subplots()

    # Строим график
    ax.plot(x_values, y_values, marker='o')
    plt.xlabel('T')
    plt.ylabel('Ø')
    plt.title('Импульсная переходная характеристика')
    plt.grid(True)

    # Сохраняем график в буфер
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    # Кодируем буфер в base64 строку
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    # Генерируем HTML-код для отображения картинки
    image_impulse = '<img src="data:image/png;base64,{}">'.format(image_base64)

    return image_impulse


# Функция для передаточной фукнции мат модели граф. метода
def transmission_function_for_math_model(k=22.9, T2=1712.0, T1=126.4, t_stop=2000):
    # Define the transfer function
    num = [k]
    den = [T2, T1, 1]
    sys = ctl.TransferFunction(num, den)

    # Define the transport delay parameters
    time_delay = 30
    # initial_output = 0
    # initial_buffer_size = 1024

    # Create time vector
    t = np.linspace(0, t_stop, 5000)

    # Create input step function with delay
    u = np.zeros_like(t)
    u[int(time_delay / (t_stop / len(t))):] = 1

    # Simulate the system response
    t_out, y_out = ctl.forced_response(sys, T=t, U=u)

    # Create a new figure and axes
    fig, ax = plt.subplots()

    # Plot the results
    ax.plot(t_out, y_out, 'y', linewidth=1.5)
    plt.title('Передаточная функция объекта управления')
    plt.grid(True)

    # Save the plot to a buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    # Encode the buffer to base64 string
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    # Generate HTML code for displaying the image
    image_html = '<img src="data:image/png;base64,{}">'.format(image_base64)

    return image_html


def generate_system_response(k=22.9, T2=1712.0, T1=126.4):
    # Define the transfer function
    m = 0.192
    w = np.linspace(0, 0.18, 100)  # диапазон частоты от 0 до 0.18, 100 точек
    p = -m * w + 1j * w  # комплексная частота

    # Задаем ПФ объекта
    Wo = k / (T2 * p**2 + T1 * p + 1) * np.exp(-30 * p)  # задаем ПФ

    # инверсная ПФ
    W1 = 1 / Wo

    C0 = w * (1 + m**2) * np.imag(W1)
    C1 = -np.real(W1) + m * np.imag(W1)

    # Ограничение области графика
    x_min, x_max = 0, 0.2
    y_min, y_max = 0, 0.002

    # Фильтруем данные в пределах области графика
    valid_indices = (C1 >= x_min) & (C1 <= x_max) & (C0 >= y_min) & (C0 <= y_max)
    filtered_C1 = C1[valid_indices]
    filtered_C0 = C0[valid_indices]

    # Найти точку максимума в пределах области
    max_index = np.argmax(filtered_C0)

    # Найти дополнительные точки правее максимума
    num_points = 5
    points_indices = np.arange(max_index, max_index + num_points + 1)
    points_indices = points_indices[(points_indices >= 0) & (points_indices < len(filtered_C1))]

    points_C1 = filtered_C1[points_indices]
    points_C0 = filtered_C0[points_indices]

    # Создаем список коэффициентов с интегралами
    coefficients_with_integrals = []
    for C1_val, C0_val in zip(points_C0, points_C1):
        integral = calculate_integral(T1, T2, k, round(C0_val, 6), round(C1_val, 6), True)
        coefficients_with_integrals.append((C0_val, C1_val, integral))

    # Найти наименьший интеграл
    min_integral = min(coefficients_with_integrals, key=lambda x: x[2])
    min_C0, min_C1, _ = min_integral

    # Создаем новую фигуру и оси
    fig, ax = plt.subplots()

    # Строим график
    ax.plot(C1, C0)
    ax.scatter(points_C1, points_C0, color='red', zorder=5, label='Выбранные точки')  # Отметить точки на графике
    ax.axis((0, 0.2, 0, 0.002))
    ax.set_title('Кривая равной степени колебательности')
    ax.grid(True)
    ax.legend()

    # Сохраняем график в буфер
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    # Кодируем буфер в base64 строку
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    # Генерируем HTML-код для отображения картинки
    image_html = '<img src="data:image/png;base64,{}">'.format(image_base64)

    return image_html, coefficients_with_integrals, min_C0, min_C1



def calculate_integral(T1, T2, K, C0, C1, choise_plot_or_not=True):
    # Параметры системы
    numerator = [1]
    denominator = [T2, T1, 1]
    plant = ctl.TransferFunction(numerator, denominator)

    # Коэффициент усиления
    gain = K
    gain_block = ctl.TransferFunction([gain], [1])

    # ПИ-регулятор с заданными значениями в параллельной форме
    Kp = C0
    Ki = C1
    pi_controller = ctl.TransferFunction([Kp, Ki], [1, 0])

    # Добавление транспортной задержки с аппроксимацией Pade
    time_delay = 30
    num, den = ctl.pade(time_delay, 10)
    delay_block = ctl.TransferFunction(num, den)

    # Формирование открытой системы с учетом ПИ-регулятора и усилителя
    open_loop = ctl.series(gain_block, plant, delay_block)

    # Замкнутая система с отрицательной обратной связью
    closed_loop = ctl.feedback(open_loop, pi_controller, sign=-1)

    # Входное воздействие
    t = np.linspace(0, 2000, 10000, dtype=np.float64)
    u = np.ones_like(t, dtype=np.float64) * 1

    # Ответ системы
    t, y = ctl.forced_response(closed_loop, t, u)

    # Вычисление интеграла от квадрата сигнала управления
    u_response = np.abs(y) ** 2
    integral = np.trapz(u_response, t)

    if choise_plot_or_not == True:
        return integral
    else:
        return t,y


def plot_response_pi_controller(t, y):
    # Создаем новую фигуру и оси
    fig, ax = plt.subplots(figsize=(8, 5))

    # Строим график
    ax.plot(t, y, 'y')
    ax.set_title('Step Response with PI Controller and Feedback')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Response')
    ax.grid(True)

    # Сохраняем график в буфер
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    # Кодируем буфер в base64 строку
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    # Генерируем HTML-код для отображения картинки
    image_html = '<img src="data:image/png;base64,{}">'.format(image_base64)

    return image_html
# ======================================================================================================
# =============== Одноконтурная АСР с ПИД-регулятором и всё, что к ней относится =======================
# ======================================================================================================
@app.route("/PID_part_1", methods=["GET", "POST"])
def function_of_main_pid_page_first():
    if request.method == "POST":
        # ===== Беру значения с сайта =====
        x_values = [float(request.form[f"xValues_{i}"]) for i in range(17)]
        y_values = [float(request.form[f"yValues_{i}"]) for i in range(17)]

        # ===== Беру также значения с сайта =====
        tau = float(request.form["tau"])
        x_values_imp = float(request.form["x_values_imp"])
        at = float(request.form["at"])
        a_theta = float(request.form["a_theta"])
        M = float(request.form["M"])
        input_K_from_input = int(request.form["input_signal"])

        # Вычисляю Ta, k, τ, Fλ, Fσ
        Fl = (tau * at)
        o = max(y_values)
        Ta = round(Fl / max(y_values), 1)
        Fo = calculate_total_area(x_values, y_values)
        k = round(Fo / Fl, 1)
        t = find_first_positive_index(y_values, x_values)

        # ====================== Тут создается импульсная переходная характеристика ===================================

        image_impulse = impulse_transient_plot(x_values, y_values)

        # ====================================== Разгонная характеристика =============================================

        image_acceleration_charact, z_values = acceleration_characteristic(x_values, y_values)

        # =============================================== Метод Симою ================================================

        # Тут я пытаюсь найти первый положительный индекс в z_values - 1,
        # чтобы с нулём было в начале (если у нас 0 0 0 1 3 8, то он найдет 1, но мне также нужен ноль перед этим)
        first_positive_index = next((index for index, value in enumerate(z_values) if value > 0), None) - 1
        # Тут я делаю новый список со значениями по индексу, который нашел сверху
        z_values_for_simou = z_values[first_positive_index:]
        # убираю ненужную часть (в конце там 104 105 105, вторая 105 не нужна)
        z_values_for_simou = z_values_for_simou[:-1]
        # Вызов функции с присвоением значений
        t1, t2, t3, input_K, extended_y, response_1st, response_2nd, response_3rd, image_simou = simou_method(
            np.array(z_values_for_simou),
            np.array(x_values[:len(z_values_for_simou)]),
            input_K_from_input)

        # ======================================= ОБЩИЙ ВЫВОД НА САЙТ =================================================
        return render_template('PID/PID_full_page_1.html',
                               image_impulse=image_impulse,
                               numbers=[tau, x_values_imp, at, a_theta, M, Ta, k, t, o, Fl, Fo],
                               image_acceleration_charact=image_acceleration_charact,
                               numbers_simou=[round(t1, 2), round(t2, 2), round(t3, 2), round(input_K, 1)],
                               image_simou=image_simou)

    return render_template("PID/PID_full_page_1.html")


@app.route("/PID_part_2", methods=["GET", "POST"])
def function_of_main_pid_second():
    if request.method == "POST":
        # ======================== Передаточные фукнции, определенные тремя методами ===================================
        t1_kos = int(request.form['t1_kos'])
        t2_kos = int(request.form['t2_kos'])
        t1_aldenb = int(request.form['t1_aldenb'])
        t2_aldenb = int(request.form['t2_aldenb'])
        t1_simou = float(request.form['t1_simou'])
        t2_simou = float(request.form['t2_simou'])
        k = float(request.form['k'])
        stop_time = int(request.form['stop_time'])
        y_values = [float(request.form[f"yValues_{i}"]) for i in range(15)]

        (image_three_methods,
         y1_values,
         y2_values,
         y3_values,
         dispersion_y1,
         dispersion_y2,
         dispersion_y3,
         answer,
         answer_nums) = transfer_functions_determined_by_three_methods(
            t1_kos,
            t2_kos,
            t1_aldenb,
            t2_aldenb,
            t1_simou,
            t2_simou,
            k,
            stop_time,
            y_values, )

        # Вычисляю данные для 0, 60, 120, 180, 320 по каждому методу на графике.
        y1_values = [round(y, 4) for y in y1_values]
        y2_values = [round(y, 4) for y in y2_values]
        y3_values = [round(y, 4) for y in y3_values]

        return render_template('PID/PID_full_page_2.html',
                               image_three_methods=image_three_methods,
                               y_values=[y1_values, y2_values, y3_values],
                               dispersions=[dispersion_y1, dispersion_y2, dispersion_y3],
                               x_points=[0, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320],
                               answer=answer,
                               answer_nums=answer_nums)
    return render_template("PID/PID_full_page_2.html")


@app.route('/PID_part_3', methods=['GET', 'POST'])
def function_of_main_pid_third():
    if request.method == 'POST':
        T1 = float(request.form['t1_1'])
        T2 = float(request.form['t2_2'])
        K = float(request.form['k'])
        C2_1 = float(request.form['c2_1'])
        C2_2 = float(request.form['c2_2'])
        C2_3 = float(request.form['c2_3'])

        first_image, C0_1, C1_1 = compute_and_plot(T1, T2, K, C2_1)
        second_image, C0_2, C1_2 = compute_and_plot(T1, T2, K, C2_2)
        third_image, C0_3, C1_3 = compute_and_plot(T1, T2, K, C2_3)

        integrals_1 = calculate_integrals(T1, T2, K, C0_1, C1_1, C2_1)
        integrals_2 = calculate_integrals(T1, T2, K, C0_2, C1_2, C2_2)
        integrals_3 = calculate_integrals(T1, T2, K, C0_3, C1_3, C2_3)

        coefficients_1 = [(C0_1[i], C1_1[i], integrals_1[i]) for i in range(len(C0_1))]
        coefficients_2 = [(C0_2[i], C1_2[i], integrals_2[i]) for i in range(len(C0_2))]
        coefficients_3 = [(C0_3[i], C1_3[i], integrals_3[i]) for i in range(len(C0_3))]

        # Найти минимальный интеграл и соответствующие коэффициенты
        all_integrals = integrals_1 + integrals_2 + integrals_3
        all_coefficients = coefficients_1 + coefficients_2 + coefficients_3
        min_integral_index = all_integrals.index(min(all_integrals, key=lambda x: float(x)))
        min_C0, min_C1, _ = all_coefficients[min_integral_index]
        min_C2 = [C2_1, C2_2, C2_3][min_integral_index // len(C0_1)]

        # Построить график для минимального интеграла
        t, y, _ = transient_process_using_a_regulator(T1, T2, K, min_C0, min_C1, min_C2)
        min_integral_image, A1_value, A2_value, A3_value, t_p = plot_system_response(t, y)

        first_calculation = (A2_value / A1_value) * 100
        second_calculation = 1 - (A3_value / A1_value)

        print(f"T1 = {T1}, T2 = {T2}, K = {K},"
              f" min_C0 = {min_C0},"
              f" min_C1 =  {min_C1},"
              f" min_C2 = {min_C2},"
              f"t = {t}, y = {y}")

        return render_template("PID/PID_full_page_3.html",
                               first_image=first_image,
                               second_image=second_image,
                               third_image=third_image,
                               coefficients_1=coefficients_1,
                               coefficients_2=coefficients_2,
                               coefficients_3=coefficients_3,
                               C2_1=C2_1, C2_2=C2_2, C2_3=C2_3,
                               min_integral_image=min_integral_image,
                               min_C0=min_C0, min_C1=min_C1, min_C2=min_C2,
                               A1_value=A1_value, A2_value=A2_value, A3_value=A3_value,
                               t_M=A1_value, t_p=t_p,
                               first_calculation=first_calculation, second_calculation=second_calculation)

    return render_template("PID/PID_full_page_3.html")


# =============== Отдельная страница для построения импульсной и разгонной характеристики ==============
@app.route('/pulse_and_acceleration_characteristics', methods=['GET', 'POST'])
def function_of_pulse_and_acceleration_characteristics():
    if request.method == "POST":
        # ===== Беру значения с сайта =====
        x_values = [float(request.form[f"xValues_{i}"]) for i in range(17)]
        y_values = [float(request.form[f"yValues_{i}"]) for i in range(17)]

        # ===== Беру также значения с сайта =====
        tau = float(request.form["tau"])
        x_values_imp = float(request.form["x_values_imp"])
        at = float(request.form["at"])
        a_theta = float(request.form["a_theta"])
        M = float(request.form["M"])

        # Вычисляю Ta, k, τ, Fλ, Fσ
        Fl = (tau * at)
        o = max(y_values)
        Ta = round(Fl / max(y_values), 1)
        Fo = calculate_total_area(x_values, y_values)
        k = round(Fo / Fl, 1)
        t = find_first_positive_index(y_values, x_values)

        # ====================== Тут создается импульсная переходная характеристика =================================

        image_impulse = impulse_transient_plot(x_values, y_values)

        # ====================================== Разгонная характеристика ===========================================

        image_acceleration_charact, z_values = acceleration_characteristic(x_values, y_values)

        return render_template('PID/Separate/pulse_and_acceleration_characteristics.html',
                               image_impulse=image_impulse,
                               numbers=[tau, x_values_imp, at, a_theta, M, Ta, k, t, o, Fl, Fo],
                               image_acceleration_charact=image_acceleration_charact)

    return render_template("PID/Separate/pulse_and_acceleration_characteristics.html")


# =============== Отдельная страница для переходной характеристики трёх моделей ========================
@app.route('/step_response_of_three_models', methods=['GET', 'POST'])
def function_of_step_response_of_three_models():
    return render_template("PID/Separate/step_response_of_three_models.html")


# ==================== Отдельная страница для построения плоскости параметров ===========================
@app.route('/construction_of_parameter_plane', methods=['GET', 'POST'])
def function_of_construction_of_parameter_plane():
    if request.method == 'POST':
        T1 = float(request.form['t1_1'])
        T2 = float(request.form['t2_2'])
        K = float(request.form['k'])
        C2_1 = float(request.form['c2_1'])
        C2_2 = float(request.form['c2_2'])
        C2_3 = float(request.form['c2_3'])

        first_image, C0_1, C1_1 = compute_and_plot(T1, T2, K, C2_1)
        second_image, C0_2, C1_2 = compute_and_plot(T1, T2, K, C2_2)
        third_image, C0_3, C1_3 = compute_and_plot(T1, T2, K, C2_3)

        integrals_1 = calculate_integrals(T1, T2, K, C0_1, C1_1, C2_1)
        integrals_2 = calculate_integrals(T1, T2, K, C0_2, C1_2, C2_2)
        integrals_3 = calculate_integrals(T1, T2, K, C0_3, C1_3, C2_3)

        coefficients_1 = [(C0_1[i], C1_1[i], integrals_1[i]) for i in range(len(C0_1))]
        coefficients_2 = [(C0_2[i], C1_2[i], integrals_2[i]) for i in range(len(C0_2))]
        coefficients_3 = [(C0_3[i], C1_3[i], integrals_3[i]) for i in range(len(C0_3))]

        return render_template("PID/Separate/construction_of_parameter_plane.html",
                               first_image=first_image,
                               second_image=second_image,
                               third_image=third_image,
                               coefficients_1=coefficients_1,
                               coefficients_2=coefficients_2,
                               coefficients_3=coefficients_3,
                               C2_1=C2_1, C2_2=C2_2, C2_3=C2_3)

    return render_template("PID/Separate/construction_of_parameter_plane.html")


# ================ Отдельная страница для наиболее идентичного переходного процесса =====================
@app.route('/most_identical_transient_process', methods=['GET', 'POST'])
def function_of_most_identical_transient_process():
    # Переделать, не работает
    if request.method == 'POST':
        T1 = float(request.form['t1_1'])
        T2 = float(request.form['t2_2'])
        K = float(request.form['k'])
        min_C0 = float(request.form['min_C0'])
        min_C1 = float(request.form['min_C1'])
        min_C2 = float(request.form['min_C2'])

        # Построить график для минимального интеграла
        t, y, _ = transient_process_using_a_regulator(T1, T2, K, min_C0, min_C1, min_C2)
        min_integral_image, A1_value, A2_value, A3_value, t_p = plot_system_response(t, y)

        first_calculation = (A2_value / A1_value) * 100
        second_calculation = 1 - (A3_value / A1_value)

        return render_template("PID/Separate/most_identical_transient_process.html",
                               min_integral_image=min_integral_image,
                               min_C0=min_C0, min_C1=min_C1, min_C2=min_C2,
                               A1_value=A1_value, A2_value=A2_value, A3_value=A3_value,
                               t_M=A1_value, t_p=t_p,
                               first_calculation=first_calculation, second_calculation=second_calculation)

    return render_template("PID/Separate/most_identical_transient_process.html")


# ======================================================================================================
# ========================================== Метод Симою ===============================================
# ======================================================================================================

@app.route('/SIMOU', methods=["GET", "POST"])
def method_simou():
    if request.method == "POST":
        x_values = np.array([float(request.form[f"xValues_{i}"]) for i in range(14)])
        y_values = np.array([float(request.form[f"yValues_{i}"]) for i in range(14)])
        input_K = int(request.form['input_signal'])

        t1, t2, t3, input_K, extended_y, response_1st, response_2nd, response_3rd, image_simou = simou_method(y_values,
                                                                                                              x_values,
                                                                                                              input_K)

        return render_template('simou_methods.html',
                               image_simou=image_simou,
                               numbers=[round(t1, 2), round(t2, 2), round(t3, 2), round(input_K, 1)])

    return render_template('simou_methods.html')


# ======================================================================================================
# =========================== Мат. модели графоаналитическим методом ===================================
# ======================================================================================================
@app.route('/PI_first_page', methods=["GET", "POST"])
def function_of_main_pi_page_first():
    if request.method == "POST":
        T2_value = float(request.form['T2_value'])
        T1_value = float(request.form['T1_value'])
        t_value = float(request.form['t_value'])
        k_value = float(request.form['k_value'])
        y_value = float(request.form['y_value'])
        stop_time = int(request.form['stop_time'])
        m = -math.log(1 - y_value) / (2 * math.pi)

        image_transmission_funct = transmission_function_for_math_model(k_value, T2_value, T1_value, stop_time)
        image_D_graph, coefficients_with_integrals, min_C0, min_C1 = generate_system_response(k_value, T2_value, T1_value)
        t,y = calculate_integral(T1_value, T2_value, k_value, round(min_C0, 6), round(min_C1, 6), False)
        image_transmission_funct_PI_controller, A1_value, A2_value, A3_value, t_p = plot_system_response(t,y)

        first_calculation = (A2_value / A1_value) * 100
        second_calculation = 1 - (A3_value / A1_value)

        return render_template('PI/PI_full_page.html',
                               image_transmission_funct=image_transmission_funct,
                               y_value=y_value,
                               m=round(m, 3),
                               image_D_graph=image_D_graph,
                               coefficients_with_integrals=coefficients_with_integrals,
                               min_C0=min_C0,
                               min_C1=min_C1,
                               image_transmission_funct_PI_controller=image_transmission_funct_PI_controller,
                               A1_value=A1_value, A2_value=A2_value, A3_value=A3_value,
                               t_M=A1_value, t_p=t_p,
                               first_calculation=first_calculation, second_calculation=second_calculation)

    return render_template('PI/PI_full_page.html')


# ==================== Отдельная страница для передаточной функции объекта ===========================
@app.route('/transfer_function_of_the_object', methods=['GET', 'POST'])
def functuion_of_transfer_function_of_the_object():
    if request.method == "POST":
        T2_value = float(request.form['T2_value'])
        T1_value = float(request.form['T1_value'])
        t_value = float(request.form['t_value'])
        k_value = float(request.form['k_value'])
        y_value = float(request.form['y_value'])
        stop_time = int(request.form['stop_time'])
        m = -math.log(1 - y_value) / (2 * math.pi)

        image_transmission_funct = transmission_function_for_math_model(k_value, T2_value, T1_value, stop_time)

        return render_template('PI/Separate/transfer_function_of_the_object.html',
                               image_transmission_funct=image_transmission_funct,
                               y_value=y_value,
                               m=round(m, 3))

    return render_template('PI/Separate/transfer_function_of_the_object.html')


# ==================== Отдельная страница для кривой равной степени колебательности ===================
@app.route('/equal_degree_of_vibration_curve', methods=['GET','POST'])
def function_of_equal_degree_of_vibration_curve():
    if request.method == "POST":
        T2_value = float(request.form['T2_value'])
        T1_value = float(request.form['T1_value'])
        t_value = float(request.form['t_value'])
        k_value = float(request.form['k_value'])
        y_value = float(request.form['y_value'])
        stop_time = int(request.form['stop_time'])
        m = -math.log(1 - y_value) / (2 * math.pi)

        image_D_graph, coefficients_with_integrals, _, _ = generate_system_response(k_value, T2_value, T1_value)

        return render_template('PI/PI_full_page.html',
                               image_D_graph=image_D_graph,
                               coefficients_with_integrals=coefficients_with_integrals,)

    return render_template('PI/Separate/equal_degree_of_vibration_curve.html')


# ==================== Отдельная страница для переходного процесса с использованием ПИ-регулятора ===================
@app.route('/transient_process_using_a_PI_controller', methods=['GET', 'POST'])
def function_of_transient_process_using_a_PI_controller():
    if request.method == "POST":
        T2_value = float(request.form['T2_value'])
        T1_value = float(request.form['T1_value'])
        t_value = float(request.form['t_value'])
        k_value = float(request.form['k_value'])
        y_value = float(request.form['y_value'])
        stop_time = int(request.form['stop_time'])
        m = -math.log(1 - y_value) / (2 * math.pi)

        _, _, min_C0, min_C1 = generate_system_response(k_value, T2_value, T1_value)
        t,y = calculate_integral(T1_value, T2_value, k_value, round(min_C0, 6), round(min_C1, 6), False)
        image_transmission_funct_PI_controller, A1_value, A2_value, A3_value, t_p = plot_system_response(t,y)

        first_calculation = (A2_value / A1_value) * 100
        second_calculation = 1 - (A3_value / A1_value)

        return render_template('PI/Separate/transient_process_using_a_PI_controller.html',
                               min_C0=min_C0,
                               min_C1=min_C1,
                               image_transmission_funct_PI_controller=image_transmission_funct_PI_controller,
                               A1_value=A1_value, A2_value=A2_value, A3_value=A3_value,
                               t_M=A1_value, t_p=t_p,
                               first_calculation=first_calculation, second_calculation=second_calculation)

    return render_template('PI/Separate/transient_process_using_a_PI_controller.html')
app.run(debug=True)
