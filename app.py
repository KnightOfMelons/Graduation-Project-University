from flask import Flask, render_template, request
import matplotlib.pyplot as plt
import io
import base64
import numpy as np
from scipy.signal import lti, step
from scipy.interpolate import interp1d
import control as ctl

app = Flask(__name__)


@app.route("/")
def main_page():
    return render_template('index.html')


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
    # Number of data points
    a = len(y)

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

        coefficients_1 = [(C0_1[i], C1_1[i]) for i in range(len(C0_1))]
        coefficients_2 = [(C0_2[i], C1_2[i]) for i in range(len(C0_2))]
        coefficients_3 = [(C0_3[i], C1_3[i]) for i in range(len(C0_3))]

        return render_template("PID/PID_full_page_3.html",
                               first_image=first_image,
                               second_image=second_image,
                               third_image=third_image,
                               coefficients_1=coefficients_1,
                               coefficients_2=coefficients_2,
                               coefficients_3=coefficients_3,
                               C2_1=C2_1, C2_2=C2_2, C2_3=C2_3)

    return render_template("PID/PID_full_page_3.html")

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


app.run(debug=True)
