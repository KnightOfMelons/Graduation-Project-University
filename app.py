from flask import Flask, render_template, request
import matplotlib.pyplot as plt
import io
import base64
import numpy as np
from scipy.signal import lti, step
from scipy.interpolate import interp1d

app = Flask(__name__)


@app.route("/")
def main_page():
    return render_template('index.html')


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

    return t1, t2, t3, input_K, extended_y, response_1st, response_2nd, response_3rd


@app.route("/PID", methods=["GET", "POST"])
def plot_impulse_transient_response():
    if request.method == "POST":
        x_values = [float(request.form[f"xValues_{i}"]) for i in range(17)]
        y_values = [float(request.form[f"yValues_{i}"]) for i in range(17)]

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

        tau = float(request.form["tau"])
        x_values_imp = float(request.form["x_values_imp"])
        at = float(request.form["at"])
        a_theta = float(request.form["a_theta"])
        M = float(request.form["M"])

        Fl = (tau * at)
        o = max(y_values)
        Ta = round(Fl / max(y_values), 1)
        Fo = calculate_total_area(x_values, y_values)
        k = round(Fo / Fl, 1)
        t = find_first_positive_index(y_values, x_values)

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

        image_acceleration_charact, z_values = acceleration_characteristic(x_values, y_values)

        # Выводим HTML-код на сайт
        return render_template('PID/PID_full_page.html',
                               image_impulse=image_impulse,
                               numbers=[tau, x_values_imp, at, a_theta, M, Ta, k, t, o, Fl, Fo],
                               image_acceleration_charact=image_acceleration_charact)

    return render_template("PID/PID_full_page.html")


@app.route('/SIMOU', methods=["GET", "POST"])
def method_simou():
    if request.method == "POST":
        x_values = np.array([float(request.form[f"xValues_{i}"]) for i in range(14)])
        y_values = np.array([float(request.form[f"yValues_{i}"]) for i in range(14)])
        input_K = int(request.form['input_signal'])

        t1, t2, t3, input_K, extended_y, response_1st, response_2nd, response_3rd = simou_method(y_values, x_values, input_K)

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
        plt.xticks(np.arange(0, max(x_values) + 1, 20))
        plt.yticks(np.arange(0, max(y_values) + 1, 10))
        plt.minorticks_on()
        plt.grid(which='minor', linestyle=':', linewidth=0.5)

        # Сохраняем график в буфер
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)

        # Кодируем буфер в base64 строку
        image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

        # Генерируем HTML-код для отображения картинки
        image_html = '<img src="data:image/png;base64,{}">'.format(image_base64)

        return render_template('simou_methods.html',
                               image_html=image_html,
                               numbers=[round(t1, 2), round(t2, 2), round(t3, 2), round(input_K, 1)])

    return render_template('simou_methods.html')

app.run(debug=True)
