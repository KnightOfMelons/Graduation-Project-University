from flask import Flask, render_template, request
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)


@app.route("/")
def main_page():
    return render_template('index.html')


@app.route("/impulse_transient_response", methods=["GET", "POST"])
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
        image = '<img src="data:image/png;base64,{}">'.format(image_base64)

        # Выводим HTML-код на сайт
        return render_template('PID/impulse_transient_response.html', image=image,
                               numbers=[tau, x_values_imp, at, a_theta, M, Ta, k, t, o, Fl, Fo])

    return render_template("PID/impulse_transient_response.html")


app.run(debug=True)
