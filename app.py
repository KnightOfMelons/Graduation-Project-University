from flask import Flask, render_template, request  # send_file
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
        # Получение данных из формы
        x_values = [float(request.form[f"xValues_{i}"]) for i in range(17)]
        y_values = [float(request.form[f"yValues_{i}"]) for i in range(17)]

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
        return render_template('PID/impulse_transient_response.html', image=image)

    return render_template("PID/impulse_transient_response.html")


app.run(debug=True)
