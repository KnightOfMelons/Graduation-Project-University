from flask import Flask, render_template, request
import matplotlib.pyplot as plt
import base64
from io import BytesIO

app = Flask(__name__)


@app.route("/")
def main_page():
    return render_template('index.html')


@app.route("/plot_impulse_transient_response", methods=['GET', 'POST'])
def impulse_transient_response():
    pass
    # if request.method == 'POST':
    #     x_values = [request.form[f'xValues_{i}'] for i in range(17)]
    #     y_values = [request.form[f'yValues_{i}'] for i in range(17)]
    #     plt.figure(figsize=(8, 4))
    #     plt.plot(x_values, y_values)
    #     plt.title('График')
    #     plt.xlabel('X')
    #     plt.ylabel('Y')
    #     plt.grid(True)
    #     buffer = BytesIO()
    #     plt.savefig(buffer, format='png')
    #     buffer.seek(0)
    #     image_png = buffer.getvalue()
    #     graph = base64.b64encode(image_png)
    #     graph = graph.decode('utf-8')
    #     buffer.close()
    #     return render_template('PID/PID.html', graph=graph)
    # return render_template('PID/PID.html')


app.run(debug=True)
