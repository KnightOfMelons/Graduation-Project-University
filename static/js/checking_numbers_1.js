const inputNames = ['tau', 'x_values_imp', 'at', 'a_theta', 'M', 'input_signal'];

inputNames.forEach(name => {
    const input = document.querySelector(`input[name="${name}"]`);
    if (input) {
        input.addEventListener('input', function(event) {
            let value = parseFloat(this.value);
            if (isNaN(value)) {
                this.value = ''; // Очищаем поле, если введенное значение не является числом
            } else if (value < 0) {
                this.value = 0; // Устанавливаем минимальное значение
            } else if (value > 100) {
                this.value = 100; // Устанавливаем максимальное значение
            }
        });
    }
});