const inputNames = ['tau', 'x_values_imp', 'at', 'a_theta', 'M', 'input_signal'];

inputNames.forEach(name => {
    const input = document.querySelector(`input[name="${name}"]`);
    if (input) {
        input.addEventListener('blur', function(event) {
            let value = parseFloat(this.value);
            if (isNaN(value)) {
                this.value = ''; // Очищаем поле, если введенное значение не является числом
            } else {
                if (name === 'input_signal') {
                    if (value < 1) {
                        this.value = 1; // Устанавливаем минимальное значение для input_signal
                    } else if (value > 100) {
                        this.value = 100; // Устанавливаем максимальное значение для input_signal
                    }
                } else {
                    if (value < 0.1) {
                        this.value = 0.1; // Устанавливаем минимальное значение для других полей
                    } else if (value > 100) {
                        this.value = 100; // Устанавливаем максимальное значение для других полей
                    }
                }
            }
        });
    }
});