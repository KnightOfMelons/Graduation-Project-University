const inputNames = ['t2_1', 't1_1', 'first_x_value_1', 'input_K_1', 't2_2', 't1_2', 'first_x_value_2', 'input_K_2'];

inputNames.forEach(name => {
    const input = document.querySelector(`input[name="${name}"]`);
    if (input) {
        input.addEventListener('blur', function(event) {
            let value = parseFloat(this.value);
            if (isNaN(value)) {
                this.value = ''; // Очищаем поле, если введенное значение не является числом
            } else {
                if (name.startsWith('t2') || name.startsWith('t1')) {
                    if (value < 0.1) {
                        this.value = 0.1;
                    } else if (value > 5000) {
                        this.value = 5000;
                    }
                } else if (name.startsWith('first_x_value')) {
                    if (value < 1) {
                        this.value = 1;
                    } else if (value > 350) {
                        this.value = 350;
                    }
                } else if (name.startsWith('input_K')) {
                    if (value < 0.1) {
                        this.value = 0.1;
                    } else if (value > 100) {
                        this.value = 100;
                    }
                }
            }
        });
    }
});