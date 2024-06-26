const inputNames = ['T2_value', 'T1_value', 't_value', 'k_value', 'x_min', 'x_max', 'y_min', 'y_max'];

inputNames.forEach(name => {
    const input = document.querySelector(`input[name="${name}"]`);
    if (input) {
        input.addEventListener('blur', function(event) {
            let value = parseFloat(this.value);
            if (isNaN(value)) {
                this.value = ''; // Очищаем поле, если введенное значение не является числом
            } else {
                switch (name) {
                    case 'T2_value':
                    case 'T1_value':
                        if (value < 1) {
                            this.value = 1;
                        } else if (value > 8000) {
                            this.value = 8000;
                        }
                        break;
                    case 't_value':
                        if (value < 1) {
                            this.value = 1;
                        } else if (value > 200) {
                            this.value = 200;
                        }
                        break;
                    case 'k_value':
                        if (value < 0.1) {
                            this.value = 0.1;
                        } else if (value > 100) {
                            this.value = 100;
                        }
                        break;
                    case 'x_min':
                    case 'x_max':
                    case 'y_min':
                    case 'y_max':
                        if (value < 0) {
                            this.value = 0;
                        } else if (value > 100) {
                            this.value = 100;
                        }
                        break;
                }
            }
        });
    }
});