const inputNames = ['T2_value', 'T1_value', 't_value', 'k_value', 'y_value', 'stop_time'];

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
                    case 'y_value':
                        if (value < 0.0001) {
                            this.value = 0.0001;
                        } else if (value > 100) {
                            this.value = 100;
                        }
                        break;
                    case 'stop_time':
                        if (value < 600) {
                            this.value = 600;
                        } else if (value > 2500) {
                            this.value = 2500;
                        }
                        break;
                }
            }
        });
    }
});