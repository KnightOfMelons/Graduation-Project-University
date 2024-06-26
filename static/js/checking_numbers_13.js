const inputNames = ['t1_value', 't2_value', 'm', 'first_x_value', 'k', 'stop_time'];

inputNames.forEach(name => {
    const input = document.querySelector(`input[name="${name}"]`);
    if (input) {
        input.addEventListener('blur', function(event) {
            let value = parseFloat(this.value);
            if (isNaN(value)) {
                this.value = ''; // Очищаем поле, если введенное значение не является числом
            } else {
                switch (name) {
                    case 't1_value':
                    case 't2_value':
                        if (value < 0.1) {
                            this.value = 0.1;
                        } else if (value > 5000) {
                            this.value = 5000;
                        }
                        break;
                    case 'm':
                        if (value < 0.01) {
                            this.value = 0.01;
                        } else if (value > 100) {
                            this.value = 100;
                        }
                        break;
                    case 'first_x_value':
                        if (value < 1) {
                            this.value = 1;
                        } else if (value > 350) {
                            this.value = 350;
                        }
                        break;
                    case 'k':
                        if (value < 0.1) {
                            this.value = 0.1;
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