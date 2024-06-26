const inputNames = ['T2_value', 'T1_value', 'k_value', 'min_C0', 'min_C1'];

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
                    case 'k_value':
                        if (value < 0.1) {
                            this.value = 0.1;
                        } else if (value > 100) {
                            this.value = 100;
                        }
                        break;
                    case 'min_C0':
                    case 'min_C1':
                        if (value < 0.0000001) {
                            this.value = 0.0000001;
                        } else if (value > 100) {
                            this.value = 100;
                        }
                        break;
                }
            }
        });
    }
});