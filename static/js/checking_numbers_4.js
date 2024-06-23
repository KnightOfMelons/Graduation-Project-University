const inputNames = ['t1_1', 't2_2', 'min_C0', 'min_C1', 'min_C2', 'k'];

inputNames.forEach(name => {
    const input = document.querySelector(`input[name="${name}"]`);
    if (input) {
        input.addEventListener('input', function(event) {
            let value = parseFloat(this.value);
            if (isNaN(value)) {
                this.value = ''; // Очищаем поле, если введенное значение не является числом
            } else {
                switch (name) {
                    case 't1_1':
                    case 't2_2':
                        if (value < 1) {
                            this.value = 1;
                        } else if (value > 8000) {
                            this.value = 8000;
                        }
                        break;
                    case 'min_C0':
                    case 'min_C1':
                        if (value < 0.0000000001) {
                            this.value = 0.0000000001;
                        } else if (value > 100) {
                            this.value = 100;
                        }
                        break;
                    case 'min_C2':
                        if (value < 0.001) {
                            this.value = 0.001;
                        } else if (value > 300) {
                            this.value = 300;
                        }
                        break;
                    case 'k':
                        if (value < 1) {
                            this.value = 1;
                        } else if (value > 100) {
                            this.value = 100;
                        }
                        break;
                }
            }
        });
    }
});