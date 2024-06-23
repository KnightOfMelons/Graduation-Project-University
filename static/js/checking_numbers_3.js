const inputNames = ['t1_1', 't2_2', 'c2_1', 'c2_2', 'c2_3', 'k'];

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
                    case 'c2_1':
                    case 'c2_2':
                    case 'c2_3':
                        if (value < 1) {
                            this.value = 1;
                        } else if (value > 200) {
                            this.value = 200;
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