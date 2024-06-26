const inputNames = ['t1_kos', 't2_kos', 't1_aldenb', 't2_aldenb', 't1_simou', 't2_simou', 'k'];

inputNames.forEach(name => {
    const input = document.querySelector(`input[name="${name}"]`);
    if (input) {
        input.addEventListener('blur', function(event) {
            let value = parseFloat(this.value);
            if (isNaN(value)) {
                this.value = ''; // Очищаем поле, если введенное значение не является числом
            } else if (value < 1) {
                this.value = 1; // Устанавливаем минимальное значение
            } else if (value > 8000) {
                this.value = 8000; // Устанавливаем максимальное значение
            }
        });
    }
});

const stopTimeInput = document.querySelector('input[name="stop_time"]');
if (stopTimeInput) {
    stopTimeInput.addEventListener('blur', function(event) {
        let value = parseFloat(this.value);
        if (isNaN(value)) {
            this.value = ''; // Очищаем поле, если введенное значение не является числом
        } else if (value < 600) {
            this.value = 600; // Устанавливаем минимальное значение
        } else if (value > 2000) {
            this.value = 2000; // Устанавливаем максимальное значение
        }
    });
}