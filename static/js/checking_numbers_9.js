const inputNames = ['yValues_0', 'yValues_1', 'yValues_2', 'yValues_3', 'yValues_4', 'yValues_5', 'yValues_6', 'yValues_7', 'yValues_8', 'yValues_9', 'yValues_10', 'yValues_11', 'yValues_12', 'yValues_2_0', 'yValues_2_1', 'yValues_2_2', 'yValues_2_3', 'yValues_2_4', 'yValues_2_5', 'yValues_2_6', 'yValues_2_7', 'yValues_2_8', 'yValues_2_9', 'yValues_2_10', 'yValues_2_11', 'yValues_2_12'];

inputNames.forEach(name => {
    const input = document.querySelector(`input[name="${name}"]`);
    if (input) {
        input.addEventListener('blur', function(event) {
            let value = parseFloat(this.value);
            if (isNaN(value)) {
                this.value = ''; // Очищаем поле, если введенное значение не является числом
            } else {
                if (value < 0.001) {
                    this.value = 0.001;
                } else if (value > 300) {
                    this.value = 300;
                }
            }
        });
    }
});