document.addEventListener('DOMContentLoaded', function() {
    document.getElementById('resultK').style.display = 'none';
    document.getElementById('equalSign').style.display = 'none';
});

document.getElementById('calculateK').addEventListener('click', function() {
    var Yvih_value = parseFloat(document.getElementsByName('Yvih_value')[0].value);
    var Xvhod_imp = parseFloat(document.getElementsByName('Xvhod_imp')[0].value);
    var K = Yvih_value / Xvhod_imp;
    document.getElementById('resultK').textContent = K.toFixed(1);
    document.getElementById('resultK').style.display = 'block';
    document.getElementById('equalSign').style.display = 'block';
});