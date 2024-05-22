document.addEventListener('DOMContentLoaded', function() {
    document.getElementById('resultTcTa').style.display = 'none';
    document.getElementById('equalSignTcTa').style.display = 'none';
});

document.getElementById('calculateTcTa').addEventListener('click', function() {
    var Tc_value = parseFloat(document.getElementsByName('Tc_value')[0].value);
    var Ta_value = parseFloat(document.getElementsByName('Ta_value')[0].value);
    var TcTa = Tc_value / Ta_value;
    document.getElementById('resultTcTa').textContent = TcTa.toFixed(2);
    document.getElementById('resultTcTa').style.display = 'block';
    document.getElementById('equalSignTcTa').style.display = 'block';
});