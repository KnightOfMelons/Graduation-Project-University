document.addEventListener('DOMContentLoaded', function() {
    document.getElementById('resultT2Ta').style.display = 'none';
    document.getElementById('resultT1Ta').style.display = 'none';
});

document.getElementById('calculateT2Ta').addEventListener('click', function() {
    var endanswer1 = parseFloat(document.getElementsByName('endanswer_1')[0].value);
    var endanswer2 = parseFloat(document.getElementsByName('endanswer_2')[0].value);
    var Ta_value = parseFloat(document.getElementsByName('Ta_value')[0].value);

    var T2_value = (Ta_value * endanswer1).toFixed(0);
    var T1_value = (Ta_value * endanswer2).toFixed(0);

    document.getElementById('resultT2Ta').textContent = "T2 = " + T2_value;
    document.getElementById('resultT1Ta').textContent = "T1 = " + T1_value;

    document.getElementById('resultT2Ta').style.display = 'block';
    document.getElementById('resultT1Ta').style.display = 'block';
});