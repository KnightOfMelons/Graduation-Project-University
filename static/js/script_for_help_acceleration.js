document.getElementById('helpButton').addEventListener('click', function() {
    var helpContent = document.getElementById('helpContent');
    if (helpContent.classList.contains('d-none')) {
        helpContent.classList.remove('d-none');
    } else {
        helpContent.classList.add('d-none');
    }
});