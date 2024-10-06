document.addEventListener("DOMContentLoaded", function() {
    const experiments = document.querySelectorAll(".experiment");
    experiments.forEach(exp => {
        exp.addEventListener("click", displayVersion);
    });
});

function displayVersion(event) {
    var experiment = event.target.textContent;
    window.location.href = "./experimentView?exp=" + experiment;
}
