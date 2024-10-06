document.addEventListener("DOMContentLoaded", function() {
    
    var experiments = document.querySelectorAll(".version");
    experiments.forEach(exp => {
        exp.addEventListener("click", displayVersion);
    });
    
});

function displayVersion(event) {
    var version = event.target.textContent;
    window.location.href = "./detailedView?exp=" + version;
}
