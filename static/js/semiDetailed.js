document.addEventListener("DOMContentLoaded", function() {
    var homeButton = document.querySelector(".homeButton")
    homeButton.addEventListener("click", returnHome)
    var experiments = document.querySelectorAll(".version");
    experiments.forEach(exp => {
        exp.addEventListener("click", displayVersion);
    });
    
});

function returnHome(){
    window.location.href = "./";
}

// function displayVersion(event) {
//     var version = event.target.textContent;
//     window.location.href = "./detailedView?vrs=" + version + "exp=" + document.querySelector("experiment").textContent;
// }
function displayVersion(event) {
    var version = event.target.textContent;
    var experiment = document.querySelector(".experimentName");
    window.location.href = "./detailedView?vrs=" + version + "&exp=" + experiment.textContent;
}