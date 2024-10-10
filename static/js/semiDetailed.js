document.addEventListener("DOMContentLoaded", function() {
    var homeButton = document.querySelector(".homeButton");
    homeButton.addEventListener("click", returnHome);
    var experimentName = document.querySelector(".experimentName");
    experimentName.addEventListener("click", returnToExperiment);
    var experiments = document.querySelectorAll(".version");
    experiments.forEach(exp => {
        exp.addEventListener("click", displayVersion);
    });
    formatVersionNumbers(document.querySelectorAll(".version"));
});

function returnHome(){
    window.location.href = "./";
}


function formatVersionNumbers(versions){
    versions.forEach(version => removeLeadingZero(version));

}

function returnToExperiment() {
    var experimentName = document.querySelector(".experimentName");
    window.location.href = "/experimentView?exp=" + experimentName.textContent;
}

function removeLeadingZero(version){
    if (version.textContent.length > 1) {
        version.textContent = version.textContent.substring(2);
    }
}

function displayVersion(event) {
    var version = event.target.textContent;
    var experiment = document.querySelector(".experimentName");
    window.location.href =  (version == "0" ? "./detailedView?vrs=" : "./detailedView?vrs=0.") + version + "&exp=" + experiment.textContent;
}