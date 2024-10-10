document.addEventListener("DOMContentLoaded", function() {
    var homeButton = document.querySelector(".homeButton")
    homeButton.addEventListener("click", returnHome)
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

function removeLeadingZero(version){
    if (version.textContent.length > 1) {
        version.textContent = version.textContent.substring(2);
    }
}

function displayVersion(event) {
    var version = event.target.textContent;
    var experiment = document.querySelector(".experimentName");
    window.location.href = "./detailedView?vrs=" + version + "&exp=" + experiment.textContent;
}