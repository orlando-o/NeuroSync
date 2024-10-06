document.addEventListener("DOMContentLoaded", function() {
    var homeButton = document.querySelector(".homeButton")
    homeButton.addEventListener("click", returnHome)
    var branch = document.querySelector(".branch");
    branch.addEventListener("click", createBranch);
});

function createBranch(){
    var version = document.querySelector("#version"); 
    var experiment = document.querySelector(".experimentName");
    window.location.href = "./addBranch?vrs=" + version.textContent + "&exp=" + experiment.textContent;
}

function returnHome(){
    window.location.href = "./";
}
