document.addEventListener("DOMContentLoaded", function() {
    var addButton = document.querySelector(".addExperiment");
    addButton.addEventListener("click", openPopUp);
    var exitButton = document.querySelector(".exitButton");
    exitButton.addEventListener("click", closePopUp);
    var complete = document.querySelector(".complete");
    complete.addEventListener("click", addExperiment);
    var experiments = document.querySelectorAll(".experiment");
    experiments.forEach(exp => {
        exp.addEventListener("click", displayExperiment);
    });
    
});

function addExperiment(){
    var name = document.querySelector("#nameWindow").value;
    window.location.href = "./addExperiment?name=" + name;
    closePopUp();
}

function closePopUp(){
    var screenCover = document.querySelector(".screenCover");
    screenCover.style.visibility = "hidden";
}

function openPopUp(){
    var screenCover = document.querySelector(".screenCover");
    screenCover.style.visibility = "visible";

}

function displayExperiment(event) {
    var experiment = event.target.textContent;
    window.location.href = "./experimentView?exp=" + experiment;
}
