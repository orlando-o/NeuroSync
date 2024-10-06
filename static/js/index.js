document.addEventListener("DOMContentLoaded", function() {
    var addExperiment = document.querySelector(".addExperiment");
    var experiments = document.querySelectorAll(".experiment");
    experiments.forEach(exp => {
        exp.addEventListener("click", displayExperiment);
    });
    addExperiment.addEventListener("click", addExperiment);
    
});

function addExperiment(){
    var popUp = document.createElement("div");
    var exitButton = document.createElement("div");
    var prompt = document.createElement("div");
    var nameWindow = document.createElement("")

    exitButton.append("X");
    prompt.append("Enter Experiment Name:")
}

function displayExperiment(event) {
    var experiment = event.target.textContent;
    window.location.href = "./experimentView?exp=" + experiment;
}
