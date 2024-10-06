document.addEventListener("DOMContentLoaded", function() {
    var homeButton = document.querySelector(".homeButton")
    homeButton.addEventListener("click", returnHome)
    var branch = document.querySelector(".branch");
    branch.addEventListener("click", createBranch);
    var submitButton = document.querySelector(".upload");
    submitButton.addEventListener("click", openPopUp)
    var exitButton = document.querySelector(".exitButton");
    exitButton.addEventListener("click", closePopUp);
    var uploadButton = document.querySelector("#uploadButton");
    uploadButton.addEventListener("click", uploadFile);
});

function uploadFile(){
    

    closePopUp()
}

function closePopUp(){
    var screenCover = document.querySelector(".screenCover");
    screenCover.style.visibility = "hidden";
}

function openPopUp(){
    var screenCover = document.querySelector(".screenCover");
    screenCover.style.visibility = "visible";

}

function createBranch(){
    var version = document.querySelector("#version"); 
    var experiment = document.querySelector(".experimentName");
    window.location.href = "./addBranch?vrs=" + version.textContent + "&exp=" + experiment.textContent;
}

function returnHome(){
    window.location.href = "./";
}
