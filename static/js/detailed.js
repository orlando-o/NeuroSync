document.addEventListener("DOMContentLoaded", function() {
    var homeButton = document.querySelector(".homeButton")
    homeButton.addEventListener("click", returnHome)
        
});

function returnHome(){
    window.location.href = "./";
}
