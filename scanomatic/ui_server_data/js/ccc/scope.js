

function Scope() {
    this.File = null;
    this.FixtureName = null;
    this.PinFormat = null;
    this.Markers = null;
    this.CurrentImageId = null;
    this.cccId = null;
    this.AccessToken = null;
    this.Plate = null;
    this.PlateNextTaskInQueue = null;
    this.PlateDataURL = null;
    this.PlateGridding = null;
    this.PlateCurrentColony = null;
    this.PlateCurrentColonyRow = null;
    this.PlateCurrentColonyCol = null;
    //stats
    this.ColonyDetection = null;
    this.ColonyDetected = null;
    this.ColonyNotDetected = null;
    this.ColonyCompression = null;
    this.ColonyCompressed = null;
    this.ColonyNotCompressed = null;

}

export function iniColonyStats(scope) {
    scope.PlateColonies = [];
    scope.ColonyDetection = [];
    scope.ColonyDetected = 0;
    scope.ColonyNotDetected = 0;
    scope.ColonyCompression = [];
    scope.ColonyCompressed = 0;
    scope.ColonyNotCompressed = 0;
}

export function getCurrentScope() {
    //var scope = JSON.parse($("#inData").data("idCCCcurrentScope"));
    //$("#inData").data("idCCCcurrentScope", null);
    var stringScope = localStorage.getItem("scope");
    localStorage.removeItem("scope");
    var scope = JSON.parse(stringScope);
    return Object.preventExtensions(scope);
}

export function setCurrentScope(scope) {
    var stringScope = JSON.stringify(scope);
    //$("#inData").data("idCCCcurrentScope", stringScope);
    localStorage.setItem("scope", stringScope);
}

export function createScope() {
    var scope = new Scope();
    Object.preventExtensions(scope);
    return scope;
}
