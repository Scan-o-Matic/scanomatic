function getURLParam(uri, name) {
    const results = new RegExp(`[?&]${name}=([^&#]*)`)
        .exec(uri);
    return results && decodeURI(results[1]);
}

function setQCProjectFromURL()  {
    const uri = window.location.href;
    const analysisDirectory = getURLParam(uri, 'analysisdirectory');
    if (analysisDirectory) {
        return $.get(`/api/results/browse/${analysisDirectory}`)
            .then((r) => {
                $('#btnBrowseProject-box').hide();
                window.projectSelectionStage('project');
                if (!r.is_project) {
                    window.modalMessage('<strong>Error</strong>: No analyis found!');
                    return Promise.resolve();
                }
                const analysisInfo = Object.assign({}, r);
                if (!analysisInfo.project_name) {
                    analysisInfo.project_name = getURLParam(uri, 'project');
                }
                window.wait();
                window.fillProjectDetails(analysisInfo);
                return Promise.resolve();
            });
    }
    return Promise.reject();
}
