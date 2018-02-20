const getURLParam = (name) => {
    const results = new RegExp(`[?&]${name}=([^&#]*)`)
        .exec(window.location.href);
    return results && decodeURI(results[1]);
};

const setQCProjectFromURL = () => {
    const analysisDirectory = getURLParam('analysisdirectory');
    if (analysisDirectory) {
        return $.get(`/api/results/browse/${analysisDirectory}`)
            .then((r) => {
                $('#btnBrowseProject-box').hide();
                projectSlelectionStage('project');
                if (!r.is_project) {
                    modalMessage('<strong>Error</strong>: No analyis found!');
                    return;
                }
                const analysisInfo = Object.assign({}, r);
                if (!analysisInfo.project_name) {
                    analysisInfo.project_name = getURLParam('project');
                }
                wait();
                fillProjectDetails(analysisInfo);
                return Promise.resolve();
            });
    }
    return Promise.reject();
};
