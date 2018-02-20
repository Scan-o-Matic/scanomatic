const getURLParam = (name) => {
    const results = new RegExp(`[?&]${name}=([^&#]*)`)
        .exec(window.location.href);
    return results && results[1];
};

const setQCProjectFromURL = () => {
    const projectHint = getURLParam('projectdirectory');
    if (projectHint) {
        return $.get(`/api/results/browse/${projectHint}/analysis`)
            .then((r) => {
                $('#btnBrowseProject-box').hide();
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
