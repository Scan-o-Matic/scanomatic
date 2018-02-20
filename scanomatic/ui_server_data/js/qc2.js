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
                const analysisInfo = Object.assign({}, r);
                if (!analysisInfo.project_name) {
                    analysisInfo.project_name = getURLParam('project');
                }
                $('#btnBrowseProject-box').hide();
                wait();
                fillProjectDetails(analysisInfo);
                return Promise.resolve();
            });
    }
    return Promise.reject();
};
