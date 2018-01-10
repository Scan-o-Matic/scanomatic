import API from '../common/api.js';

export function submitJob(job) {
    return API.postJSON('/api/project/experiment/new', job);
}

export function getJobs() {
    return API.get('/api/project/experiment');
}
