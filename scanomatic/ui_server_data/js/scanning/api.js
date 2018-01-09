import API from '../common/api.js';

export function submitJob(job) {
    return API.postJSON('/api/project/experiment', job);
}
