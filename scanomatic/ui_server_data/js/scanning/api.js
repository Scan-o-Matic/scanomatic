import API from '../common/api.js';

export function submitJob(job) {
    return API.postJSON('/api/project/experiment/new', job);
}

export function getJobs() {
    return API.get('/api/project/experiment');
}

export function getFreeScanners() {
    return API.get('/api/status/scanners/free')
        .then(r => r.scanners.map(scanner => ({
            name: scanner.name,
            power: scanner.power,
            owned: !!scanner.owner,
        })));
}
