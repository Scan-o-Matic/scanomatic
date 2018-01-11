import API from '../common/api.js';

export function submitJob(job) {
    return API.postJSON('/api/scan-jobs', job);
}

export function getJobs() {
    return API.get('/api/scan-jobs').then((r) => {
        const jobs = r.jobs.map((job) => {
            const newJob = Object.assign({}, job);
            newJob.scanner = {
                name: job.scanner.name,
                power: job.scanner.power,
                owned: !!job.scanner.owner,
            };
            return newJob;
        });
        return jobs;
    });
}

export function getScanners() {
    return API.get('/api/scanners')
        .then(r => r.scanners.map(scanner => ({
            name: scanner.name,
            power: scanner.power,
            owned: !!scanner.owner,
        })));
}
