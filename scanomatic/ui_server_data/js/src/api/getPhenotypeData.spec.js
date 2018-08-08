import * as API from '.';

describe('API', () => {
    const mostRecentRequest = () => jasmine.Ajax.requests.mostRecent();

    beforeEach(() => {
        jasmine.Ajax.install();
    });

    afterEach(() => {
        jasmine.Ajax.uninstall();
    });

    describe('getPhenotypeData', () => {
        const project = 'something/somewhere';
        const plate = 1;
        const phenotype = 'GenerationTime';
        const args = [project, plate, phenotype];

        it('should query the default phenotype uri if not normalized', () => {
            API.getPhenotypeData(...args, false);
            expect(mostRecentRequest().url)
                .toBe(`/api/results/phenotype/${phenotype}/${plate}/${project}`);
        });

        it('should query the nomalized phenotype uri if normalized', () => {
            API.getPhenotypeData(...args, true);
            expect(mostRecentRequest().url)
                .toBe(`/api/results/normalized_phenotype/${phenotype}/${plate}/${project}`);
        });

        it('should return a promise that resolves on success', (done) => {
            API.getPhenotypeData(...args, false).then((response) => {
                expect(response).toEqual({
                    phenotypes: [[1, 2, 3], [4, 5, 6]],
                    qcmarks: new Map([
                        ['BadData', [[0], [0]]],
                        ['Empty', [[0, 0], [1, 2]]],
                        ['NoGrowth', [[1], [2]]],
                        ['UndecidedProblem', [[], []]],
                    ]),
                    qIndexQueue: [
                        { idx: 0, col: 0, row: 1 },
                        { idx: 1, col: 1, row: 0 },
                        { idx: 2, col: 0, row: 0 },
                        { idx: 3, col: 2, row: 1 },
                        { idx: 4, col: 2, row: 0 },
                        { idx: 5, col: 1, row: 1 },
                    ],
                    normalized: false,
                });
                done();
            });
            mostRecentRequest().respondWith({
                status: 200,

                responseText: JSON.stringify({
                    data: [[1, 2, 3], [4, 5, 6]],
                    BadData: [[0], [0]],
                    Empty: [[0, 0], [1, 2]],
                    NoGrowth: [[1], [2]],
                    UndecidedProblem: [[], []],
                    qindex_cols: [0, 1, 0, 2, 2, 1],
                    qindex_rows: [1, 0, 0, 1, 0, 1],
                }),
            });
        });

        it('should return a promise that rejects on error', (done) => {
            API.getPhenotypeData(...args).catch((response) => {
                expect(response).toEqual('yesyesno');
                done();
            });
            mostRecentRequest().respondWith({
                status: 400,
                responseText: JSON.stringify({ reason: 'yesyesno' }),
            });
        });
    });
});
