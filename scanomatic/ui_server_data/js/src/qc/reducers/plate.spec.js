import plate from './plate';

describe('/qc/reducers/plate', () => {
    it('returns the initial state', () => {
        expect(plate(undefined, {})).toEqual({ number: 0, qIndex: 0 });
    });

    it('resets to initial state on PROJECT_SET', () => {
        const action = { type: 'PROJECT_SET', project: 'test.a.test' };
        expect(plate({ number: 4 }, action)).toEqual({ number: 0, qIndex: 0 });
    });

    describe('PLATE_SET', () => {
        it('sets the plate number and removes curves', () => {
            const action = { type: 'PLATE_SET', plate: 3 };
            expect(plate({ number: 2, raw: [[[1, 2]]], qIndex: 0 }, action))
                .toEqual({ number: 3, qIndex: 0 });
        });

        it('doesnt do a thing if trying to set current plate', () => {
            const action = { type: 'PLATE_SET', plate: 3 };
            expect(plate({ number: 3, raw: [[[1, 2]]], qIndex: 1 }, action))
                .toEqual({ number: 3, raw: [[[1, 2]]], qIndex: 1 });
        });
    });

    describe('PLATE_GROWTHDATA_SET', () => {
        it('sets the plate growth data', () => {
            const times = [1, 2, 3];
            const raw = [[[2, 3, 5]]];
            const smooth = [[[4, 4, 4]]];
            const action = {
                type: 'PLATE_GROWTHDATA_SET',
                times,
                raw,
                smooth,
                plate: 0,
            };
            expect(plate(undefined, action)).toEqual({
                number: 0,
                times,
                raw,
                smooth,
                qIndex: 0,
            });
        });

        it('doesnt do a thing on plate missmatch', () => {
            const times = [1, 2, 3];
            const raw = [[[2, 3, 5]]];
            const smooth = [[[4, 4, 4]]];
            const action = {
                type: 'PLATE_GROWTHDATA_SET',
                times,
                raw,
                smooth,
                plate: 0,
            };
            expect(plate({ number: 3, qIndex: 2 }, action)).toEqual({
                number: 3,
                qIndex: 2,
            });
        });
    });

    describe('PLATE_PHENOTYPEDATA_SET', () => {
        const phenotype = 'GenerationTime';
        const phenotypes = [[1, 1], [2, 1]];

        it('sets the plate phenotype data', () => {
            const action = {
                type: 'PLATE_PHENOTYPEDATA_SET',
                plate: 0,
                phenotype,
                phenotypes,
            };
            expect(plate(undefined, action)).toEqual({
                number: 0,
                qIndex: 0,
                phenotypes: {
                    [phenotype]: phenotypes,
                },
            });
        });

        it('keeps previous phenotypes for same plate', () => {
            const action = {
                type: 'PLATE_PHENOTYPEDATA_SET',
                plate: 0,
                phenotype,
                phenotypes,
            };
            expect(plate(
                {
                    number: 0,
                    qIndex: 0,
                    phenotypes: { InitialValue: [[]] },
                },
                action,
            )).toEqual({
                number: 0,
                qIndex: 0,
                phenotypes: {
                    InitialValue: [[]],
                    [phenotype]: phenotypes,
                },
            });
        });

        it('doesnt do a thing on plate missmatch', () => {
            const action = {
                type: 'PLATE_PHENOTYPEDATA_SET',
                plate: 1,
                phenotype,
                phenotypes,
            };
            expect(plate(undefined, action)).toEqual({
                number: 0,
                qIndex: 0,
            });
        });
    });

    describe('PLATE_PHENOTYPEQC_SET', () => {
        const phenotype = 'GenerationTime';
        const badData = [[], []];
        const empty = [[0], [1]];
        const noGrowth = [[1, 1], [0, 1]];
        const undecidedProblem = [[0], [0]];

        it('sets the plate phenotype data', () => {
            const action = {
                type: 'PLATE_PHENOTYPEQC_SET',
                plate: 0,
                phenotype,
                badData,
                empty,
                noGrowth,
                undecidedProblem,
            };
            expect(plate(undefined, action)).toEqual({
                number: 0,
                qIndex: 0,
                qcmarks: {
                    [phenotype]: {
                        badData,
                        empty,
                        noGrowth,
                        undecidedProblem,
                    },
                },
            });
        });

        it('doesnt do a thing on plate missmatch', () => {
            const action = {
                type: 'PLATE_PHENOTYPEQC_SET',
                plate: 1,
                phenotype,
                badData,
                empty,
                noGrowth,
                undecidedProblem,
            };
            expect(plate(undefined, action)).toEqual({
                number: 0,
                qIndex: 0,
            });
        });
    });

    describe('QUALITYINDEX_QUEUE_SET', () => {
        const queue = [{ idx: 5, col: 4, row: 10 }, { idx: 0, col: 2, row: 55 }];

        it('sets a re-indexed, index-sorted queue', () => {
            const action = { type: 'QUALITYINDEX_QUEUE_SET', queue };
            expect(plate({ number: 2, qIndex: 0 }, action)).toEqual({
                number: 2,
                qIndex: 0,
                qIndexQueue: [
                    { idx: 0, col: 2, row: 55 },
                    { idx: 1, col: 4, row: 10 },
                ],
            });
        });
    });

    describe('qIndex', () => {
        const state = {
            number: 0,
            qIndex: 1,
            qIndexQueue: [
                { idx: 0, col: 1, row: 1 },
                { idx: 1, col: 0, row: 1 },
                { idx: 2, col: 1, row: 0 },
                { idx: 3, col: 0, row: 0 },
            ],
        };

        describe('QUALITYINDEX_SET', () => {
            it('doesnt set when missing queue', () => {
                const action = { type: 'QUALITYINDEX_SET', index: 42 };
                expect(plate(Object.assign({}, state, { qIndexQueue: null }).qIndex, action))
                    .toEqual(1);
            });

            it('sets new index', () => {
                const action = { type: 'QUALITYINDEX_SET', index: 2 };
                expect(plate(state, action).qIndex).toEqual(2);
            });

            it('limits max to last in queue', () => {
                const action = { type: 'QUALITYINDEX_SET', index: 42 };
                expect(plate(state, action).qIndex).toEqual(3);
            });

            it('limits min to first in queue', () => {
                const action = { type: 'QUALITYINDEX_SET', index: -42 };
                expect(plate(state, action).qIndex).toEqual(0);
            });
        });

        describe('QUALITYINDEX_NEXT', () => {
            it('doesnt set when missing queue', () => {
                const action = { type: 'QUALITYINDEX_NEXT' };
                expect(plate(Object.assign({}, state, { qIndexQueue: null }).qIndex, action))
                    .toEqual(1);
            });

            it('sets next index', () => {
                const action = { type: 'QUALITYINDEX_NEXT' };
                expect(plate(state, action).qIndex).toEqual(2);
            });

            it('wraps around to first', () => {
                const action = { type: 'QUALITYINDEX_NEXT' };
                expect(plate(Object.assign({}, state, { qIndex: 3 }), action).qIndex).toEqual(0);
            });
        });

        describe('QUALITYINDEX_PREVIOUS', () => {
            it('doesnt set when missing queue', () => {
                const action = { type: 'QUALITYINDEX_PREVIOUS' };
                expect(plate(Object.assign({}, state, { qIndexQueue: null }).qIndex, action))
                    .toEqual(1);
            });

            it('sets previous index', () => {
                const action = { type: 'QUALITYINDEX_PREVIOUS' };
                expect(plate(state, action).qIndex).toEqual(0);
            });

            it('wraps around to last', () => {
                const action = { type: 'QUALITYINDEX_PREVIOUS' };
                expect(plate(Object.assign({}, state, { qIndex: 0 }), action).qIndex).toEqual(3);
            });
        });
    });

    describe('CURVE_QCMARK_SET', () => {
        it('doesnt do a thing if plates missmatch', () => {
            const state = { number: 0, qIndex: 0 };
            const action = {
                type: 'CURVE_QCMARK_SET',
                mark: 'BadData',
                plate: 5,
                row: 0,
                col: 0,
            };
            expect(plate(state, action)).toEqual(state);
        });

        describe('with phenotype in action', () => {
            const action = {
                type: 'CURVE_QCMARK_SET',
                phenotype: 'GenerationTime',
                mark: 'BadData',
                plate: 0,
                row: 0,
                col: 0,
            };

            it('adds a mark even if no existing phenotype marks loaded', () => {
                const state = { number: 0, qIndex: 0 };
                expect(plate(state, action))
                    .toEqual(Object.assign({}, state, {
                        qcmarks: {
                            GenerationTime: {
                                badData: [[0], [0]],
                                noGrowth: null,
                                empty: null,
                                undecidedProblem: null,
                            },
                        },
                    }));
            });

            it('adds mark along-side other', () => {
                const state = {
                    number: 0,
                    qIndex: 0,
                    qcmarks: {
                        GenerationTime: {
                            badData: [[0], [1]],
                            empty: [[1], [1]],
                        },
                        GenerationTimeWhen: {
                            badData: [[], []],
                        },
                    },
                };
                expect(plate(state, action)).toEqual({
                    number: 0,
                    qIndex: 0,
                    qcmarks: {
                        GenerationTime: {
                            badData: [[0, 0], [1, 0]],
                            empty: [[1], [1]],
                            noGrowth: null,
                            undecidedProblem: null,
                        },
                        GenerationTimeWhen: {
                            badData: [[], []],
                        },
                    },
                });
            });

            it('removes previous mark for new mark', () => {
                const state = {
                    number: 0,
                    qIndex: 0,
                    qcmarks: {
                        GenerationTime: {
                            empty: [[0], [0]],
                        },
                    },
                };
                expect(plate(state, action)).toEqual({
                    number: 0,
                    qIndex: 0,
                    qcmarks: {
                        GenerationTime: {
                            badData: [[0], [0]],
                            empty: [[], []],
                            noGrowth: null,
                            undecidedProblem: null,
                        },
                    },
                });
            });
        });

        describe('without phenotype in action / settings for all loaded phenotypes', () => {
            it('adds no mark if no existing phenotype marks loaded', () => {
                const state = { number: 0, qIndex: 0 };
                const action = {
                    type: 'CURVE_QCMARK_SET',
                    mark: 'BadData',
                    plate: 0,
                    row: 0,
                    col: 0,
                };
                expect(plate(state, action))
                    .toEqual(Object.assign({}, state, { qcmarks: {} }));
            });

            it('updats mark for all phenotypes', () => {
                const state = {
                    number: 0,
                    qIndex: 0,
                    qcmarks: {
                        GenerationTime: {
                            badData: [[1], [1]],
                            empty: [[0], [0]],
                        },
                        GenerationTimeWhen: {},
                    },
                };
                const action = {
                    type: 'CURVE_QCMARK_SET',
                    mark: 'BadData',
                    plate: 0,
                    row: 0,
                    col: 0,
                };
                expect(plate(state, action))
                    .toEqual({
                        number: 0,
                        qIndex: 0,
                        qcmarks: {
                            GenerationTime: {
                                badData: [[1, 0], [1, 0]],
                                empty: [[], []],
                                noGrowth: null,
                                undecidedProblem: null,
                            },
                            GenerationTimeWhen: {
                                badData: [[0], [0]],
                                empty: null,
                                noGrowth: null,
                                undecidedProblem: null,
                            },
                        },
                    });
            });
        });
    });
});
