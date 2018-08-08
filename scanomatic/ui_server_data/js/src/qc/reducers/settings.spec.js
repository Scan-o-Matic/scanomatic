import settings from './settings';

describe('/qc/reducers/settings', () => {
    it('returns the initial state', () => {
        const action = {};
        expect(settings(undefined, action)).toEqual({ showNormalized: false });
    });

    describe('PROJECT_SET', () => {
        it('handles PROJECT_SET', () => {
            const action = { type: 'PROJECT_SET', project: 'my/path/to/somewhere' };
            expect(settings(undefined, action))
                .toEqual({ project: action.project, showNormalized: false });
        });

        it('sets showNormalized to false', () => {
            const action = { type: 'PROJECT_SET', project: 'my/path/to/somewhere' };
            expect(settings({ project: 'my/other/project', showNormalized: true }, action))
                .toEqual({ project: action.project, showNormalized: false });
        });
    });

    it('handles PHENOTYPE_SET', () => {
        const action = { type: 'PHENOTYPE_SET', phenotype: 'yield' };
        expect(settings({ project: '/my/proj' }, action)).toEqual({
            project: '/my/proj',
            phenotype: 'yield',
        });
    });

    it('handles SHOWNORMALIZED_SET', () => {
        const action = { type: 'SHOWNORMALIZED_SET', value: true };
        expect(settings({ showNormalized: false }, action))
            .toEqual({ showNormalized: true });
    });
});
