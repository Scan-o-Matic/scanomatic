import settings from './settings';

describe('/qc/reducers/settings', () => {
    it('returns the initial state', () => {
        const action = {};
        expect(settings(undefined, action)).toEqual({});
    });

    it('handles PROJECT_SET', () => {
        const action = { type: 'PROJECT_SET', project: 'my/path/to/somewhere' };
        expect(settings(undefined, action)).toEqual({ project: action.project });
    });

    it('handles PHENOTYPE_SET', () => {
        const action = { type: 'PHENOTYPE_SET', phenotype: 'yield' };
        expect(settings({ project: '/my/proj' }, action)).toEqual({
            project: '/my/proj',
            phenotype: 'yield',
        });
    });
});
