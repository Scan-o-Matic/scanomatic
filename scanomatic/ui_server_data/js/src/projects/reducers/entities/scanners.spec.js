import reducer from './scanners';

describe('projects/reducers/entities/scanners', () => {
    it('should return the default state', () => {
        expect(reducer(undefined, {})).toEqual(new Map());
    });
});
