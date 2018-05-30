import Bridge from './bridge';
import * as actions from './actions';
import * as selectors from './selectors';


describe('/qc/bridge', () => {
    const state = { plate: {}, settings: {} };
    const store = {
        subscribe: jasmine.createSpy('subscribe'),
        dispatch: jasmine.createSpy('dispatch'),
        getState: jasmine.createSpy('getState').and.returnValue(state),
    };

    const bridge = Bridge(store);
    let retrievePlateCurves;

    beforeEach(() => {
        store.subscribe.calls.reset();
        store.dispatch.calls.reset();
        store.getState.calls.reset();
        retrievePlateCurves = spyOn(actions, 'retrievePlateCurves');
    });

    it('registers subscription to the store changes', () => {
        const fn = () => {};
        bridge.subscribe(fn);
        expect(store.subscribe).toHaveBeenCalledWith(fn);
    });

    describe('actions', () => {
        it('dispatches an PROJECT_SET on setProject', () => {
            bridge.actions.setProject('hello');
            expect(store.dispatch).toHaveBeenCalledWith({
                type: 'PROJECT_SET',
                project: 'hello',
            });
        });

        it('dispatches a PLATE_SET on setPlate', () => {
            bridge.actions.setPlate(42);
            expect(store.dispatch).toHaveBeenCalledWith({
                type: 'PLATE_SET',
                plate: 42,
            });
        });

        it('dispatches, a PINNING_SET on setPinning', () => {
            bridge.actions.setPinning(0, 4, 2);
            expect(store.dispatch).toHaveBeenCalledWith({
                type: 'PINNING_SET',
                plate: 0,
                rows: 4,
                cols: 2,
            });
        });

        it('dispatches a retrievePlateCurves ThunkAction on retrievePlateCurves', () => {
            bridge.actions.retrievePlateCurves();
            expect(store.dispatch).toHaveBeenCalled();
            expect(retrievePlateCurves).toHaveBeenCalled();
        });

        it('dispatches a PLATE_SET on retrievePlateCurves if argument set', () => {
            bridge.actions.retrievePlateCurves(2);
            expect(store.dispatch).toHaveBeenCalledWith({
                type: 'PLATE_SET',
                plate: 2,
            });
        });

        it('dispatches a PINNING_SET on retrievePlateCurves if arguments set', () => {
            bridge.actions.retrievePlateCurves(2, { rows: 5, cols: 3 });
            expect(store.dispatch).toHaveBeenCalledWith({
                type: 'PLATE_SET',
                plate: 2,
            });
            expect(store.dispatch).toHaveBeenCalledWith({
                type: 'PINNING_SET',
                plate: 2,
                rows: 5,
                cols: 3,
            });
        });
    });

    describe('selectors', () => {
        it('calls getRawCurve on getRawCurve', () => {
            const getRawCurve = spyOn(selectors, 'getRawCurve').and.returnValue([5, 2]);
            const raw = bridge.selectors.getRawCurve(0, 1, 2);
            expect(store.getState).toHaveBeenCalled();
            expect(getRawCurve).toHaveBeenCalledWith(state, 0, 1, 2);
            expect(raw).toEqual([5, 2]);
        });

        it('calls getSmoothCurve on getSmoothCurve', () => {
            const getSmoothCurve = spyOn(selectors, 'getSmoothCurve').and.returnValue([5, 2]);
            const smooth = bridge.selectors.getSmoothCurve(0, 1, 2);
            expect(store.getState).toHaveBeenCalled();
            expect(getSmoothCurve).toHaveBeenCalledWith(state, 0, 1, 2);
            expect(smooth).toEqual([5, 2]);
        });

        it('calls getTimes on getTimes', () => {
            const getTimes = spyOn(selectors, 'getTimes').and.returnValue([5, 2]);
            const times = bridge.selectors.getTimes(0);
            expect(store.getState).toHaveBeenCalled();
            expect(getTimes).toHaveBeenCalledWith(state, 0);
            expect(times).toEqual([5, 2]);
        });

        it('calls getPlate on getPlate', () => {
            const getPlate = spyOn(selectors, 'getPlate').and.returnValue(52);
            const plate = bridge.selectors.getPlate();
            expect(store.getState).toHaveBeenCalled();
            expect(getPlate).toHaveBeenCalledWith(state);
            expect(plate).toEqual(52);
        });
    });
});
