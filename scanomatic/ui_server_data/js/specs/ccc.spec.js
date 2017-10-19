describe('createSetGrayScaleTransformTask', () => {

    beforeEach(() => {
        spyOn(window, 'SetGrayScaleTransform');
        spyOn(window, '$').and.returnValue({
            hide: () => {},
            show: () => {},
            add: ()=>{return {add: ()=>{} };},
            click: ()=>{},
            change: ()=>{},
            dialog: ()=>{return {find: ()=>{return {on: () => {}};}};},
        });
        executeCCC();
    });

    it('returns a function', () => {
        const f = cccFunctions.createSetGrayScaleTransformTask();
        expect(typeof f).toBe('function');
    });

    describe('invoking its returned function', () => {
        it('it sets the step of the process', () => {
            spyOn(cccFunctions, 'setStep');
            const scope = {};
            const f = cccFunctions.createSetGrayScaleTransformTask(
                scope,
                'whatever'
            );
            f('something');
            expect(cccFunctions.setStep).toHaveBeenCalledWith(2);
        });

        it('it updates the scope', () => {
            const scope = {};
            const f = cccFunctions.createSetGrayScaleTransformTask(
                scope,
                'whatever'
            );
            f('something');
            expect(scope.Plate).toBe('whatever');
            expect(scope.PlateNextTaskInQueue).toBe('something');
        });

        it('it calls the API helper', () => {
            const scope = {};
            const f = cccFunctions.createSetGrayScaleTransformTask(
                scope,'whatever'
            );
            f('something');
            expect(SetGrayScaleTransform).toHaveBeenCalledWith(
                scope,
                undefined,
                undefined,
                'whatever',
                undefined,
                cccFunctions.setGrayScaleTransformSuccess,
                cccFunctions.setGrayScaleTransformError,
            );

        });
    });
});
