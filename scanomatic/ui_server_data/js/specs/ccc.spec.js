describe('createSetGrayScaleTransformTask', () => {

    beforeEach(() => {
        spyOn(window, 'SetGrayScaleTransform');
        spyOn(window, '$').and.returnValue({
            hide: ()=>{},
            show: ()=>{},
            add: ()=>{return {add: ()=>{}};},
            click: ()=>{},
            change: ()=>{},
            dialog: ()=>{return {find: ()=>{return {on: ()=>{}};}};},
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

describe('setGriddingError', () => {
    let textSpy = null;
    beforeEach(() => {
        spyOn(cccFunctions, 'renderGridFail');
        spyOn(cccFunctions, 'setStep');
        textSpy = jasmine.createSpy('text');
        spyOn(window, '$').and.returnValue({text: textSpy});
    });

    it('Sets step and renders grid', () => {
        const data = 1;
        const scope = 2;
        cccFunctions.setGriddingError(data, scope);
        expect(cccFunctions.setStep).toHaveBeenCalledWith(2.3);
        expect(textSpy).toHaveBeenCalled();
        expect(cccFunctions.renderGridFail).toHaveBeenCalledWith(
            data, scope
        );
    });
});

describe('setGriddingSuccess', () => {
    let textSpy = null;
    beforeEach(() => {
        spyOn(cccFunctions, 'renderGrid');
        spyOn(cccFunctions, 'setStep');
        textSpy = jasmine.createSpy('text');
        spyOn(window, '$').and.returnValue({text: textSpy});
    });

    it('Sets step and renders grid', () => {
        const data = 1;
        const scope = 2;
        cccFunctions.setGriddingSuccess(data, scope);
        expect(cccFunctions.setStep).toHaveBeenCalledWith(2.2);
        expect(textSpy).toHaveBeenCalledWith("Gridding was sucessful!");
        expect(cccFunctions.renderGrid).toHaveBeenCalledWith(
            data, scope
        );
    });
});

describe('checkName', () => {

  it('rejects names starting with a number', () => {
    // returns false
  });

  it('rejects names starting with forbidded chars', () => {
    // returns false
  });

  it('accepts names with allowed chars', () => {
    // returns true
  });

  it('sets correct error message', () => {
    // spy on text
  });

});

describe('checkLength', () => {

  it('rejects too short names', () => {
    // returns false
  });

  it('rejects too long names', () => {
    // returns false
  });

  it('accepts names with allowed length', () => {
    // returns true
  });

  it('sets correct error message', () => {
    // spy on text
  });

});

describe('initiateCccError', () => {

  it('sets correct error message', () => {
    // spy on text
  });

});

describe('initiateNewCcc', () => {

  it('rejects invalid species', () => {
    // returns false
  });

  it('rejects invalid reference', () => {
    // returns false
  });

  it('accepts valid species and reference', () => {
    // initiateCccSuccess called once, initiateCccError not called
    // returns true
  });

  it('rejects duplicated valid species and reference', () => {
    // initiateCccSuccess called once, initiateCccError not called
    // returns true
    // initiateCccSuccess called once, initiateCccError called once
    // returns true
  });

});
