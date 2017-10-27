describe('createSetGrayScaleTransformTask', () => {

    beforeEach(() => {
        spyOn(window, 'SetGrayScaleTransform');
        spyOn(window, '$').and.returnValue({
            hide: () => {},
            show: () => {},
            add: () => {return {add: () => {}};},
            click: () => {},
            change: () => {},
            dialog: () => {return {find: () => {return {on: () => {}};}};},
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

    const validNameRegexp = /^[a-z]([0-9a-z_\s])+$/i;
    const invalidNameMsg = 'This foo';

    beforeEach(() => {
        spyOn(cccFunctions, 'updateTips');
    });

    it('rejects names starting with a number', () => {
        const obj = {val: () => '42foo', addClass: () => true};
        expect(cccFunctions.checkName(
            obj, validNameRegexp, invalidNameMsg)).toBe(false);
    });

    it('rejects names with forbidded chars', () => {
        const obj = {val: () => 'foobar;', addClass: () => true};
        expect(cccFunctions.checkName(
            obj, validNameRegexp, invalidNameMsg)).toBe(false);
    });

    it('accepts names with allowed chars', () => {
        const obj = {val: () => 'foobar', addClass: () => true};
        expect(cccFunctions.checkName(
            obj, validNameRegexp, invalidNameMsg)).toBe(true);
    });

    it('sets correct error message', () => {
        const obj = {val: () => 'foobar;', addClass: () => true};
        cccFunctions.checkName(obj, validNameRegexp, invalidNameMsg);
        expect(cccFunctions.updateTips).toHaveBeenCalledWith(invalidNameMsg);
    });

});

describe('checkLength', () => {

    const minLength = 3;
    const maxLength = 20;
    const field = 'answer';

    beforeEach(() => {
        spyOn(cccFunctions, 'updateTips');
    });

    it('rejects too short names', () => {
        const obj = {val: () => '42', addClass: () => true};
        expect(cccFunctions.checkLength(
            obj, minLength, maxLength, field)).toBe(false);
    });

    it('rejects too long names', () => {
        const obj = {
            val: () => 'six multiplied by nine', addClass: () => true};
        expect(cccFunctions.checkLength(
            obj, minLength, maxLength, field)).toBe(false);
    });

    it('accepts names with allowed length', () => {
        const obj = {val: () => 'forty-two', addClass: () => true};
        expect(cccFunctions.checkLength(
            obj, minLength, maxLength, field)).toBe(true);
    });

    it('sets correct error message', () => {
        const obj = {val: () => '42', addClass: () => true};
        cccFunctions.checkLength(obj, minLength, maxLength, field);
        expect(cccFunctions.updateTips).toHaveBeenCalledWith(
            'Length of answer must be between 3 and 20.');
    });

});

describe('initiateCccError', () => {

    beforeEach(() => {
        spyOn(cccFunctions, 'updateTips');
    });

    it('sets correct error message', () => {
        cccFunctions.initiateCccError({responseJSON: {reason: 'foo'}});
        expect(cccFunctions.updateTips).toHaveBeenCalledWith('foo');

    });

});

describe('initiateNewCcc', () => {

    beforeEach(() => {
        spyOn(cccFunctions, 'updateTips');
        spyOn(cccFunctions, 'initiateCccSuccess');
        spyOn(cccFunctions, 'initiateCccError');
    });

    it('rejects invalid species', () => {
        const species = {
            val: () => 'Ravenous Bugblatterbeast of Traal',
            addClass: () => true};
        const reference = {val: () => 'The Guide', addClass: () => true};
        const allFields = {removeClass: () => true};
        expect(cccFunctions.initiateNewCcc(species, reference, allFields))
            .toBe(false);
    });

    it('rejects invalid reference', () => {
        const species = {val: () => 'Hoolovoo', addClass: () => true};
        const reference = {
            val: () => 'The Encyclopedia Galactica', addClass: () => true};
        const allFields = {removeClass: () => true};
        expect(cccFunctions.initiateNewCcc(species, reference, allFields))
            .toBe(false);
    });

    it('accepts valid species and reference', () => {
        const species = {val: () => 'Hooloovoo', addClass: () => true};
        const reference = {val: () => 'The Guide', addClass: () => true};
        const allFields = {removeClass: () => true};
        expect(cccFunctions.initiateNewCcc(species, reference, allFields))
            .toBe(true);
    });

});

describe('renderGridFail', () => {
    let scope;
    let btnReGrid;

    beforeEach(() => {
        jasmine.Ajax.install();
        btnReGrid = document.createElement('button');
        btnReGrid.id = 'btnReGrid';
        document.body.appendChild(btnReGrid);
        spyOn(window, 'SetGridding');
        scope = {};
    });

    afterEach(() => {
        jasmine.Ajax.uninstall();
        document.body.removeChild(btnReGrid);
    });

    it('should let the user re-grid by clicking the corresponding button', () => {
        const data = 1;
        cccFunctions.setGriddingError(data, scope);
        btnReGrid.click();
        expect(window.SetGridding).toHaveBeenCalled();
    });

    it('should make only one request each time the button is clicked', () => {
        const data = 1;
        cccFunctions.setGriddingError(data, scope);
        btnReGrid.click();
        expect(window.SetGridding).toHaveBeenCalledTimes(1);
        cccFunctions.setGriddingError(data, scope);
        btnReGrid.click();
        expect(window.SetGridding).toHaveBeenCalledTimes(2);
    });
});
