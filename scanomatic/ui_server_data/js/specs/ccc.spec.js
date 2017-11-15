import * as cccAPI from '../ccc/api';
import cccFunctions from '../ccc/index';

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
