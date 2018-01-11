import { shallow } from 'enzyme';
import React from 'react';

import '../components/enzyme-setup';
import CCCInitializationContainer from '../../src/containers/CCCInitializationContainer';
import * as API from '../../src/api';
import FakePromise from '../helpers/FakePromise';


describe('<CCCInitializationContainer />', () => {
    const onInitialize = jasmine.createSpy('onInitialize');
    const onError = jasmine.createSpy('onError');
    const props = { onError, onInitialize };

    beforeEach(() => {
        onError.calls.reset();
        onInitialize.calls.reset();
        spyOn(API, 'GetFixtures').and.returnValue(new FakePromise());
        spyOn(API, 'GetPinningFormats').and.returnValue(new FakePromise());
    });

    it('should render a <CCCInitialization />', () => {
        const wrapper = shallow(<CCCInitializationContainer {...props} />);
        expect(wrapper.find('CCCInitialization').exists()).toBeTruthy();
    });

    it('should initialize species to ""', () => {
        const wrapper = shallow(<CCCInitializationContainer {...props} />);
        expect(wrapper.prop('species')).toEqual('');
    });

    it('should initialize reference to ""', () => {
        const wrapper = shallow(<CCCInitializationContainer {...props} />);
        expect(wrapper.prop('reference')).toEqual('');
    });

    it('should initialize the fixture to ""', () => {
        const wrapper = shallow(<CCCInitializationContainer {...props} />);
        expect(wrapper.prop('fixtureName')).toEqual('');
    });

    it('should initialize the pinning format name to ""', () => {
        const wrapper = shallow(<CCCInitializationContainer {...props} />);
        expect(wrapper.prop('pinningFormatName')).toEqual('');
    });

    it('should initialize the available fixtures to []', () => {
        const wrapper = shallow(<CCCInitializationContainer {...props} />);
        expect(wrapper.prop('fixtureNames')).toEqual([]);
    });

    it('should initialize the available pinning format namess to []', () => {
        const wrapper = shallow(<CCCInitializationContainer {...props} />);
        expect(wrapper.prop('pinningFormatNames')).toEqual([]);
    });

    it('should get the list of fixtures', () => {
        shallow(<CCCInitializationContainer {...props} />);
        expect(API.GetFixtures).toHaveBeenCalledWith();
    });

    it('should set the error prop if getting fixtures fails', () => {
        API.GetFixtures.and.returnValue(FakePromise.reject('Wibbly'));
        shallow(<CCCInitializationContainer {...props} />);
        expect(onError).toHaveBeenCalledWith('Error getting fixtures: Wibbly');
    });

    it('should set the error prop if there is no fixtures', () => {
        API.GetFixtures.and.returnValue(FakePromise.resolve([]));
        shallow(<CCCInitializationContainer {...props} />);
        expect(onError)
            .toHaveBeenCalledWith('You need to setup a fixture first.');
    });

    it('should request the list of pinning formats', () => {
        shallow(<CCCInitializationContainer {...props} />);
        expect(API.GetPinningFormats).toHaveBeenCalledWith();
    });

    it('should set the error prop if getting pinning formats fails', () => {
        API.GetPinningFormats.and.returnValue(FakePromise.reject('Wobbly'));
        shallow(<CCCInitializationContainer {...props} />);
        expect(onError)
            .toHaveBeenCalledWith('Error getting pinning formats: Wobbly');
    });

    describe('with pinning formats and fixture names', () => {
        const fixtureNames = ['MyFix1', 'MyFix2'];
        const pinningFormats = [
            { name: '1x1', nRows: 1, nCols: 1 },
            { name: '2x4', nRows: 4, nCols: 2 },
        ];

        beforeEach(() => {
            API.GetFixtures.and.returnValue(FakePromise.resolve(fixtureNames));
            API.GetPinningFormats.and.returnValue(FakePromise.resolve(pinningFormats));
        });

        it('should populate the availabe pinning format names', () => {
            const wrapper = shallow(<CCCInitializationContainer {...props} />);
            expect(wrapper.prop('pinningFormatNames')).toEqual(['1x1', '2x4']);
        });

        it('should populate the pinning format name prop', () => {
            const wrapper = shallow(<CCCInitializationContainer {...props} />);
            expect(wrapper.prop('pinningFormatName')).toEqual('1x1');
        });

        it('should populate the fixtures prop', () => {
            const wrapper = shallow(<CCCInitializationContainer {...props} />);
            expect(wrapper.prop('fixtureNames')).toEqual(fixtureNames);
        });

        it('should populate the fixtureName prop', () => {
            const wrapper = shallow(<CCCInitializationContainer {...props} />);
            expect(wrapper.prop('fixtureName')).toEqual(fixtureNames[0]);
        });

        it('should update the species on onSpeciesChange', () => {
            const wrapper = shallow(<CCCInitializationContainer {...props} />);
            wrapper.prop('onSpeciesChange')({ target: { value: 'XXX' } });
            wrapper.update();
            expect(wrapper.prop('species')).toEqual('XXX');
        });

        it('should update the reference on onReferenceChange', () => {
            const wrapper = shallow(<CCCInitializationContainer {...props} />);
            wrapper.prop('onReferenceChange')({ target: { value: 'YYY' } });
            wrapper.update();
            expect(wrapper.prop('reference')).toEqual('YYY');
        });

        it('should update the fixture name on onFixtureNameChange', () => {
            const wrapper = shallow(<CCCInitializationContainer {...props} />);
            wrapper.prop('onFixtureNameChange')({ target: { value: 'MyFix2' } });
            wrapper.update();
            expect(wrapper.prop('fixtureName')).toEqual('MyFix2');
        });

        it('should update the pinning format name on onPinningFormatNameChange', () => {
            const wrapper = shallow(<CCCInitializationContainer {...props} />);
            wrapper.prop('onPinningFormatNameChange')({ target: { value: '2x4' } });
            wrapper.update();
            expect(wrapper.prop('pinningFormatName')).toEqual('2x4');
        });

        describe('initializing', () => {
            it('should set the error if species is empty', () => {
                const wrapper = shallow(<CCCInitializationContainer {...props} />);
                wrapper.setState({ reference: 'XXX' });
                wrapper.prop('onSubmit')();
                expect(onError).toHaveBeenCalledWith('Species cannot be empty');
                expect(onInitialize).not.toHaveBeenCalled();
            });

            it('should set the error if reference is empty', () => {
                const wrapper = shallow(<CCCInitializationContainer {...props} />);
                wrapper.setState({ species: 'XXX' });
                wrapper.prop('onSubmit')();
                expect(onError).toHaveBeenCalledWith('Reference cannot be empty');
                expect(onInitialize).not.toHaveBeenCalled();
            });

            it('should set the error if fixture name is empty', () => {
                const wrapper = shallow(<CCCInitializationContainer {...props} />);
                wrapper.setState({ species: 'XXX', reference: 'YYY', fixtureName: '' });
                wrapper.prop('onSubmit')();
                expect(onError).toHaveBeenCalledWith('Fixture name cannot be empty');
                expect(onInitialize).not.toHaveBeenCalled();
            });

            it('should set the error if pinning format is empty', () => {
                const wrapper = shallow(<CCCInitializationContainer {...props} />);
                wrapper.setState({ species: 'XXX', reference: 'YYY', pinningFormat: null });
                wrapper.prop('onSubmit')();
                expect(onError).toHaveBeenCalledWith('Pinning format cannot be empty');
                expect(onInitialize).not.toHaveBeenCalled();
            });

            it('should call onInitialize if everything is set', () => {
                const wrapper = shallow(<CCCInitializationContainer {...props} />);
                wrapper.setState({ species: 'XXX', reference: 'YYY' });
                wrapper.prop('onSubmit')();
                wrapper.update();
                expect(onInitialize)
                    .toHaveBeenCalledWith('XXX', 'YYY', fixtureNames[0], pinningFormats[0]);
            });
        });
    });
});
