import { shallow } from 'enzyme';
import React from 'react';

import '../components/enzyme-setup';
import NewScanningJobContainer from '../../src/containers/NewScanningJobContainer';
import * as API from '../../src/api';
import FakePromise from '../helpers/FakePromise';


describe('<NewScanningJobContainer />', () => {
    const onClose = jasmine.createSpy('onError');
    const props = { onClose };

    describe('Not resolving any promises', () => {
        beforeEach(() => {
            spyOn(API, 'submitScanningJob').and.returnValue(new FakePromise());
            spyOn(API, 'getScanners').and.returnValue(new FakePromise());
            onClose.calls.reset();
        });

        it('should render a <NewScanningJob />', () => {
            const wrapper = shallow(<NewScanningJobContainer {...props} />);
            expect(wrapper.find('NewScanningJob').exists()).toBeTruthy();
        });

        describe('initializing', () => {
            it('should set duration', () => {
                const wrapper = shallow(<NewScanningJobContainer {...props} />);
                expect(wrapper.prop('duration')).toEqual({
                    days: 3,
                    hours: 0,
                    minutes: 0,
                });
            });

            it('should set interval', () => {
                const wrapper = shallow(<NewScanningJobContainer {...props} />);
                expect(wrapper.prop('interval')).toEqual(20);
            });

            it('should call `getScanners`', () => {
                shallow(<NewScanningJobContainer {...props} />);
                expect(API.getScanners).toHaveBeenCalled();
            });
        });

        describe('updating state', () => {
            it('should set name', () => {
                const wrapper = shallow(<NewScanningJobContainer {...props} />);
                wrapper.prop('onNameChange')({ target: { value: 'Test' } });
                wrapper.update();
                expect(wrapper.prop('name')).toEqual('Test');
            });

            it('should set scanner name', () => {
                const wrapper = shallow(<NewScanningJobContainer {...props} />);
                wrapper.prop('onScannerNameChange')({ target: { value: 'Testing' } });
                wrapper.update();
                expect(wrapper.prop('scannerName')).toEqual('Testing');
            });

            it('should set interval', () => {
                const wrapper = shallow(<NewScanningJobContainer {...props} />);
                wrapper.prop('onIntervalChange')({ target: { value: '30' } });
                wrapper.update();
                expect(wrapper.prop('interval')).toEqual(30);
            });

            it('should limit short intervals', () => {
                const wrapper = shallow(<NewScanningJobContainer {...props} />);
                wrapper.prop('onIntervalChange')({ target: { value: '1' } });
                wrapper.update();
                expect(wrapper.prop('interval')).toEqual(5);
            });

            it('should set days', () => {
                const wrapper = shallow(<NewScanningJobContainer {...props} />);
                wrapper.prop('onDurationDaysChange')({ target: { value: '1' } });
                wrapper.update();
                expect(wrapper.prop('duration')).toEqual({
                    days: 1,
                    hours: 0,
                    minutes: 0,
                });
            });

            it('should refuse negative days', () => {
                const wrapper = shallow(<NewScanningJobContainer {...props} />);
                wrapper.prop('onDurationDaysChange')({ target: { value: '-1' } });
                wrapper.update();
                expect(wrapper.prop('duration')).toEqual({
                    days: 0,
                    hours: 0,
                    minutes: 0,
                });
            });

            it('should set hours', () => {
                const wrapper = shallow(<NewScanningJobContainer {...props} />);
                wrapper.prop('onDurationHoursChange')({ target: { value: '1' } });
                wrapper.update();
                expect(wrapper.prop('duration')).toEqual({
                    days: 3,
                    hours: 1,
                    minutes: 0,
                });
            });

            it('should increase days if setting more than 23 hours', () => {
                const wrapper = shallow(<NewScanningJobContainer {...props} />);
                wrapper.prop('onDurationHoursChange')({ target: { value: '24' } });
                wrapper.update();
                expect(wrapper.prop('duration')).toEqual({
                    days: 4,
                    hours: 0,
                    minutes: 0,
                });
            });

            it('should refuse negative hours', () => {
                const wrapper = shallow(<NewScanningJobContainer {...props} />);
                wrapper.prop('onDurationHoursChange')({ target: { value: '-1' } });
                wrapper.update();
                expect(wrapper.prop('duration')).toEqual({
                    days: 3,
                    hours: 0,
                    minutes: 0,
                });
            });

            it('should set minutes', () => {
                const wrapper = shallow(<NewScanningJobContainer {...props} />);
                wrapper.prop('onDurationMinutesChange')({ target: { value: '1' } });
                wrapper.update();
                expect(wrapper.prop('duration')).toEqual({
                    days: 3,
                    hours: 0,
                    minutes: 1,
                });
            });

            it('should increase hours if setting more than 59 minutes', () => {
                const wrapper = shallow(<NewScanningJobContainer {...props} />);
                wrapper.prop('onDurationMinutesChange')({ target: { value: '60' } });
                wrapper.update();
                expect(wrapper.prop('duration')).toEqual({
                    days: 3,
                    hours: 1,
                    minutes: 0,
                });
            });

            it('should increase days if setting more than 59 minutes', () => {
                const wrapper = shallow(<NewScanningJobContainer {...props} />);
                wrapper.setState({ duration: { days: 41, hours: 23, minutes: 0 } });
                wrapper.update();
                wrapper.prop('onDurationMinutesChange')({ target: { value: '60' } });
                wrapper.update();
                expect(wrapper.prop('duration')).toEqual({
                    days: 42,
                    hours: 0,
                    minutes: 0,
                });
            });

            it('should refuse negative minutes', () => {
                const wrapper = shallow(<NewScanningJobContainer {...props} />);
                wrapper.prop('onDurationMinutesChange')({ target: { value: '-1' } });
                wrapper.update();
                expect(wrapper.prop('duration')).toEqual({
                    days: 3,
                    hours: 0,
                    minutes: 0,
                });
            });
        });

        describe('actions', () => {
            it('should submit', () => {
                const state = {
                    name: 'Hyperion',
                    duration: '-eternal',
                    interval: 'limes null',
                    scannerName: 'Shrike',
                };
                const wrapper = shallow(<NewScanningJobContainer {...props} />);
                wrapper.setState(state);
                wrapper.update();
                wrapper.prop('onSubmit')();
                expect(API.submitScanningJob).toHaveBeenCalledWith(state);
            });

            it('should cancel', () => {
                const wrapper = shallow(<NewScanningJobContainer {...props} />);
                wrapper.prop('onCancel')();
                expect(onClose).toHaveBeenCalled();
            });
        });
    });

    describe('Resolving `submitScanningJob`', () => {
        it('should call `onClose` after submit', () => {
            onClose.calls.reset();
            spyOn(API, 'submitScanningJob').and.returnValue(FakePromise.resolve(5));
            spyOn(API, 'getScanners').and.returnValue(new FakePromise());
            const wrapper = shallow(<NewScanningJobContainer {...props} />);
            wrapper.prop('onSubmit')();
            expect(API.submitScanningJob).toHaveBeenCalled();
            expect(onClose).toHaveBeenCalled();
        });
    });

    describe('Rejecting `submitScanningJob`', () => {
        beforeEach(() => {
            onClose.calls.reset();
            spyOn(API, 'submitScanningJob').and
                .returnValue(FakePromise.reject('Sad King Billy'));
            spyOn(API, 'getScanners').and.returnValue(new FakePromise());
        });

        it('should call not `onClose`', () => {
            const wrapper = shallow(<NewScanningJobContainer {...props} />);
            wrapper.prop('onSubmit')();
            expect(API.submitScanningJob).toHaveBeenCalled();
        });

        it('should set error', () => {
            const wrapper = shallow(<NewScanningJobContainer {...props} />);
            wrapper.prop('onSubmit')();
            wrapper.update();
            expect(wrapper.prop('error'))
                .toEqual('Error submitting job: Sad King Billy');
        });
    });

    describe('Resovling `getScanners`', () => {
        beforeEach(() => {
            onClose.calls.reset();
            spyOn(API, 'submitScanningJob').and.returnValue(new FakePromise());
            spyOn(API, 'getScanners').and.returnValue(FakePromise
                .resolve([{ name: 'hello' }, { name: 'world' }]));
        });

        it('should set scanners', () => {
            const wrapper = shallow(<NewScanningJobContainer {...props} />);
            expect(wrapper.prop('scanners'))
                .toEqual([{ name: 'hello' }, { name: 'world' }]);
        });

        it('should default select first scanner', () => {
            const wrapper = shallow(<NewScanningJobContainer {...props} />);
            expect(wrapper.prop('scannerName')).toEqual('hello');
        });
    });

    describe('Rejecting `getScanners`', () => {
        beforeEach(() => {
            onClose.calls.reset();
            spyOn(API, 'submitScanningJob').and.returnValue(new FakePromise());
            spyOn(API, 'getScanners').and.returnValue(FakePromise
                .reject('World Tree'));
        });

        it('should set error', () => {
            const wrapper = shallow(<NewScanningJobContainer {...props} />);
            expect(wrapper.prop('error'))
                .toEqual('Error retrieving scanners: World Tree');
        });
    });
});
