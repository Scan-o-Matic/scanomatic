
import { shallow } from 'enzyme';
import React from 'react';

import '../components/enzyme-setup';
import ScanningRootContainer from '../../src/containers/ScanningRootContainer';
import * as API from '../../src/api';
import FakePromise from '../helpers/FakePromise';


describe('<ScanningRootContainer />', () => {
    describe('Jobs Request doing nothing', () => {
        beforeEach(() => {
            spyOn(API, 'getScanningJobs').and.returnValue(new FakePromise());
        });

        it('should render <ScanningRoot />', () => {
            const wrapper = shallow(<ScanningRootContainer />);
            expect(wrapper.find('ScanningRoot').exists()).toBeTruthy();
        });

        it('should update the error on onError', () => {
            const wrapper = shallow(<ScanningRootContainer />);
            wrapper.prop('onError')('foobar');
            wrapper.update();
            expect(wrapper.prop('error')).toEqual('foobar');
        });

        it('should pass `newJob=False`', () => {
            const wrapper = shallow(<ScanningRootContainer />);
            expect(wrapper.prop('newJob')).toBeFalsy();
        });

        it('should pass no error', () => {
            const wrapper = shallow(<ScanningRootContainer />);
            expect(wrapper.prop('error')).toBeFalsy();
        });

        it('should pass no jobs', () => {
            const wrapper = shallow(<ScanningRootContainer />);
            expect(wrapper.prop('jobs')).toEqual([]);
        });

        it('should set errors', () => {
            const wrapper = shallow(<ScanningRootContainer />);
            wrapper.prop('onError')('Bad');
            wrapper.update();
            expect(wrapper.prop('error')).toEqual('Bad');
        });
        describe('New jobs', () => {
            it('should show new job', () => {
                const wrapper = shallow(<ScanningRootContainer />);
                wrapper.prop('onNewJob')();
                wrapper.update();
                expect(wrapper.prop('newJob')).toBeTruthy();
            });

            it('should close new job', () => {
                const wrapper = shallow(<ScanningRootContainer />);
                wrapper.setState({ newJob: true });
                wrapper.prop('onCloseNewJob')();
                wrapper.update();
                expect(wrapper.prop('newJob')).toBeFalsy();
            });

            it('should update jobs when closing', () => {
                const wrapper = shallow(<ScanningRootContainer />);
                wrapper.prop('onNewJob')();
                expect(API.getScanningJobs).toHaveBeenCalled();
            });
        });

        it('should update jobs when mounting', () => {
            shallow(<ScanningRootContainer />);
            expect(API.getScanningJobs).toHaveBeenCalled();
        });
    });

    describe('Jobs request resolving', () => {
        beforeEach(() => {
            spyOn(API, 'getScanningJobs').and
                .returnValue(FakePromise.resolve([1, 2]));
        });

        it('should pass updated jobs', () => {
            const wrapper = shallow(<ScanningRootContainer />);
            expect(wrapper.prop('jobs')).toEqual([1, 2]);
        });

        it('should clear errors when updating jobs', () => {
            const wrapper = shallow(<ScanningRootContainer />);
            wrapper.setState({ error: 'test' });
            wrapper.update();
            expect(wrapper.prop('error')).toEqual('test');
            wrapper.prop('onCloseNewJob')();
            wrapper.update();
            expect(wrapper.prop('error')).toBeFalsy();
        });
    });

    describe('Jobs request rejecting', () => {
        it('should set error', () => {
            spyOn(API, 'getScanningJobs').and
                .returnValue(FakePromise.reject('Fake'));
            const wrapper = shallow(<ScanningRootContainer />);
            expect(wrapper.prop('error'))
                .toEqual('Error requesting jobs: Fake');
        });
    });
});
