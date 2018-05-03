import { shallow } from 'enzyme';
import React from 'react';

import '../components/enzyme-setup';
import ScanningJobContainer from './ScanningJobContainer';
import FakePromise from '../helpers/FakePromise';
import * as API from '../api';
import Duration from '../Duration';

describe('<ScanningJobContainer />', () => {
    const onRemoveJob = jasmine.createSpy('onRemoveJob');
    const updateFeed = jasmine.createSpy('updateFeed');

    const props = {
        scanningJob: {
            name: 'Omnibus',
            identifier: 'job0000',
            duration: new Duration(123456),
            interval: new Duration(123),
            scannerId: 'hoho',
            status: 'Planned',
        },
        onRemoveJob,
        updateFeed,
    };

    beforeEach(() => {
        onRemoveJob.calls.reset();
        updateFeed.calls.reset();
    });

    describe('onFeatureExtract', () => {
        it('should call the API', () => {
            const extractFeatures = spyOn(API, 'extractFeatures').and.returnValue(new FakePromise());
            const wrapper = shallow(<ScanningJobContainer {...props} />);
            wrapper.find('ScanningJobPanel').prop('onFeatureExtract')(false);
            expect(extractFeatures).toHaveBeenCalledWith('job0000', 'analysis', false);
        });

        describe('Error warnings', () => {
            it('server error should produce a generic warning on error without reason', () => {
                spyOn(API, 'extractFeatures').and.returnValue(FakePromise.reject());
                const wrapper = shallow(<ScanningJobContainer {...props} />);
                wrapper.find('ScanningJobPanel').prop('onFeatureExtract')(false);
                wrapper.update();
                expect(wrapper.find('ScanningJobPanel').prop('error')).toContain('Unexpected error:');
            });

            it('should produce a warning with reason', () => {
                spyOn(API, 'extractFeatures').and.returnValue(FakePromise.reject('Wont work anyways ;('));
                const wrapper = shallow(<ScanningJobContainer {...props} />);
                wrapper.find('ScanningJobPanel').prop('onFeatureExtract')(false);
                wrapper.update();
                expect(wrapper.find('ScanningJobPanel').prop('error'))
                    .toContain('Extraction refused: Wont work anyways ;(');
            });
        });

        describe('Success info', () => {
            it('should produce a success alert', () => {
                spyOn(API, 'extractFeatures').and.returnValue(FakePromise.resolve({}));
                const wrapper = shallow(<ScanningJobContainer {...props} />);
                wrapper.find('ScanningJobPanel').prop('onFeatureExtract')(false);
                wrapper.update();
                expect(wrapper.find('ScanningJobPanel').prop('successInfo'))
                    .toContain('Feature extraction enqueued.');
            });
        });
    });

    describe('onStartJob', () => {
        describe('stalling', () => {
            beforeEach(() => {
                spyOn(API, 'startScanningJob').and.returnValue(new FakePromise());
            });

            it('should call startScanningJob', () => {
                const wrapper = shallow(<ScanningJobContainer {...props} />);
                wrapper.find('ScanningJobPanel').prop('onStartJob')();
                expect(API.startScanningJob).toHaveBeenCalledWith(props.scanningJob);
            });


            it('should deactivate button', () => {
                const wrapper = shallow(<ScanningJobContainer {...props} />);
                wrapper.find('ScanningJobPanel').prop('onStartJob')();
                wrapper.update();
                expect(wrapper.find('ScanningJobPanel').prop('disableStart')).toBeTruthy();
            });
        });

        describe('resolving', () => {
            beforeEach(() => {
                spyOn(API, 'startScanningJob').and.returnValue(FakePromise.resolve());
            });

            it('should update the feed', () => {
                const wrapper = shallow(<ScanningJobContainer {...props} />);
                wrapper.find('ScanningJobPanel').prop('onStartJob')();
                expect(updateFeed).toHaveBeenCalled();
            });
        });

        describe('rejecting', () => {
            beforeEach(() => {
                spyOn(API, 'startScanningJob').and.returnValue(FakePromise.reject('Busy'));
            });

            it('should set error', () => {
                const wrapper = shallow(<ScanningJobContainer {...props} />);
                wrapper.find('ScanningJobPanel').prop('onStartJob')();
                wrapper.update();
                expect(wrapper.find('ScanningJobPanel').prop('error')).toEqual('Error starting job: Busy');
            });
        });
    });

    describe('onStopJob', () => {
        describe('stalling', () => {
            beforeEach(() => {
                spyOn(API, 'terminateScanningJob').and.returnValue(new Promise(() => {}));
            });

            it('should call the expected API method', () => {
                const wrapper = shallow(<ScanningJobContainer {...props} />);
                const onStopJob = wrapper.find('ScanningJobPanel').prop('onStopJob');
                onStopJob('job1iamindeed', 'My reasons');
                expect(API.terminateScanningJob)
                    .toHaveBeenCalledWith('job1iamindeed', 'My reasons');
            });
        });

        describe('resolving', () => {
            beforeEach(() => {
                spyOn(API, 'terminateScanningJob').and.returnValue(Promise.resolve());
            });

            it('triggers an update of the jobs', (done) => {
                const wrapper = shallow(<ScanningJobContainer {...props} />);
                const onStopJob = wrapper.find('ScanningJobPanel').prop('onStopJob');
                onStopJob('job1iamindeed')
                    .then(() => {
                        expect(updateFeed).toHaveBeenCalled();
                        done();
                    });
            });
        });

        describe('rejecting', () => {
            beforeEach(() => {
                spyOn(API, 'terminateScanningJob').and.returnValue(Promise.reject('not good'));
            });

            it('should set the error', (done) => {
                const wrapper = shallow(<ScanningJobContainer {...props} />);
                const onStopJob = wrapper.find('ScanningJobPanel').prop('onStopJob');
                onStopJob('job1iamindeed')
                    .then(() => {
                        wrapper.update();
                        expect(wrapper.find('ScanningJobPanel').prop('error'))
                            .toEqual('Error deleting job: not good');
                        done();
                    });
            });
        });
    });
});
