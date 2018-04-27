import { shallow } from 'enzyme';
import React from 'react';

import './enzyme-setup';
import ScanningJobPanel from './ScanningJobPanel';
import Duration from '../Duration';
import * as API from '../api';
import FakePromise from '../helpers/FakePromise';


describe('<ScanningJobPanel />', () => {
    const props = {
        scanningJob: {
            name: 'Omnibus',
            identifier: 'job0000',
            duration: new Duration(123456),
            interval: new Duration(123),
            scannerId: 'hoho',
            status: 'Planned',
        },
        onStartJob: () => {},
        onRemoveJob: () => {},
        onStopJob: () => {},
    };

    const scanner = {
        name: 'Consule',
        owned: false,
        power: true,
        identifier: 'hoho',
    };

    it('should render a panel-title with the name', () => {
        const wrapper = shallow(<ScanningJobPanel {...props} />);
        const title = wrapper.find('h3.panel-title');
        expect(title.exists()).toBeTruthy();
        expect(title.text()).toContain('Omnibus');
    });

    it('should render a panel with id from the job', () => {
        const wrapper = shallow(<ScanningJobPanel {...props} />);
        const panel = wrapper.find('div.panel');
        expect(panel.prop('id')).toEqual('job-job0000');
    });

    it('should render a <ScanningJobStatusLabel />', () => {
        const wrapper = shallow(<ScanningJobPanel
            {...props}
        />);
        const label = wrapper.find('ScanningJobStatusLabel');
        expect(label.exists()).toBeTruthy();
        expect(label.prop('status')).toEqual('Planned');
    });

    it('should render a <ScanningJobPanelBody />', () => {
        const wrapper = shallow(<ScanningJobPanel
            {...props}
            scanner={scanner}
        />);
        const body = wrapper.find('ScanningJobPanelBody');
        expect(body.exists()).toBeTruthy();
        expect(body.prop('scanner')).toEqual(scanner);
        expect(body.props())
            .toEqual(jasmine.objectContaining(props.scanningJob));
    });

    describe('on remove', () => {
        it('should hide <ScanningJobPanelBody/>', () => {
            const wrapper = shallow(<ScanningJobPanel {...props} />);
            wrapper.find('ScanningJobPanelBody').prop('onRemoveJob')();
            wrapper.update();
            const body = wrapper.find('ScanningJobPanelBody');
            expect(body.exists()).toBeFalsy();
        });

        it('should render <ScanningJobRemoveDialogue/>', () => {
            const wrapper = shallow(<ScanningJobPanel {...props} />);
            wrapper.find('ScanningJobPanelBody').prop('onRemoveJob')();
            wrapper.update();
            const dialogue = wrapper.find('ScanningJobRemoveDialogue');
            expect(dialogue.exists()).toBeTruthy();
        });
    });

    describe('on confirm remove', () => {
        it('should render <ScanningJobPanelBody/>', () => {
            const wrapper = shallow(<ScanningJobPanel {...props} />);
            wrapper.find('ScanningJobPanelBody').prop('onRemoveJob')();
            wrapper.update();
            wrapper.find('ScanningJobRemoveDialogue').prop('onConfirm')();
            wrapper.update();
            const dialogue = wrapper.find('ScanningJobPanelBody');
            expect(dialogue.exists()).toBeTruthy();
        });

        it('should hide <ScanningJobRemoveDialogue/>', () => {
            const wrapper = shallow(<ScanningJobPanel {...props} />);
            wrapper.find('ScanningJobPanelBody').prop('onRemoveJob')();
            wrapper.update();
            wrapper.find('ScanningJobRemoveDialogue').prop('onConfirm')();
            wrapper.update();
            const dialogue = wrapper.find('ScanningJobRemoveDialogue');
            expect(dialogue.exists()).toBeFalsy();
        });

        it('should call onDelete with the job identifier', () => {
            const onRemoveJob = jasmine.createSpy('onRemoveJob');
            const wrapper = shallow(<ScanningJobPanel
                {...props}
                onRemoveJob={onRemoveJob}
            />);
            wrapper.find('ScanningJobPanelBody').prop('onRemoveJob')();
            wrapper.update();
            wrapper.find('ScanningJobRemoveDialogue').prop('onConfirm')();
            wrapper.update();
            expect(onRemoveJob).toHaveBeenCalledWith(props.scanningJob.identifier);
        });
    });

    describe('on cancel remove', () => {
        it('should render <ScanningJobPanelBody/>', () => {
            const wrapper = shallow(<ScanningJobPanel {...props} />);
            wrapper.find('ScanningJobPanelBody').prop('onRemoveJob')();
            wrapper.update();
            wrapper.find('ScanningJobRemoveDialogue').prop('onCancel')();
            wrapper.update();
            const dialogue = wrapper.find('ScanningJobPanelBody');
            expect(dialogue.exists()).toBeTruthy();
        });

        it('should hide <ScanningJobRemoveDialogue/>', () => {
            const wrapper = shallow(<ScanningJobPanel {...props} />);
            wrapper.find('ScanningJobPanelBody').prop('onRemoveJob')();
            wrapper.update();
            wrapper.find('ScanningJobRemoveDialogue').prop('onCancel')();
            wrapper.update();
            const dialogue = wrapper.find('ScanningJobRemoveDialogue');
            expect(dialogue.exists()).toBeFalsy();
        });

        it('should not call onDelete', () => {
            const onRemoveJob = jasmine.createSpy('onRemoveJob');
            const wrapper = shallow(<ScanningJobPanel
                {...props}
                onRemoveJob={onRemoveJob}
            />);
            wrapper.find('ScanningJobPanelBody').prop('onRemoveJob')();
            wrapper.update();
            wrapper.find('ScanningJobRemoveDialogue').prop('onCancel')();
            wrapper.update();
            expect(onRemoveJob).not.toHaveBeenCalled();
        });
    });

    describe('on stop', () => {
        it('should hide <ScanningJobPanelBody/>', () => {
            const wrapper = shallow(<ScanningJobPanel {...props} />);
            wrapper.find('ScanningJobPanelBody').prop('onStopJob')();
            wrapper.update();
            const body = wrapper.find('ScanningJobPanelBody');
            expect(body.exists()).toBeFalsy();
        });

        it('should render <ScanningJobStopDialogue/>', () => {
            const wrapper = shallow(<ScanningJobPanel {...props} />);
            wrapper.find('ScanningJobPanelBody').prop('onStopJob')();
            wrapper.update();
            const dialogue = wrapper.find('ScanningJobStopDialogue');
            expect(dialogue.exists()).toBeTruthy();
        });
    });

    describe('on confirm stop', () => {
        it('should render <ScanningJobPanelBody/>', () => {
            const wrapper = shallow(<ScanningJobPanel {...props} />);
            wrapper.find('ScanningJobPanelBody').prop('onStopJob')();
            wrapper.update();
            wrapper.find('ScanningJobStopDialogue').prop('onConfirm')();
            wrapper.update();
            const dialogue = wrapper.find('ScanningJobPanelBody');
            expect(dialogue.exists()).toBeTruthy();
        });

        it('should hide <ScanningJobStopDialogue/>', () => {
            const wrapper = shallow(<ScanningJobPanel {...props} />);
            wrapper.find('ScanningJobPanelBody').prop('onStopJob')();
            wrapper.update();
            wrapper.find('ScanningJobStopDialogue').prop('onConfirm')();
            wrapper.update();
            const dialogue = wrapper.find('ScanningJobStopDialogue');
            expect(dialogue.exists()).toBeFalsy();
        });

        it('should call onStopJob with the job indentifier and reason', () => {
            const onStopJob = jasmine.createSpy('onStopJob');
            const wrapper = shallow(<ScanningJobPanel
                {...props}
                onStopJob={onStopJob}
            />);
            wrapper.find('ScanningJobPanelBody').prop('onStopJob')();
            wrapper.update();
            wrapper.find('ScanningJobStopDialogue')
                .prop('onConfirm')('My reason');
            wrapper.update();
            expect(onStopJob)
                .toHaveBeenCalledWith(props.scanningJob.identifier, 'My reason');
        });
    });

    describe('on cancel stop', () => {
        it('should render <ScanningJobPanelBody/>', () => {
            const wrapper = shallow(<ScanningJobPanel {...props} />);
            wrapper.find('ScanningJobPanelBody').prop('onStopJob')();
            wrapper.update();
            wrapper.find('ScanningJobStopDialogue').prop('onCancel')();
            wrapper.update();
            const dialogue = wrapper.find('ScanningJobPanelBody');
            expect(dialogue.exists()).toBeTruthy();
        });

        it('should hide <ScanningJobStopDialogue/>', () => {
            const wrapper = shallow(<ScanningJobPanel {...props} />);
            wrapper.find('ScanningJobPanelBody').prop('onStopJob')();
            wrapper.update();
            wrapper.find('ScanningJobStopDialogue').prop('onCancel')();
            wrapper.update();
            const dialogue = wrapper.find('ScanningJobStopDialogue');
            expect(dialogue.exists()).toBeFalsy();
        });

        it('should not call onDelete', () => {
            const onStopJob = jasmine.createSpy('onStopJob');
            const wrapper = shallow(<ScanningJobPanel
                {...props}
                onStopJob={onStopJob}
            />);
            wrapper.find('ScanningJobPanelBody').prop('onStopJob')();
            wrapper.update();
            wrapper.find('ScanningJobStopDialogue').prop('onCancel')();
            wrapper.update();
            expect(onStopJob).not.toHaveBeenCalled();
        });
    });

    describe('Feature Extract', () => {
        const propsCompleted = {
            scanningJob: {
                name: 'Omnibus',
                identifier: 'job0000',
                duration: new Duration(123456),
                interval: new Duration(123),
                scannerId: 'hoho',
                status: 'Completed',
                startTime: new Date('1980-03-23T13:00:00Z'),
                endTime: new Date('1980-03-26T15:51:00Z'),
            },
            onStartJob: () => {},
            onRemoveJob: () => {},
            onStopJob: () => {},
        };

        it('should render <ScanningJobFeatureExtractDialogue />', () => {
            const wrapper = shallow(<ScanningJobPanel {...propsCompleted} />);
            wrapper.find('ScanningJobPanelBody').prop('onShowFeatureExtractDialogue')();
            wrapper.update();
            expect(wrapper.find('ScanningJobFeatureExtractDialogue').exists()).toBeTruthy();
        });

        it('should hide ScanningJobPanelBody', () => {
            const wrapper = shallow(<ScanningJobPanel {...propsCompleted} />);
            wrapper.find('ScanningJobPanelBody').prop('onShowFeatureExtractDialogue')();
            wrapper.update();
            expect(wrapper.find('ScanningJobPanelBody').exists()).toBeFalsy();
        });

        it('should dismiss dialogue onCancel', () => {
            const wrapper = shallow(<ScanningJobPanel {...propsCompleted} />);
            wrapper.find('ScanningJobPanelBody').prop('onShowFeatureExtractDialogue')();
            wrapper.update();
            wrapper.find('ScanningJobFeatureExtractDialogue').prop('onCancel')();
            wrapper.update();
            expect(wrapper.find('ScanningJobPanelBody').exists()).toBeTruthy();
            expect(wrapper.find('ScanningJobFeatureExtractDialogue').exists()).toBeFalsy();
        });

        describe('Submit request', () => {
            it('should call the API', () => {
                const extractFeatures = spyOn(API, 'extractFeatures').and.returnValue(new FakePromise());
                const wrapper = shallow(<ScanningJobPanel {...propsCompleted} />);
                wrapper.find('ScanningJobPanelBody').prop('onShowFeatureExtractDialogue')();
                wrapper.update();
                wrapper.find('ScanningJobFeatureExtractDialogue')
                    .prop('onExtractFeatures')(false);
                wrapper.update();
                expect(extractFeatures).toHaveBeenCalledWith('job0000', 'analysis', false);
            });

            it('should hide the dialogue', () => {
                spyOn(API, 'extractFeatures').and.returnValue(new FakePromise());
                const wrapper = shallow(<ScanningJobPanel {...propsCompleted} />);
                wrapper.find('ScanningJobPanelBody').prop('onShowFeatureExtractDialogue')();
                wrapper.update();
                wrapper.find('ScanningJobFeatureExtractDialogue')
                    .prop('onExtractFeatures')(false);
                wrapper.update();
                expect(wrapper.find('ScanningJobPanelBody').exists()).toBeTruthy();
                expect(wrapper.find('ScanningJobFeatureExtractDialogue').exists()).toBeFalsy();
            });

            describe('Error warnings', () => {
                it('should produce a generic warning on error without reason', () => {
                    spyOn(API, 'extractFeatures').and.returnValue(FakePromise.reject());
                    const wrapper = shallow(<ScanningJobPanel {...propsCompleted} />);
                    wrapper.find('ScanningJobPanelBody').prop('onShowFeatureExtractDialogue')();
                    wrapper.update();
                    wrapper.find('ScanningJobFeatureExtractDialogue')
                        .prop('onExtractFeatures')(false);
                    wrapper.update();
                    const alert = wrapper.find('.alert');
                    expect(alert.exists()).toBeTruthy();
                    expect(alert.hasClass('alert-danger'));
                    expect(alert.text()).toContain('Unexpected error:');
                });

                it('should produce a warning with reason', () => {
                    spyOn(API, 'extractFeatures').and.returnValue(FakePromise.reject('Wont work anyways ;('));
                    const wrapper = shallow(<ScanningJobPanel {...propsCompleted} />);
                    wrapper.find('ScanningJobPanelBody').prop('onShowFeatureExtractDialogue')();
                    wrapper.update();
                    wrapper.find('ScanningJobFeatureExtractDialogue')
                        .prop('onExtractFeatures')(false);
                    wrapper.update();
                    const alert = wrapper.find('.alert');
                    expect(alert.text()).toContain('Extraction refused: Wont work anyways ;(');
                });

                it('warning should be dismissable', () => {
                    spyOn(API, 'extractFeatures').and.returnValue(FakePromise.reject());
                    const wrapper = shallow(<ScanningJobPanel {...propsCompleted} />);
                    wrapper.find('ScanningJobPanelBody').prop('onShowFeatureExtractDialogue')();
                    wrapper.update();
                    wrapper.find('ScanningJobFeatureExtractDialogue')
                        .prop('onExtractFeatures')(false);
                    wrapper.update();
                    const alert = wrapper.find('.alert');
                    alert.find('button').simulate('click');
                    wrapper.update();
                    expect(wrapper.find('.alert').exists()).toBeFalsy();
                });
            });

            describe('Success info', () => {
                it('should produce a success alert', () => {
                    spyOn(API, 'extractFeatures').and.returnValue(FakePromise.resolve({}));
                    const wrapper = shallow(<ScanningJobPanel {...propsCompleted} />);
                    wrapper.find('ScanningJobPanelBody').prop('onShowFeatureExtractDialogue')();
                    wrapper.update();
                    wrapper.find('ScanningJobFeatureExtractDialogue')
                        .prop('onExtractFeatures')(false);
                    wrapper.update();
                    const alert = wrapper.find('.alert');
                    expect(alert.exists()).toBeTruthy();
                    expect(alert.hasClass('alert-success'));
                    expect(alert.text()).toContain('Feature extraction enqueued.');
                });

                it('should be dismissible', () => {
                    spyOn(API, 'extractFeatures').and.returnValue(FakePromise.resolve({}));
                    const wrapper = shallow(<ScanningJobPanel {...propsCompleted} />);
                    wrapper.find('ScanningJobPanelBody').prop('onShowFeatureExtractDialogue')();
                    wrapper.update();
                    wrapper.find('ScanningJobFeatureExtractDialogue')
                        .prop('onExtractFeatures')(false);
                    wrapper.update();
                    const alert = wrapper.find('.alert');
                    alert.find('button').simulate('click');
                    wrapper.update();
                    expect(wrapper.find('.alert').exists()).toBeFalsy();
                });
            });
        });
    });
});
