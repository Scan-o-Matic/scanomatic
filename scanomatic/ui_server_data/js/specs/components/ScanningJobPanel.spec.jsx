import { shallow } from 'enzyme';
import React from 'react';

import './enzyme-setup';
import ScanningJobPanel from '../../src/components/ScanningJobPanel';


describe('<ScanningJobPanel />', () => {
    const props = {
        scanningJob: {
            name: 'Omnibus',
            identifier: 'job0000',
            duration: { days: 3, hours: 2, minutes: 51 },
            interval: 13,
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
});
