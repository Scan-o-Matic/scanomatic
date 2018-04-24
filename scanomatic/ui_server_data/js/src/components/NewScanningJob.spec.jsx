
import { shallow } from 'enzyme';
import React from 'react';

import './enzyme-setup';
import NewScanningJob from '../../src/components/NewScanningJob';


describe('<NewScanningJob/>', () => {
    const onNameChange = jasmine.createSpy('onNameChange');
    const onDurationChange = jasmine.createSpy('onDurationDaysChange');
    const onIntervalChange = jasmine.createSpy('onIntervalChange');
    const onScannerChange = jasmine.createSpy('onScannerChange');
    const onSubmit = jasmine.createSpy('onSubmit');
    const onCancel = jasmine.createSpy('onCancel');

    const props = {
        name: 'Test all the things!',
        duration: 1234567,
        scannerId: 'hoho',
        scanners: [
            {
                name: 'Tox',
                owned: false,
                power: true,
                identifier:
                'hoho',
            },
            {
                name: 'Npm',
                owned: true,
                power: false,
                identifier: 'haha',
            },
        ],
        interval: 16,
        onNameChange,
        onDurationChange,
        onIntervalChange,
        onScannerChange,
        onSubmit,
        onCancel,
    };

    beforeEach(() => {
        onDurationChange.calls.reset();
        onIntervalChange.calls.reset();
        onScannerChange.calls.reset();
        onSubmit.calls.reset();
        onCancel.calls.reset();
    });

    it('renders a heading', () => {
        const wrapper = shallow(<NewScanningJob {...props} />);
        expect(wrapper.find('div.panel-heading').text()).toEqual('New scan series');
    });

    it('renders errors', () => {
        const wrapper = shallow(<NewScanningJob {...props} error="Everything is fine!" />);
        expect(wrapper.find('div.alert').text()).toEqual('Everything is fine!');
    });

    it('renders name input', () => {
        const wrapper = shallow(<NewScanningJob {...props} />);
        const input = wrapper.find('input.name');
        expect(input.prop('value')).toEqual(props.name);
    });

    it('calls `onNameChange` when name changed', () => {
        const wrapper = shallow(<NewScanningJob {...props} />);
        const input = wrapper.find('input.name');
        input.simulate('change');
        expect(onNameChange).toHaveBeenCalled();
    });

    it('renders a DuraionInput', () => {
        const wrapper = shallow(<NewScanningJob {...props} />);
        const input = wrapper.find('DurationInput');
        expect(input.prop('duration')).toEqual(props.duration);
    });

    it('passes on `onDurationChange`', () => {
        const wrapper = shallow(<NewScanningJob {...props} />);
        const input = wrapper.find('DurationInput');
        expect(input.prop('onChange')).toEqual(onDurationChange);
    });

    it('renders interval input', () => {
        const wrapper = shallow(<NewScanningJob {...props} />);
        const input = wrapper.find('input.interval');
        expect(input.prop('value')).toEqual(props.interval);
    });

    it('calls `onIntervalChange` when hours changed', () => {
        const wrapper = shallow(<NewScanningJob {...props} />);
        const input = wrapper.find('input.interval');
        input.simulate('change');
        expect(onIntervalChange).toHaveBeenCalled();
    });

    describe('scanners', () => {
        it('shows the selected scanner', () => {
            const wrapper = shallow(<NewScanningJob {...props} />);
            const sel = wrapper.find('select.scanner');
            expect(sel.prop('value')).toEqual(props.scannerId);
        });

        it('renders the scanner options', () => {
            const wrapper = shallow(<NewScanningJob {...props} />);
            const opts = wrapper.find('option');
            expect(opts.first().text()).toEqual('Npm (offline, occupied)');
            expect(opts.last().text()).toEqual('Tox (online, free)');
        });

        it('calls `onScannerChange` when changed', () => {
            const wrapper = shallow(<NewScanningJob {...props} />);
            const sel = wrapper.find('select.scanner');
            sel.simulate('change');
            expect(onScannerChange).toHaveBeenCalled();
        });
    });

    describe('buttons', () => {
        it('renders Add to jobs-button', () => {
            const wrapper = shallow(<NewScanningJob {...props} />);
            const btn = wrapper.find('button.job-add');
            expect(btn.text()).toEqual('Add to jobs');
        });

        it('renders cancel-button', () => {
            const wrapper = shallow(<NewScanningJob {...props} />);
            const btn = wrapper.find('button.cancel');
            expect(btn.text()).toEqual('Cancel');
        });

        it('cancel-button calls `onCancel`', () => {
            const wrapper = shallow(<NewScanningJob {...props} />);
            const btn = wrapper.find('button.cancel');
            btn.simulate('click');
            expect(onCancel).toHaveBeenCalled();
        });

        it('Add to jobs-button calls `onSubmit`', () => {
            const wrapper = shallow(<NewScanningJob {...props} />);
            const btn = wrapper.find('button.job-add');
            btn.simulate('click');
            expect(onSubmit).toHaveBeenCalled();
        });
    });
});
