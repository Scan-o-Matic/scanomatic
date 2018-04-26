import React from 'react';
import { shallow } from 'enzyme';

import '../components/enzyme-setup';
import NewExperimentPanel from './NewExperimentPanel';

describe('<NewExperimentPanel/>', () => {
    const defaultProps = {
        project: 'I project',
        name: 'Cool idea',
        description: 'Test all the things',
        duration: 120000,
        interval: 60000,
        scannerId: 'hoho',
        scanners: [
            {
                name: 'Tox',
                owned: false,
                power: true,
                identifier: 'hoho',
            },
            {
                name: 'Npm',
                owned: true,
                power: false,
                identifier: 'haha',
            },
        ],
        onChange: jasmine.createSpy('onNameChange'),
        onSubmit: jasmine.createSpy('onSubmit'),
        onCancel: jasmine.createSpy('onCancel'),
    };

    const errors = {
        general: 'Cound not reach server',
        name: 'Maybe you should change yours',
        description: 'Give me at least one word',
        duration: 'Need a minute',
        interval: 'Takes at least five',
        scanner: 'Thats just a fake scanner',
    };

    let wrapper;
    let wrapperErrors;

    beforeEach(() => {
        defaultProps.onChange.calls.reset();
        defaultProps.onSubmit.calls.reset();
        defaultProps.onCancel.calls.reset();
        wrapper = shallow(<NewExperimentPanel {...defaultProps} />);
        wrapperErrors = shallow(<NewExperimentPanel {...defaultProps} errors={errors} />);
    });

    describe('actions', () => {
        it('should call onChange when name is changed', () => {
            const evt = { target: { value: 'foo' } };
            wrapper.find('input.name').simulate('change', evt);
            expect(defaultProps.onChange).toHaveBeenCalledWith('name', evt.target.value);
        });

        it('should call onChange when description is changed', () => {
            const evt = { target: { value: 'foo' } };
            wrapper.find('textarea.description').simulate('change', evt);
            expect(defaultProps.onChange).toHaveBeenCalledWith('description', evt.target.value);
        });

        it('should call onChange when duration is changed', () => {
            wrapper.find('DurationInput').prop('onChange')(44);
            expect(defaultProps.onChange).toHaveBeenCalledWith('duration', 44);
        });

        it('should call onChange when interval is changed', () => {
            const evt = { target: { value: 1 } };
            wrapper.find('input.interval').simulate('change', evt);
            expect(defaultProps.onChange).toHaveBeenCalledWith('interval', evt.target.value);
        });

        it('should call onChange when scanner is changed', () => {
            const evt = { target: { value: 'myscanner' } };
            wrapper.find('select.scanner').simulate('change', evt);
            expect(defaultProps.onChange).toHaveBeenCalledWith('scanner', evt.target.value);
        });

        it('should call onSubmit when form is submitted', () => {
            wrapper.find('button.experiment-add').simulate('click');
            expect(defaultProps.onSubmit).toHaveBeenCalled();
        });

        it('should call onCancel when cancel button is clicked', () => {
            wrapper.find('button.cancel').simulate('click');
            expect(defaultProps.onCancel).toHaveBeenCalled();
        });
    });

    describe('rendering', () => {
        it('renders the panel heading', () => {
            const heading = wrapper.find('.panel-heading');
            expect(heading.exists()).toBeTruthy();
            expect(heading.text()).toEqual('New “I project” experiment');
        });

        describe('name input', () => {
            it('renders with passed value', () => {
                const input = wrapper.find('input.name');
                expect(input.exists()).toBeTruthy();
                expect(input.prop('value')).toEqual(defaultProps.name);
            });

            it('marks as error', () => {
                const formGroup = wrapperErrors.find('div.group-name');
                expect(formGroup.exists()).toBeTruthy();
                expect(formGroup.hasClass('has-error')).toBeTruthy();
            });

            it('displays help block', () => {
                const formGroup = wrapperErrors.find('div.group-name');
                const helpBlock = formGroup.find('.help-block');
                expect(helpBlock.exists()).toBeTruthy();
                expect(helpBlock.text()).toEqual(errors.name);
            });
        });

        describe('description textarea', () => {
            it('renders with passed value', () => {
                const textarea = wrapper.find('textarea.description');
                expect(textarea.exists()).toBeTruthy();
                expect(textarea.prop('value')).toEqual(defaultProps.description);
            });

            it('marks as error', () => {
                const formGroup = wrapperErrors.find('div.group-description');
                expect(formGroup.exists()).toBeTruthy();
                expect(formGroup.hasClass('has-error')).toBeTruthy();
            });

            it('displays help block', () => {
                const formGroup = wrapperErrors.find('div.group-description');
                const helpBlock = formGroup.find('.help-block');
                expect(helpBlock.exists()).toBeTruthy();
                expect(helpBlock.text()).toEqual(errors.description);
            });
        });

        describe('<DurationInput />', () => {
            it('renders', () => {
                const duration = wrapper.find('DurationInput');
                expect(duration.exists()).toBeTruthy();
            });

            it('passes the duration', () => {
                const duration = wrapper.find('DurationInput');
                expect(duration.prop('duration')).toEqual(defaultProps.duration);
            });

            it('passes the error', () => {
                const duration = wrapperErrors.find('DurationInput');
                expect(duration.prop('error')).toEqual(errors.duration);
            });
        });

        describe('interval input', () => {
            it('renders with passed value', () => {
                const input = wrapper.find('input.interval');
                expect(input.exists()).toBeTruthy();
                expect(input.prop('value')).toEqual(defaultProps.interval);
            });

            it('marks as error', () => {
                const formGroup = wrapperErrors.find('div.group-interval');
                expect(formGroup.exists()).toBeTruthy();
                expect(formGroup.hasClass('has-error')).toBeTruthy();
            });

            it('displays help block', () => {
                const formGroup = wrapperErrors.find('div.group-interval');
                const helpBlock = formGroup.find('.help-block');
                expect(helpBlock.exists()).toBeTruthy();
                expect(helpBlock.text()).toEqual(errors.interval);
            });
        });

        describe('scanner select', () => {
            it('renders with passed value', () => {
                const select = wrapper.find('select.scanner');
                expect(select.exists()).toBeTruthy();
                expect(select.prop('value')).toEqual('hoho');
            });

            it('renders scanners as options', () => {
                const select = wrapper.find('select.scanner');
                const options = select.find('option');
                expect(options.length).toEqual(2);
            });

            it('renders first scanner in alphabetical order first', () => {
                const select = wrapper.find('select.scanner');
                const options = select.find('option');
                expect(options.at(0).text()).toEqual('Npm (offline, occupied)');
                expect(options.at(0).prop('value')).toEqual('haha');
            });

            it('renders second scanner in alphabetical order second', () => {
                const select = wrapper.find('select.scanner');
                const options = select.find('option');
                expect(options.at(1).text()).toEqual('Tox (online, free)');
                expect(options.at(1).prop('value')).toEqual('hoho');
            });

            it('marks as error', () => {
                const formGroup = wrapperErrors.find('div.group-scanner');
                expect(formGroup.exists()).toBeTruthy();
                expect(formGroup.hasClass('has-error')).toBeTruthy();
            });

            it('displays help block', () => {
                const formGroup = wrapperErrors.find('div.group-scanner');
                const helpBlock = formGroup.find('.help-block');
                expect(helpBlock.exists()).toBeTruthy();
                expect(helpBlock.text()).toEqual(errors.scanner);
            });
        });
    });
});
