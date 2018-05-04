import { shallow } from 'enzyme';
import React from 'react';

import '../enzyme-setup';
import ExperimentPanel from './index';

describe('<ExperimentPanel />', () => {
    const onStart = jasmine.createSpy('onStart');
    const onRemove = jasmine.createSpy('onRemove');
    const onDone = jasmine.createSpy('onDone');
    const onReopen = jasmine.createSpy('onReopen');
    const onStop = jasmine.createSpy('onStop');
    const onFeatureExtract = jasmine.createSpy('onFeatureExtract');

    let wrapper;
    const props = {
        id: 'job00010',
        name: 'My name is',
        description: 'blablabla',
        interval: 60000,
        duration: 1200000,
        done: false,
        scanner: {
            identifier: 'myScanner',
            name: 'myScanner',
            owned: true,
            power: true,
        },
        onStart,
        onRemove,
        onDone,
        onReopen,
        onStop,
        onFeatureExtract,
    };

    beforeEach(() => {
        onStart.calls.reset();
        onRemove.calls.reset();
        onDone.calls.reset();
        onReopen.calls.reset();
        onStop.calls.reset();
        onFeatureExtract.calls.reset();
    });

    describe('planned', () => {
        beforeEach(() => {
            wrapper = shallow(<ExperimentPanel {...props} />);
        });

        it('renders a heading', () => {
            const heading = wrapper.find('.panel-heading');
            expect(heading.exists()).toBeTruthy();
        });

        it('renders the name as a title in the heading', () => {
            const heading = wrapper.find('.panel-heading');
            const name = heading.find('.panel-title');
            expect(name.exists()).toBeTruthy();
            expect(name.text()).toEqual(props.name);
        });

        describe('<ScanningJobStatusLabel />', () => {
            it('renders', () => {
                const heading = wrapper.find('.panel-heading');
                expect(heading.find('ScanningJobStatusLabel').exists()).toBeTruthy();
            });

            it('passes the status', () => {
                const heading = wrapper.find('.panel-heading');
                expect(heading.find('ScanningJobStatusLabel').prop('status')).toEqual('Planned');
            });
        });

        it('renders the description', () => {
            const description = wrapper.find('.experiment-description');
            expect(description.exists()).toBeTruthy();
            expect(description.text()).toEqual('blablabla');
        });

        it('renders stats table', () => {
            const table = wrapper.find('.experiment-stats');
            expect(table.exists()).toBeTruthy();
        });

        describe('start button', () => {
            let btn;
            beforeEach(() => {
                const buttons = wrapper.find('.action-buttons');
                btn = buttons.find('.experiment-action-start');
            });

            it('renders', () => {
                expect(btn.exists()).toBeTruthy();
                expect(btn.text()).toEqual(' Start');
                expect(btn.find('.glyphicon-play').exists()).toBeTruthy();
            });

            it('formats correctly', () => {
                expect(btn.is('button')).toBeTruthy();
                expect(btn.hasClass('btn')).toBeTruthy();
                expect(btn.hasClass('btn-block')).toBeTruthy();
            });

            it('calls onStart on click', () => {
                btn.simulate('click');
                expect(onStart).toHaveBeenCalledWith(props.id);
            });
        });

        describe('remove button', () => {
            let btn;
            beforeEach(() => {
                const buttons = wrapper.find('.action-buttons');
                btn = buttons.find('.experiment-action-remove');
            });

            it('renders', () => {
                expect(btn.exists()).toBeTruthy();
                expect(btn.text()).toEqual(' Remove');
                expect(btn.find('.glyphicon-remove').exists()).toBeTruthy();
            });

            it('formats correctly', () => {
                expect(btn.prop('className')).toEqual('btn btn-default btn-block experiment-action-remove');
            });

            it('triggers dialogue when onRemoveJob is called', () => {
                btn.simulate('click');
                wrapper.update();
                const dialogue = wrapper.find('ScanningJobRemoveDialogue');
                expect(dialogue.exists()).toBeTruthy();
            });

            describe('remove dialogue', () => {
                let dialogue;
                beforeEach(() => {
                    btn.simulate('click');
                    wrapper.update();
                    dialogue = wrapper.find('ScanningJobRemoveDialogue');
                });

                it('passes the experiment name', () => {
                    expect(dialogue.prop('name')).toEqual(props.name);
                });

                it('removes dialogue onCancel action', () => {
                    dialogue.prop('onCancel')();
                    wrapper.update();
                    expect(wrapper.find('ScanningJobRemoveDialogue').exists())
                        .toBeFalsy();
                });

                it('calls onRemove on confirm', () => {
                    dialogue.prop('onConfirm')();
                    expect(onRemove).toHaveBeenCalledWith(props.id);
                });
            });
        });

        describe('duration', () => {
            it('renders the table row', () => {
                const table = wrapper.find('.experiment-stats');
                const duration = table.find('.experiment-duration');
                expect(duration.exists()).toBeTruthy();
                const tds = duration.find('td');
                expect(tds.exists()).toBeTruthy();
                expect(tds.length).toEqual(2);
            });

            it('renders the row title', () => {
                const table = wrapper.find('.experiment-stats');
                const duration = table.find('.experiment-duration');
                const tds = duration.find('td');
                expect(tds.at(0).text()).toEqual('Duration');
            });

            it('renders the row info', () => {
                const table = wrapper.find('.experiment-stats');
                const duration = table.find('.experiment-duration');
                const tds = duration.find('td');
                expect(tds.at(1).text()).toEqual('20 minutes');
            });
        });

        describe('interval', () => {
            it('renders the table row', () => {
                const table = wrapper.find('.experiment-stats');
                const interval = table.find('.experiment-interval');
                expect(interval.exists()).toBeTruthy();
                const tds = interval.find('td');
                expect(tds.exists()).toBeTruthy();
                expect(tds.length).toEqual(2);
            });

            it('renders the row title', () => {
                const table = wrapper.find('.experiment-stats');
                const interval = table.find('.experiment-interval');
                const tds = interval.find('td');
                expect(tds.at(0).text()).toEqual('Interval');
            });

            it('renders the row info', () => {
                const table = wrapper.find('.experiment-stats');
                const interval = table.find('.experiment-interval');
                const tds = interval.find('td');
                expect(tds.at(1).text()).toEqual('1 minutes');
            });
        });

        describe('scanner', () => {
            it('renders the table row', () => {
                const table = wrapper.find('.experiment-stats');
                const tr = table.find('.experiment-scanner');
                expect(tr.exists()).toBeTruthy();
                const tds = tr.find('td');
                expect(tds.exists()).toBeTruthy();
                expect(tds.length).toEqual(2);
            });

            it('renders the row title', () => {
                const table = wrapper.find('.experiment-stats');
                const tr = table.find('.experiment-scanner');
                const tds = tr.find('td');
                expect(tds.at(0).text()).toEqual('Scanner');
            });

            it('renders the row info', () => {
                const table = wrapper.find('.experiment-stats');
                const tr = table.find('.experiment-scanner');
                const tds = tr.find('td');
                expect(tds.at(1).text()).toEqual('myScanner (online, occupied)');
            });
        });
    });

    describe('running', () => {
        const runningProps = {
            started: new Date(),
            end: new Date(),
        };

        beforeEach(() => {
            wrapper = shallow(<ExperimentPanel {...props} {...runningProps} />);
        });

        it('renders no action buttons', () => {
            const buttons = wrapper.find('.action-buttons');
            expect(buttons.exists()).toBeTruthy();
            expect(buttons.children().length).toEqual(0);
        });

        describe('<ScanningJobStatusLabel />', () => {
            it('renders', () => {
                const heading = wrapper.find('.panel-heading');
                expect(heading.find('ScanningJobStatusLabel').exists()).toBeTruthy();
            });

            it('passes the status', () => {
                const heading = wrapper.find('.panel-heading');
                expect(heading.find('ScanningJobStatusLabel').prop('status')).toEqual('Running');
            });
        });
    });
});
