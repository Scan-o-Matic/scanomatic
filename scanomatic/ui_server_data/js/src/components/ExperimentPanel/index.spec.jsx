import { shallow } from 'enzyme';
import React from 'react';

import '../enzyme-setup';
import ExperimentPanel, { formatScannerStatus } from './index';

describe('<ExperimentPanel />', () => {
    const onStart = jasmine.createSpy('onStart');
    const onRemove = jasmine.createSpy('onRemove');
    const onDone = jasmine.createSpy('onDone');
    const onReopen = jasmine.createSpy('onReopen');
    const onStop = jasmine.createSpy('onStop');
    const onFeatureExtract = jasmine.createSpy('onFeatureExtract');

    beforeEach(() => {
        onStart.calls.reset();
        onRemove.calls.reset();
        onDone.calls.reset();
        onReopen.calls.reset();
        onStop.calls.reset();
        onFeatureExtract.calls.reset();
    });

    describe('default expanded', () => {
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
            pinning: new Map([[1, ''], [2, '384'], [3, ''], [4, '']]),
            onStart,
            onRemove,
            onDone,
            onReopen,
            onStop,
            onFeatureExtract,
            defaultExpanded: true,
        };

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

            it('renders with glyphicon-collapse-up', () => {
                const panelHeading = wrapper.find('.panel-heading');
                expect(panelHeading.find('.glyphicon-collapse-up').exists()).toBeTruthy();
            });

            it('toggles panel-body when panel-heading is clicked', () => {
                let panelBody = wrapper.find('.panel-body');
                const panelHeading = wrapper.find('.panel-heading');
                expect(panelBody.exists()).toBeTruthy();

                panelHeading.simulate('click');
                wrapper.update();
                panelBody = wrapper.find('.panel-body');
                expect(panelBody.exists()).toBeFalsy();

                panelHeading.simulate('click');
                wrapper.update();
                panelBody = wrapper.find('.panel-body');
                expect(panelBody.exists()).toBeTruthy();
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

            describe('pinning', () => {
                it('renders the table row', () => {
                    const table = wrapper.find('.experiment-stats');
                    const tr = table.find('.experiment-pinning');
                    expect(tr.exists()).toBeTruthy();
                    const tds = tr.find('td');
                    expect(tds.exists()).toBeTruthy();
                    expect(tds.length).toEqual(2);
                });

                it('renders the row title', () => {
                    const table = wrapper.find('.experiment-stats');
                    const tr = table.find('.experiment-pinning');
                    const tds = tr.find('td');
                    expect(tds.at(0).text()).toEqual('Pinning');
                });

                it('renders the row info', () => {
                    const table = wrapper.find('.experiment-stats');
                    const tr = table.find('.experiment-pinning');
                    const tds = tr.find('td');
                    const pinningFormats = tds.at(1).find('.pinning-format');
                    expect(pinningFormats.exists()).toBeTruthy();
                    expect(pinningFormats.length).toEqual(4);
                });

                it('renders empty plate info', () => {
                    const table = wrapper.find('.experiment-stats');
                    const tr = table.find('.experiment-pinning');
                    const tds = tr.find('td');
                    const pinningFormats = tds.at(1).find('.pinning-format');
                    expect(pinningFormats.at(0).text()).toEqual('Plate 1: ');
                    expect(pinningFormats.at(0).find('.glyphicon-ban-circle').exists()).toBeTruthy();
                });

                it('renders plate pinning info', () => {
                    const table = wrapper.find('.experiment-stats');
                    const tr = table.find('.experiment-pinning');
                    const tds = tr.find('td');
                    const pinningFormats = tds.at(1).find('.pinning-format');
                    expect(pinningFormats.at(1).text()).toEqual('Plate 2: 384');
                    expect(pinningFormats.at(1).html()).toContain('<em>384</em>');
                });
            });
        });

        describe('running', () => {
            const runningProps = {
                started: new Date(),
                end: new Date(new Date().getTime() + 1200000),
            };

            beforeEach(() => {
                wrapper = shallow(<ExperimentPanel {...props} {...runningProps} />);
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

            describe('stop button', () => {
                let btn;
                beforeEach(() => {
                    const buttons = wrapper.find('.action-buttons');
                    btn = buttons.find('.experiment-action-stop');
                });

                it('renders', () => {
                    expect(btn.exists()).toBeTruthy();
                    expect(btn.text()).toEqual(' Stop');
                    expect(btn.find('.glyphicon-stop').exists()).toBeTruthy();
                });

                it('formats correctly', () => {
                    expect(btn.is('button')).toBeTruthy();
                    expect(btn.hasClass('btn')).toBeTruthy();
                    expect(btn.hasClass('btn-block')).toBeTruthy();
                });

                it('renders stop dialogue on click', () => {
                    btn.simulate('click');
                    wrapper.update();
                    expect(wrapper.find('ScanningJobStopDialogue').exists()).toBeTruthy();
                });

                it('passes a dialogue dismiss function to onCancel', () => {
                    btn.simulate('click');
                    wrapper.update();
                    wrapper.find('ScanningJobStopDialogue').prop('onCancel')();
                    wrapper.update();
                    expect(wrapper.find('ScanningJobStopDialogue').exists()).toBeFalsy();
                    expect(onStop).not.toHaveBeenCalled();
                });

                it('calls onStop on onConfirm', () => {
                    btn.simulate('click');
                    wrapper.update();
                    wrapper.find('ScanningJobStopDialogue').prop('onConfirm')('test');
                    expect(onStop).toHaveBeenCalledWith(props.id, 'test');
                });
            });

            it('renders start time', () => {
                const tr = wrapper.find('.experiment-started');
                expect(tr.exists()).toBeTruthy();
                expect(tr.find('td').at(0).text()).toEqual('Started');
                expect(tr.find('td').at(1).text()).toEqual(runningProps.started.toString());
            });

            it('renders end time', () => {
                const tr = wrapper.find('.experiment-end');
                expect(tr.exists()).toBeTruthy();
                expect(tr.find('td').at(0).text()).toEqual('Ends');
                expect(tr.find('td').at(1).text()).toEqual(runningProps.end.toString());
            });
        });

        describe('analysis', () => {
            const stoppedProps = {
                started: new Date(),
                end: new Date(new Date().getTime() + 1200000),
                stopped: new Date(new Date().getTime() + 200000),
            };

            beforeEach(() => {
                wrapper = shallow(<ExperimentPanel {...props} {...stoppedProps} />);
            });

            it('renders experiment as analysis status', () => {
                expect(wrapper.find('ScanningJobStatusLabel').prop('status'))
                    .toEqual('Analysis');
            });

            it('renders end time', () => {
                const tr = wrapper.find('.experiment-end');
                expect(tr.exists()).toBeTruthy();
                expect(tr.find('td').at(0).text()).toEqual('Ended');
                expect(tr.find('td').at(1).text()).toEqual(stoppedProps.end.toString());
            });

            it('renders stopped time', () => {
                const tr = wrapper.find('.experiment-stopped');
                expect(tr.exists()).toBeTruthy();
                expect(tr.find('td').at(0).text()).toEqual('Stopped');
                expect(tr.find('td').at(1).text()).toEqual(stoppedProps.stopped.toString());
            });

            describe('compile button', () => {
                let btn;
                beforeEach(() => {
                    const buttons = wrapper.find('.action-buttons');
                    btn = buttons.find('.experiment-action-compile');
                });

                it('renders', () => {
                    expect(btn.exists()).toBeTruthy();
                    expect(btn.text()).toEqual(' Compile');
                    expect(btn.find('.glyphicon-flash').exists()).toBeTruthy();
                });

                it('formats correctly', () => {
                    expect(btn.is('a')).toBeTruthy();
                    expect(btn.hasClass('btn')).toBeTruthy();
                    expect(btn.hasClass('btn-block')).toBeTruthy();
                });

                it('links correctly', () => {
                    expect(btn.prop('href')).toEqual('/compile?projectdirectory=root/job00010');
                });
            });

            describe('analyse button', () => {
                let btn;
                beforeEach(() => {
                    const buttons = wrapper.find('.action-buttons');
                    btn = buttons.find('.experiment-action-analyse');
                });

                it('renders', () => {
                    expect(btn.exists()).toBeTruthy();
                    expect(btn.text()).toEqual(' Analyse');
                    expect(btn.find('.glyphicon-flash').exists()).toBeTruthy();
                });

                it('formats correctly', () => {
                    expect(btn.is('a')).toBeTruthy();
                    expect(btn.hasClass('btn')).toBeTruthy();
                    expect(btn.hasClass('btn-block')).toBeTruthy();
                });

                it('links correctly', () => {
                    expect(btn.prop('href'))
                        .toEqual('/analysis?compilationfile=root/job00010/job00010.project.compilation');
                });
            });

            describe('qc button', () => {
                let btn;
                beforeEach(() => {
                    const buttons = wrapper.find('.action-buttons');
                    btn = buttons.find('.experiment-action-qc');
                });

                it('renders', () => {
                    expect(btn.exists()).toBeTruthy();
                    expect(btn.text()).toEqual(' Quality Control');
                    expect(btn.find('.glyphicon-flash').exists()).toBeTruthy();
                });

                it('formats correctly', () => {
                    expect(btn.is('a')).toBeTruthy();
                    expect(btn.hasClass('btn')).toBeTruthy();
                    expect(btn.hasClass('btn-block')).toBeTruthy();
                });

                it('links correctly', () => {
                    expect(btn.prop('href'))
                        .toEqual('/qc_norm?analysisdirectory=job00010/analysis&project=My%20name%20is');
                });
            });

            describe('extract button', () => {
                let btn;
                beforeEach(() => {
                    const buttons = wrapper.find('.action-buttons');
                    btn = buttons.find('.experiment-action-extract');
                });

                it('renders', () => {
                    expect(btn.exists()).toBeTruthy();
                    expect(btn.text()).toEqual(' Extract Features');
                    expect(btn.find('.glyphicon-flash').exists()).toBeTruthy();
                });

                it('formats correctly', () => {
                    expect(btn.is('button')).toBeTruthy();
                    expect(btn.hasClass('btn')).toBeTruthy();
                    expect(btn.hasClass('btn-block')).toBeTruthy();
                });

                it('renders extract dialogue on click', () => {
                    btn.simulate('click');
                    wrapper.update();
                    expect(wrapper.find('ScanningJobFeatureExtractDialogue').exists()).toBeTruthy();
                });

                it('passes a dialogue dismiss function to onCancel', () => {
                    btn.simulate('click');
                    wrapper.update();
                    wrapper.find('ScanningJobFeatureExtractDialogue').prop('onCancel')();
                    wrapper.update();
                    expect(wrapper.find('ScanningJobFeatureExtractDialogue').exists()).toBeFalsy();
                    expect(onStop).not.toHaveBeenCalled();
                });

                it('calls onFeatureExtract on onConfirm', () => {
                    btn.simulate('click');
                    wrapper.update();
                    wrapper.find('ScanningJobFeatureExtractDialogue').prop('onConfirm')(true);
                    expect(onFeatureExtract).toHaveBeenCalledWith(props.id, true);
                });
            });

            describe('done button', () => {
                let btn;
                beforeEach(() => {
                    const buttons = wrapper.find('.action-buttons');
                    btn = buttons.find('.experiment-action-done');
                });

                it('renders', () => {
                    expect(btn.exists()).toBeTruthy();
                    expect(btn.text()).toEqual(' Done');
                    expect(btn.find('.glyphicon-ok').exists()).toBeTruthy();
                });

                it('formats correctly', () => {
                    expect(btn.is('button')).toBeTruthy();
                    expect(btn.hasClass('btn')).toBeTruthy();
                    expect(btn.hasClass('btn-block')).toBeTruthy();
                });

                it('calls onDone on click', () => {
                    btn.simulate('click');
                    expect(onDone).toHaveBeenCalledWith(props.id);
                });
            });
        });

        describe('done', () => {
            const doneProps = {
                started: new Date(),
                end: new Date(new Date().getTime() + 1200000),
                stopped: new Date(new Date().getTime() + 200000),
                done: true,
            };

            beforeEach(() => {
                wrapper = shallow(<ExperimentPanel {...props} {...doneProps} />);
            });

            it('renders experiment as analysis status', () => {
                expect(wrapper.find('ScanningJobStatusLabel').prop('status'))
                    .toEqual('Done');
            });

            describe('reopen button', () => {
                let btn;
                beforeEach(() => {
                    const buttons = wrapper.find('.action-buttons');
                    btn = buttons.find('.experiment-action-reopen');
                });

                it('renders', () => {
                    expect(btn.exists()).toBeTruthy();
                    expect(btn.text()).toEqual(' Re-open');
                    expect(btn.find('.glyphicon-pencil').exists()).toBeTruthy();
                });

                it('formats correctly', () => {
                    expect(btn.is('button')).toBeTruthy();
                    expect(btn.hasClass('btn')).toBeTruthy();
                    expect(btn.hasClass('btn-block')).toBeTruthy();
                });

                it('calls onReopen on click', () => {
                    btn.simulate('click');
                    expect(onReopen).toHaveBeenCalledWith(props.id);
                });
            });
        });
    });

    describe('implicit default collapsed', () => {
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
            pinning: new Map([[1, ''], [2, '384'], [3, ''], [4, '']]),
            onStart,
            onRemove,
            onDone,
            onReopen,
            onStop,
            onFeatureExtract,
        };

        beforeEach(() => {
            wrapper = shallow(<ExperimentPanel {...props} />);
        });

        it('toggles panel-body when panel-heading is clicked', () => {
            let panelBody = wrapper.find('.panel-body');
            const panelHeading = wrapper.find('.panel-heading');
            expect(panelBody.exists()).toBeFalsy();

            panelHeading.simulate('click');
            wrapper.update();
            panelBody = wrapper.find('.panel-body');
            expect(panelBody.exists()).toBeTruthy();
        });

        it('renders with glyphicon-collapse-down', () => {
            const panelHeading = wrapper.find('.panel-heading');
            expect(panelHeading.find('.glyphicon-collapse-down').exists()).toBeTruthy();
        });
    });

    describe('explicit default collapsed', () => {
        let wrapper;
        const props = {
            id: 'job00010',
            name: 'My name is',
            description: 'blablabla',
            interval: 60000,
            duration: 1200000,
            done: false,
            pinning: new Map([[1, ''], [2, '384'], [3, ''], [4, '']]),
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
            defaultExpanded: false,
        };

        beforeEach(() => {
            wrapper = shallow(<ExperimentPanel {...props} />);
        });

        it('toggles panel-body when panel-heading is clicked', () => {
            let panelBody = wrapper.find('.panel-body');
            const panelHeading = wrapper.find('.panel-heading');
            expect(panelBody.exists()).toBeFalsy();

            panelHeading.simulate('click');
            wrapper.update();
            panelBody = wrapper.find('.panel-body');
            expect(panelBody.exists()).toBeTruthy();
        });
    });
});

describe('formatScannerStatus', () => {
    const scanner = {
        identifier: 'myScanner',
        name: 'myScanner',
    };

    it('returns (online, occupied) when scanner is owned and on', () => {
        scanner.owned = true;
        scanner.power = true;

        expect(formatScannerStatus(scanner)).toEqual('myScanner (online, occupied)');
    });

    it('returns (offline, occupied) when scanner is owned and off', () => {
        scanner.owned = true;
        scanner.power = false;

        expect(formatScannerStatus(scanner)).toEqual('myScanner (offline, occupied)');
    });

    it('returns (online, free) when scanner is free and on', () => {
        scanner.owned = false;
        scanner.power = true;
        expect(formatScannerStatus(scanner)).toEqual('myScanner (online, free)');
    });

    it('returns (offline, free) when scanner is free and off', () => {
        scanner.owned = false;
        scanner.power = false;

        expect(formatScannerStatus(scanner)).toEqual('myScanner (offline, free)');
    });
});
