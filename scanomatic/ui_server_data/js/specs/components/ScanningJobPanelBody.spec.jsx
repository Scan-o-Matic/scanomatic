import { shallow } from 'enzyme';
import React from 'react';

import './enzyme-setup';
import ScanningJobPanelBody, { duration2milliseconds, getProgress }
    from '../../src/components/ScanningJobPanelBody';


describe('<ScanningJobPanelBody />', () => {
    const onStartJob = jasmine.createSpy('onStartJob');
    const props = {
        name: 'Omnibus',
        identifier: 'job0000',
        duration: { days: 3, hours: 2, minutes: 51 },
        interval: 13,
        scannerId: 'hoho',
        status: 'Planned',
        onStartJob,
        onRemoveJob: () => {},
    };

    const scanner = {
        name: 'Consule',
        owned: false,
        power: true,
        identifier: 'hoho',
    };

    const scannerOffline = {
        name: 'Consule',
        owned: false,
        power: false,
        identifier: 'hoho',
    };

    const scannerOccupied = {
        name: 'Consule',
        owned: true,
        power: true,
        identifier: 'hoho',
    };

    beforeEach(() => {
        onStartJob.calls.reset();
    });

    describe('Status Planned', () => {
        it('should render a job-start button', () => {
            const wrapper = shallow(<ScanningJobPanelBody {...props} scanner={scanner} />);
            const btn = wrapper.find('button.job-start');
            expect(btn.exists()).toBeTruthy();
            expect(btn.text()).toContain('Start');
            expect(btn.find('span.glyphicon-play').exists()).toBeTruthy();
        });

        it('should disable job-start button if scanner unknown', () => {
            const wrapper = shallow(<ScanningJobPanelBody
                {...props}
                scanner={null}
            />);
            const btn = wrapper.find('button.job-start');
            expect(btn.prop('disabled')).toBeTruthy();
            expect(btn.find('span.glyphicon-ban-circle').exists()).toBeTruthy();
        });

        it('should disable job-start button if scanner offline', () => {
            const wrapper = shallow(<ScanningJobPanelBody
                {...props}
                scanner={scannerOffline}
            />);
            const btn = wrapper.find('button.job-start');
            expect(btn.prop('disabled')).toBeTruthy();
            expect(btn.find('span.glyphicon-ban-circle').exists()).toBeTruthy();
        });

        it('should disable job-start button if scanner ownded', () => {
            const wrapper = shallow(<ScanningJobPanelBody
                {...props}
                scanner={scannerOccupied}
            />);
            const btn = wrapper.find('button.job-start');
            expect(btn.prop('disabled')).toBeTruthy();
            expect(btn.find('span.glyphicon-ban-circle').exists()).toBeTruthy();
        });

        it('should not render a start button if job is starting', () => {
            const wrapper = shallow(<ScanningJobPanelBody {...props} disableStart />);
            const btn = wrapper.find('button.job-start');
            expect(btn.find('span.glyphicon-ban-circle').exists()).toBeTruthy();
        });

        it('should not render a link to the compile page', () => {
            const wrapper = shallow(<ScanningJobPanelBody {...props} />);
            const link = wrapper
                .find('[href="/compile?projectdirectory=root/job0000"]');
            expect(link.exists()).toBeFalsy();
        });

        it('should not render a link to the qc page', () => {
            const wrapper = shallow(<ScanningJobPanelBody {...props} />);
            const link = wrapper
                .find('[href="/qc_norm?analysisdirectory=job0000/analysis&project=Omnibus"]');
            expect(link.exists()).toBeFalsy();
        });

        it('should say scanning frequency', () => {
            const wrapper = shallow(<ScanningJobPanelBody {...props} />);
            const desc = wrapper.find('tr.job-interval');
            expect(desc.exists()).toBeTruthy();
            expect(desc.find('td').at(0).text()).toEqual('Frequency');
            expect(desc.find('td').at(1).text()).toEqual('Scan every 13 minutes');
        });

        describe('scanner', () => {
            it('should have a table row', () => {
                const wrapper = shallow(<ScanningJobPanelBody {...props} />);
                const desc = wrapper.find('tr.job-scanner');
                expect(desc.exists()).toBeTruthy();
                expect(desc.find('td').at(0).text()).toEqual('Scanner');
            });

            it('should render the scanner retrieving scanner status', () => {
                const wrapper = shallow(<ScanningJobPanelBody {...props} />);
                const desc = wrapper.find('tr.job-scanner');
                expect(desc.find('td').at(1).text()).toEqual('Retrieving scanner status...');
            });

            it('should render the scanner offline', () => {
                const wrapper = shallow(<ScanningJobPanelBody
                    {...props}
                    scanner={scannerOffline}
                />);
                const desc = wrapper.find('tr.job-scanner');
                expect(desc.find('td').at(1).text()).toEqual('Consule (offline, free)');
            });

            it('should render the scanner occupied', () => {
                const wrapper = shallow(<ScanningJobPanelBody
                    {...props}
                    scanner={scannerOccupied}
                />);
                const desc = wrapper.find('tr.job-scanner');
                expect(desc.find('td').at(1).text()).toEqual('Consule (online, occupied)');
            });
        });

        it('should render a remove button', () => {
            const wrapper = shallow(<ScanningJobPanelBody
                {...props}
                status="Planned"
                identifier="scnjb001"
            />);
            const btn = wrapper.find('ScanningJobRemoveButton');
            expect(btn.exists()).toBeTruthy();
            expect(btn.prop('identifier')).toEqual('scnjb001');
            expect(btn.prop('onRemoveJob')).toBe(props.onRemoveJob);
        });
    });

    describe('Status Running', () => {
        let wrapper;

        beforeEach(() => {
            jasmine.clock().install();
            jasmine.clock().mockDate(new Date('1980-03-24T13:00:00Z'));
            wrapper = shallow(<ScanningJobPanelBody
                {...props}
                startTime={new Date('1980-03-23T13:00:00Z')}
                endTime={new Date('1980-03-26T15:51:00Z')}
                status="Running"
            />);
        });

        afterEach(() => {
            jasmine.clock().uninstall();
        });

        it('should not render a start button', () => {
            const btn = wrapper.find('button.job-start');
            expect(btn.exists()).toBeFalsy();
        });

        it('should not render a link to the compile page', () => {
            const link = wrapper
                .find('[href="/compile?projectdirectory=root/job0000"]');
            expect(link.exists()).toBeFalsy();
        });

        it('should not render a link to the qc page', () => {
            const link = wrapper
                .find('[href="/qc_norm?analysisdirectory=job0000/analysis&project=Omnibus"]');
            expect(link.exists()).toBeFalsy();
        });

        it('should say when a job was started', () => {
            const desc = wrapper.find('tr.job-start');
            expect(desc.exists()).toBeTruthy();
            expect(desc.find('td').at(0).text()).toEqual('Started');
            expect(desc.find('td').at(1).text()).toEqual(`${new Date('1980-03-23T13:00:00Z')}`);
        });

        it('should say when a job will end', () => {
            const desc = wrapper.find('tr.job-end');
            expect(desc.exists()).toBeTruthy();
            expect(desc.find('td').at(0).text()).toEqual('Will end');
            expect(desc.find('td').at(1).text()).toEqual(`${new Date('1980-03-26T15:51:00Z')}`);
        });

        it('should say scanning frequency', () => {
            const desc = wrapper.find('tr.job-interval');
            expect(desc.exists()).toBeTruthy();
            expect(desc.find('td').at(0).text()).toEqual('Frequency');
            expect(desc.find('td').at(1).text()).toEqual('Scanning every 13 minutes');
        });

        describe('Progress bar', () => {
            it('should render a progress bar', () => {
                const progress = wrapper.find('.progress');
                expect(progress.exists()).toBeTruthy();
            });

            it('should have the progressbar display the progress', () => {
                const progressbar = wrapper.find('.progress-bar');
                expect(progressbar.prop('style').width).toEqual('32.1%');
            });

            it('should have the progressbar display the progress as text', () => {
                const progressbar = wrapper.find('.progress-bar');
                expect(progressbar.render().text()).toEqual('32.1%');
            });
        });

        it('should have also show scanner info', () => {
            const desc = wrapper.find('tr.job-scanner');
            expect(desc.exists()).toBeTruthy();
            expect(desc.find('td').at(0).text()).toEqual('Scanner');
        });

        it('should not render a remove button', () => {
            const btn = wrapper.find('ScanningJobRemoveButton');
            expect(btn.exists()).toBeFalsy();
        });
    });

    describe('Status Completed', () => {
        let wrapper;
        beforeEach(() => {
            wrapper = shallow(<ScanningJobPanelBody
                {...props}
                startTime={new Date('1980-03-23T13:00:00Z')}
                endTime={new Date('1980-03-26T15:51:00Z')}
                status="Completed"
            />);
        });

        it('should render a link to the compile page', () => {
            const link = wrapper
                .find('[href="/compile?projectdirectory=root/job0000"]');
            expect(link.exists()).toBeTruthy();
            expect(link.text()).toEqual('Compile project');
        });

        it('should render a link to the qc page', () => {
            const link = wrapper
                .find('[href="/qc_norm?analysisdirectory=job0000/analysis&project=Omnibus"]');
            expect(link.exists()).toBeTruthy();
            expect(link.text()).toEqual('QC project');
        });

        it('should not show scanner info', () => {
            const desc = wrapper.find('tr.job-scanner');
            expect(desc.exists()).toBeFalsy();
        });

        it('should say when a job was started', () => {
            const desc = wrapper.find('tr.job-start');
            expect(desc.exists()).toBeTruthy();
            expect(desc.find('td').at(0).text()).toEqual('Started');
            expect(desc.find('td').at(1).text()).toEqual(`${new Date('1980-03-23T13:00:00Z')}`);
        });

        it('should say when a job ended', () => {
            const desc = wrapper.find('tr.job-end');
            expect(desc.exists()).toBeTruthy();
            expect(desc.find('td').at(0).text()).toEqual('Ended');
            expect(desc.find('td').at(1).text()).toEqual(`${new Date('1980-03-26T15:51:00Z')}`);
        });

        it('should say scanning frequency', () => {
            const desc = wrapper.find('tr.job-interval');
            expect(desc.exists()).toBeTruthy();
            expect(desc.find('td').at(0).text()).toEqual('Frequency');
            expect(desc.find('td').at(1).text()).toEqual('Scanned every 13 minutes');
        });

        it('should not render a remove button', () => {
            const btn = wrapper.find('ScanningJobRemoveButton');
            expect(btn.exists()).toBeFalsy();
        });
    });

    describe('duration', () => {
        it('should render the duration', () => {
            const wrapper = shallow(<ScanningJobPanelBody {...props} />);
            const desc = wrapper.find('tr.job-duration');
            expect(desc.find('td').at(0).text()).toEqual('Duration');
            expect(desc.find('td').at(1).text()).toEqual('3 days 2 hours 51 minutes');
        });

        it('should skip days if zero', () => {
            const wrapper = shallow(<ScanningJobPanelBody
                {...props}
                duration={{ days: 0, hours: 2, minutes: 51 }}
            />);
            const desc = wrapper.find('tr.job-duration');
            expect(desc.find('td').at(1).text()).toEqual('2 hours 51 minutes');
        });

        it('should skip hours if zero', () => {
            const wrapper = shallow(<ScanningJobPanelBody
                {...props}
                duration={{ days: 2, hours: 0, minutes: 51 }}
            />);
            const desc = wrapper.find('tr.job-duration');
            expect(desc.find('td').at(1).text()).toEqual('2 days 51 minutes');
        });

        it('should skip minues if zero', () => {
            const wrapper = shallow(<ScanningJobPanelBody
                {...props}
                duration={{ days: 3, hours: 2, minutes: 0 }}
            />);
            const desc = wrapper.find('tr.job-duration');
            expect(desc.find('td').at(1).text()).toEqual('3 days 2 hours');
        });
    });
});

describe('duration2milliseconds', () => {
    it('should convert a minutes to 60000', () => {
        expect(duration2milliseconds({ minutes: 2, hours: 0, days: 0 }))
            .toEqual(120000);
    });

    it('should convert an hour to 3600000', () => {
        expect(duration2milliseconds({ minutes: 0, hours: 1, days: 0 }))
            .toEqual(3600000);
    });

    it('should convert a day to 8.64 x 10^7', () => {
        expect(duration2milliseconds({ minutes: 0, hours: 0, days: 1 }))
            .toEqual(86400000);
    });

    it('should sum it up', () => {
        expect(duration2milliseconds({ minutes: 1, hours: 1, days: 1 }))
            .toEqual(90060000);
    });

    it('should return 0 if no duration', () => {
        expect(duration2milliseconds()).toEqual(0);
    });
});

describe('getProgress', () => {
    beforeEach(() => {
        jasmine.clock().install();
        jasmine.clock().mockDate(new Date('2040-05-02T12:00:00.000Z'));
    });

    afterEach(() => {
        jasmine.clock().uninstall();
    });

    it('should cap duration at 100', () => {
        expect(getProgress({
            startTime: new Date('1024-07-02T00:00:00.000Z'),
            duration: { days: 666, hours: 66, minutes: 6 },
        })).toEqual(100);
    });

    it('should return percent completed', () => {
        expect(getProgress({
            startTime: new Date('2040-05-01T12:00:00.000Z'),
            duration: { days: 2, hours: 0, minutes: 0 },
        })).toEqual(50);
    });
});
