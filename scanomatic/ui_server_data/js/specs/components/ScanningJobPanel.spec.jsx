import { shallow } from 'enzyme';
import React from 'react';

import './enzyme-setup';
import ScanningJobPanel from '../../src/components/ScanningJobPanel';


describe('<ScanningJobPanel />', () => {
    const onStartJob = jasmine.createSpy('onStartJob');
    const props = {
        name: 'Omnibus',
        identifier: 'job0000',
        duration: { days: 3, hours: 2, minutes: 51 },
        interval: 13,
        scannerId: 'hoho',
        onStartJob,
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

    it('should render a panel-title with the name', () => {
        const wrapper = shallow(<ScanningJobPanel {...props} />);
        const title = wrapper.find('h3.panel-title');
        expect(title.exists()).toBeTruthy();
        expect(title.text()).toContain(props.name);
    });

    it('should render a job-start button', () => {
        const wrapper = shallow(<ScanningJobPanel {...props} scanner={scanner} />);
        const btn = wrapper.find('button.job-start');
        expect(btn.exists()).toBeTruthy();
        expect(btn.text()).toContain('Start');
        expect(btn.find('span.glyphicon-play').exists()).toBeTruthy();
    });

    it('should disable job-start button if scanner unknown', () => {
        const wrapper = shallow(<ScanningJobPanel
            {...props}
            scanner={null}
        />);
        const btn = wrapper.find('button.job-start');
        expect(btn.prop('disabled')).toBeTruthy();
        expect(btn.find('span.glyphicon-ban-circle').exists()).toBeTruthy();
    });

    it('should disable job-start button if scanner offline', () => {
        const wrapper = shallow(<ScanningJobPanel
            {...props}
            scanner={scannerOffline}
        />);
        const btn = wrapper.find('button.job-start');
        expect(btn.prop('disabled')).toBeTruthy();
        expect(btn.find('span.glyphicon-ban-circle').exists()).toBeTruthy();
    });

    it('should disable job-start button if scanner ownded', () => {
        const wrapper = shallow(<ScanningJobPanel
            {...props}
            scanner={scannerOccupied}
        />);
        const btn = wrapper.find('button.job-start');
        expect(btn.prop('disabled')).toBeTruthy();
        expect(btn.find('span.glyphicon-ban-circle').exists()).toBeTruthy();
    });

    it('should not render a start button if job is starting', () => {
        const wrapper = shallow(<ScanningJobPanel {...props} disableStart />);
        const btn = wrapper.find('button.job-start');
        expect(btn.exists()).toBeFalsy();
    });

    it('should not render a start button if job has started', () => {
        const wrapper = shallow(<ScanningJobPanel
            {...props}
            startTime="1980-03-23T13:00:00Z"
        />);
        const btn = wrapper.find('button.job-start');
        expect(btn.exists()).toBeFalsy();
    });

    it('should render the description', () => {
        const wrapper = shallow(<ScanningJobPanel {...props} />);
        const desc = wrapper.find('div.job-description');
        expect(desc.exists()).toBeTruthy();
    });

    it('should render the interval', () => {
        const wrapper = shallow(<ScanningJobPanel {...props} />);
        const desc = wrapper.find('div.job-description');
        expect(desc.text()).toContain('Scan every 13 minutes');
    });

    it('should render a link to the compile page', () => {
        const wrapper = shallow(<ScanningJobPanel {...props} />);
        const link = wrapper
            .find('[href="/compile?projectdirectory=root/job0000"]');
        expect(link.exists()).toBe(true);
        expect(link.text()).toEqual('Compile project');
    });

    it('should render a link to the qc page', () => {
        const wrapper = shallow(<ScanningJobPanel {...props} />);
        const link = wrapper
            .find('[href="/qc_norm?analysisdirectory=job0000/analysis&project=Omnibus"]');
        expect(link.exists()).toBe(true);
        expect(link.text()).toEqual('QC project');
    });

    describe('Scan verb', () => {
        it('should say Scan if not started', () => {
            const wrapper = shallow(<ScanningJobPanel {...props} />);
            const desc = wrapper.find('div.job-description');
            expect(desc.text()).toContain('Scan every');
        });

        it('should say Scanning if started', () => {
            const wrapper = shallow(<ScanningJobPanel
                {...props}
                startTime="1980-03-23T13:00:00Z"
            />);
            const desc = wrapper.find('div.job-description');
            expect(desc.text()).toContain('Scanning every');
        });

        it('should say Scanning if starting', () => {
            const wrapper = shallow(<ScanningJobPanel {...props} disableStart />);
            const desc = wrapper.find('div.job-description');
            expect(desc.text()).toContain('Scanning every');
        });
    });

    describe('duration', () => {
        it('should render the duration', () => {
            const wrapper = shallow(<ScanningJobPanel {...props} />);
            const desc = wrapper.find('div.job-description');
            expect(desc.text()).toContain('for 3 days 2 hours 51 minutes.');
        });

        it('should skip days if zero', () => {
            const wrapper = shallow(<ScanningJobPanel
                {...props}
                duration={{ days: 0, hours: 2, minutes: 51 }}
            />);
            const desc = wrapper.find('div.job-description');
            expect(desc.text()).toContain('for 2 hours 51 minutes.');
        });

        it('should skip days if zero', () => {
            const wrapper = shallow(<ScanningJobPanel
                {...props}
                duration={{ days: 2, hours: 0, minutes: 51 }}
            />);
            const desc = wrapper.find('div.job-description');
            expect(desc.text()).toContain('for 2 days 51 minutes.');
        });

        it('should skip days if zero', () => {
            const wrapper = shallow(<ScanningJobPanel
                {...props}
                duration={{ days: 3, hours: 2, minutes: 0 }}
            />);
            const desc = wrapper.find('div.job-description');
            expect(desc.text()).toContain('for 3 days 2 hours.');
        });
    });

    describe('scanner', () => {
        it('should render the scanner retrieving scanner status', () => {
            const wrapper = shallow(<ScanningJobPanel {...props} />);
            const desc = wrapper.find('div.scanner-status');
            expect(desc.text()).toContain('Retrieving scanner status...');
        });

        it('should render the scanner offline', () => {
            const wrapper = shallow(<ScanningJobPanel
                {...props}
                scanner={scannerOffline}
            />);
            const desc = wrapper.find('div.scanner-status');
            expect(desc.text()).toContain('Using scanner Consule (offline, free).');
        });

        it('should render the scanner occupied', () => {
            const wrapper = shallow(<ScanningJobPanel
                {...props}
                scanner={scannerOccupied}
            />);
            const desc = wrapper.find('div.scanner-status');
            expect(desc.text()).toContain('Using scanner Consule (online, occupied).');
        });
    });

    it('should say when a job was started', () => {
        const wrapper = shallow(<ScanningJobPanel
            {...props}
            startTime="1980-03-23T13:00:00Z"
        />);
        const desc = wrapper.find('div.job-status');
        expect(desc.text()).toContain('Started at 1980-03-23T13:00:00Z.');
    });
});
