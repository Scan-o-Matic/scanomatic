import { shallow } from 'enzyme';
import React from 'react';

import './enzyme-setup';
import ScanningJobPanel from '../../src/components/ScanningJobPanel';


describe('<ScanningJobPanel />', () => {
    const job = {
        name: 'Omnibus',
        duration: { days: 3, hours: 2, minutes: 51 },
        interval: 13,
        scannerId: 'hoho',
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

    it('should render a panel-title with the name', () => {
        const wrapper = shallow(<ScanningJobPanel {...job} />);
        const title = wrapper.find('h3.panel-title');
        expect(title.exists()).toBeTruthy();
        expect(title.text()).toContain(job.name);
    });

    it('should render a job-start button', () => {
        const wrapper = shallow(<ScanningJobPanel {...job} scanner={scanner} />);
        const btn = wrapper.find('button.job-start');
        expect(btn.exists()).toBeTruthy();
        expect(btn.text()).toContain('Start');
        expect(btn.find('span.glyphicon-play').exists()).toBeTruthy();
    });

    it('should disable job-start button if scanner unknown', () => {
        const wrapper = shallow(<ScanningJobPanel
            {...job}
            scanner={null}
        />);
        const btn = wrapper.find('button.job-start');
        expect(btn.prop('disabled')).toBeTruthy();
        expect(btn.find('span.glyphicon-ban-circle').exists()).toBeTruthy();
    });

    it('should disable job-start button if scanner offline', () => {
        const wrapper = shallow(<ScanningJobPanel
            {...job}
            scanner={scannerOffline}
        />);
        const btn = wrapper.find('button.job-start');
        expect(btn.prop('disabled')).toBeTruthy();
        expect(btn.find('span.glyphicon-ban-circle').exists()).toBeTruthy();
    });

    it('should disable job-start button if scanner ownded', () => {
        const wrapper = shallow(<ScanningJobPanel
            {...job}
            scanner={scannerOccupied}
        />);
        const btn = wrapper.find('button.job-start');
        expect(btn.prop('disabled')).toBeTruthy();
        expect(btn.find('span.glyphicon-ban-circle').exists()).toBeTruthy();
    });

    it('should render the description', () => {
        const wrapper = shallow(<ScanningJobPanel {...job} />);
        const desc = wrapper.find('div.job-description');
        expect(desc.exists()).toBeTruthy();
    });

    it('should render the interval', () => {
        const wrapper = shallow(<ScanningJobPanel {...job} />);
        const desc = wrapper.find('div.job-description');
        expect(desc.text()).toContain('Scan every 13 minutes');
    });

    describe('duration', () => {
        it('should render the duration', () => {
            const wrapper = shallow(<ScanningJobPanel {...job} />);
            const desc = wrapper.find('div.job-description');
            expect(desc.text()).toContain('for 3 days 2 hours 51 minutes.');
        });

        it('should skip days if zero', () => {
            const wrapper = shallow(<ScanningJobPanel
                {...job}
                duration={{ days: 0, hours: 2, minutes: 51 }}
            />);
            const desc = wrapper.find('div.job-description');
            expect(desc.text()).toContain('for 2 hours 51 minutes.');
        });

        it('should skip days if zero', () => {
            const wrapper = shallow(<ScanningJobPanel
                {...job}
                duration={{ days: 2, hours: 0, minutes: 51 }}
            />);
            const desc = wrapper.find('div.job-description');
            expect(desc.text()).toContain('for 2 days 51 minutes.');
        });

        it('should skip days if zero', () => {
            const wrapper = shallow(<ScanningJobPanel
                {...job}
                duration={{ days: 3, hours: 2, minutes: 0 }}
            />);
            const desc = wrapper.find('div.job-description');
            expect(desc.text()).toContain('for 3 days 2 hours.');
        });
    });

    describe('scanner', () => {
        it('should render the scanner retrieving scanner status', () => {
            const wrapper = shallow(<ScanningJobPanel {...job} />);
            const desc = wrapper.find('div.scanner-status');
            expect(desc.text()).toContain('Retrieving scanner status...');
        });

        it('should render the scanner offline', () => {
            const wrapper = shallow(<ScanningJobPanel
                {...job}
                scanner={scannerOffline}
            />);
            const desc = wrapper.find('div.scanner-status');
            expect(desc.text()).toContain('Using scanner Consule (offline, free).');
        });

        it('should render the scanner occupied', () => {
            const wrapper = shallow(<ScanningJobPanel
                {...job}
                scanner={scannerOccupied}
            />);
            const desc = wrapper.find('div.scanner-status');
            expect(desc.text()).toContain('Using scanner Consule (online, occupied).');
        });
    });
});
