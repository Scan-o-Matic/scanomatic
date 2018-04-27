import { shallow } from 'enzyme';
import React from 'react';

import '../enzyme-setup';
import ExperimentPanel from './index';

describe('<ExperimentPanel />', () => {
    let wrapper;
    const props = {
        name: 'My name is',
        description: 'blablabla',
        interval: 60000,
        duration: 1200000,
        status: 'Planned',
        scanner: {
            identifier: 'myScanner',
            name: 'myScanner',
            owned: true,
            power: true,
        },
    };

    beforeEach(() => {
        wrapper = shallow(<ExperimentPanel {...props} />);
    });

    it('redners a heading', () => {
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
