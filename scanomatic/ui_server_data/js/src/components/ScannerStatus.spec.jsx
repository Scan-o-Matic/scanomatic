import { shallow } from 'enzyme';
import React from 'react';
import Timeline from 'react-calendar-timeline/lib';

import './enzyme-setup';
import ScannersStatus from './ScannersStatus';

const milliPerDay = 1000 * 3600 * 24;
const milliPerHour = 1000 * 3600;

describe('<ScannersStatus />', () => {
    describe('No scanners', () => {
        it('renders an alert', () => {
            const wrapper = shallow(<ScannersStatus scanners={[]} jobs={[]} />);
            const alert = wrapper.find('.alert');
            expect(alert.exists()).toBeTruthy();
            expect(alert.hasClass('alert-danger'));
            expect(alert.text()).toContain('No scanners attached');
        });
    });

    describe('With scanners', () => {
        const props = {
            scanners: [
                {
                    id: 'scanner001',
                    name: 'Resourceful Red robin',
                    isOnline: true,
                },
                {
                    id: 'scanner002',
                    name: 'Lazy Lark',
                    isOnline: false,
                },
                {
                    id: 'scanner003',
                    name: 'Boiserous Bowerbird',
                    isOnline: true,
                },
            ],
            jobs: [
                {
                    id: 'job001',
                    name: 'testing stuff',
                    scannerId: 'scanner001',
                    started: new Date().getTime() - (2 * milliPerDay),
                    end: new Date().getTime() + milliPerDay,
                },
                {
                    id: 'job002',
                    name: 'over and done',
                    scannerId: 'scanner003',
                    started: new Date().getTime() - (2 * milliPerDay),
                    stopped: new Date().getTime() - (milliPerHour),
                    end: new Date().getTime() + milliPerDay,
                },
                {
                    id: 'job003',
                    name: 'old one',
                    scannerId: 'scanner003',
                    started: new Date().getTime() - (5 * milliPerDay),
                    end: new Date().getTime() - (2 * milliPerDay) - (2 * milliPerHour),
                },
            ],
        };
        let wrapper;
        let now;

        beforeEach(() => {
            jasmine.clock().install();
            now = new Date();
            jasmine.clock().mockDate(now);
            wrapper = shallow(<ScannersStatus {...props} />);
        });

        afterEach(() => {
            jasmine.clock().uninstall();
        });

        describe('<Timeline />', () => {
            it('renders', () => {
                const tl = wrapper.find(Timeline);
                expect(tl.exists()).toBeTruthy();
            });

            it('passes scanners as groups', () => {
                const tl = wrapper.find(Timeline);
                expect(tl.prop('groups')).toEqual([
                    {
                        id: 'scanner003',
                        title: 'Boiserous Bowerbird',
                        rightTitle: <span className="label label-info">On</span>,
                    },
                    {
                        id: 'scanner002',
                        title: 'Lazy Lark',
                        rightTitle: <span className="label label-danger">Off</span>,
                    },
                    {
                        id: 'scanner001',
                        title: 'Resourceful Red robin',
                        rightTitle: <span className="label label-info">On</span>,
                    },
                ]);
            });

            it('passes jobs as items', () => {
                const tl = wrapper.find(Timeline);
                expect(tl.prop('items')).toEqual([
                    jasmine.objectContaining({
                        id: 'job001',
                        group: 'scanner001',
                        title: 'testing stuff',
                        className: 'scanner-job scanner-job-running',
                    }),
                    jasmine.objectContaining({
                        id: 'job002',
                        group: 'scanner003',
                        title: 'over and done',
                        className: 'scanner-job scanner-job-stopped',
                    }),
                    jasmine.objectContaining({
                        id: 'job003',
                        group: 'scanner003',
                        title: 'old one',
                        className: 'scanner-job scanner-job-ended',
                    }),
                ]);
            });

            it('uses the correct times for items (stop time if stopped as end)', () => {
                const tl = wrapper.find(Timeline);
                expect(tl.prop('items')).toEqual([
                    jasmine.objectContaining({
                        start_time: props.jobs[0].started,
                        end_time: props.jobs[0].end,
                    }),
                    jasmine.objectContaining({
                        start_time: props.jobs[1].started,
                        end_time: props.jobs[1].stopped,
                    }),
                    jasmine.objectContaining({
                        start_time: props.jobs[2].started,
                        end_time: props.jobs[2].end,
                    }),
                ]);
            });

            it('sets default view around now', () => {
                const tl = wrapper.find(Timeline);
                expect(tl.prop('defaultTimeStart')).toBeLessThan(now);
                expect(tl.prop('defaultTimeEnd')).toBeGreaterThan(now);
            });

            it('sets Scanner and Status column headers', () => {
                const tl = wrapper.find(Timeline);
                expect(tl.prop('sidebarContent')).toEqual('Scanner');
                expect(tl.prop('rightSidebarContent')).toEqual('Status');
            });
        });

        describe('legend', () => {
            let legend;
            beforeEach(() => {
                legend = wrapper.find('.legend');
            });

            it('renders', () => {
                expect(legend.exists()).toBeTruthy();
                expect(legend.find('.panel').exists()).toBeTruthy();
            });

            it('renders with heading', () => {
                const heading = legend.find('.panel-heading');
                expect(heading.exists()).toBeTruthy();
                expect(heading.text()).toEqual('Legend');
            });

            it('renders running job description', () => {
                const span = legend.find('.legend-job-running');
                expect(span.exists()).toBeTruthy();
                expect(span.text()).toEqual('Running Job');
            });

            it('renders stopped job description', () => {
                const span = legend.find('.legend-job-stopped');
                expect(span.exists()).toBeTruthy();
                expect(span.text()).toEqual('Stopped Job');
            });

            it('renders ended job description', () => {
                const span = legend.find('.legend-job-ended');
                expect(span.exists()).toBeTruthy();
                expect(span.text()).toEqual('Ended Job');
            });
        });
    });
});
