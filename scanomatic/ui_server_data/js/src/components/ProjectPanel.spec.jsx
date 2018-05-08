import { shallow } from 'enzyme';
import React from 'react';

import './enzyme-setup';
import ProjectPanel from './ProjectPanel';

describe('<ProjectPanel />', () => {
    let panel;
    let wrapper;
    const onNewExperiment = jasmine.createSpy('onNewExperiment');

    describe('default expanded', () => {
        beforeEach(() => {
            onNewExperiment.calls.reset();
            wrapper = shallow(<ProjectPanel
                name="Test"
                description="Debugging the system."
                onNewExperiment={onNewExperiment}
                defaultExpanded
            />);
            panel = wrapper.find('.panel');
        });

        it('renders a panel', () => {
            expect(panel.exists()).toBeTruthy();
            expect(panel.prop('data-projectname')).toEqual('Test');
        });

        it('renders a panel-heading with name', () => {
            const panelHeading = panel.find('.panel-heading');
            expect(panelHeading.exists()).toBeTruthy();
            expect(panelHeading.text()).toEqual(' Test');
        });

        it('renders with glyphicon-collapse-up', () => {
            const panelHeading = wrapper.find('.panel-heading');
            expect(panelHeading.find('.glyphicon-collapse-up').exists()).toBeTruthy();
        });

        it('renders a panel-body with the description', () => {
            const panelBody = panel.find('.panel-body');
            expect(panelBody.exists()).toBeTruthy();
            const description = panelBody.find('.project-description');
            expect(description.exists()).toBeTruthy();
            expect(description.text()).toEqual('Debugging the system.');
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

        describe('Add Experiment button', () => {
            it('renders', () => {
                const btn = panel.find('.panel-body').find('.new-experiment');
                expect(btn.exists()).toBeTruthy();
                expect(btn.hasClass('btn')).toBeTruthy();
                expect(btn.text()).toEqual(' New Experiment');
            });

            it('calls onNewExperiment with project name', () => {
                const btn = panel.find('.panel-body').find('.new-experiment');
                btn.simulate('click');
                expect(onNewExperiment).toHaveBeenCalledWith('Test');
            });
        });
    });

    describe('implicit default collapsed', () => {
        beforeEach(() => {
            onNewExperiment.calls.reset();
            wrapper = shallow(<ProjectPanel
                name="Test"
                description="Debugging the system."
                onNewExperiment={onNewExperiment}
            />);
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

    describe('explicit default collapsed', () => {
        beforeEach(() => {
            onNewExperiment.calls.reset();
            wrapper = shallow(<ProjectPanel
                name="Test"
                description="Debugging the system."
                onNewExperiment={onNewExperiment}
                defaultExpanded={false}
            />);
        });

        it('renders with glyphicon-collapse-down', () => {
            const panelHeading = wrapper.find('.panel-heading');
            expect(panelHeading.find('.glyphicon-collapse-down').exists()).toBeTruthy();
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
