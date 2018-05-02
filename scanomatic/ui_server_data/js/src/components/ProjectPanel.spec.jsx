import { shallow } from 'enzyme';
import React from 'react';

import './enzyme-setup';
import ProjectPanel from './ProjectPanel';

describe('<ProjectPanel />', () => {
    let panel;
    const onNewExperiment = jasmine.createSpy('onNewExperiment');

    beforeEach(() => {
        onNewExperiment.calls.reset();
        const wrapper = shallow(<ProjectPanel
            name="Test"
            description="Debugging the system."
            onNewExperiment={onNewExperiment}
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

    it('renders a panel-body with the description', () => {
        const panelBody = panel.find('.panel-body');
        expect(panelBody.exists()).toBeTruthy();
        const description = panelBody.find('.project-description');
        expect(description.exists()).toBeTruthy();
        expect(description.text()).toEqual('Debugging the system.');
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
