import { shallow } from 'enzyme';
import React from 'react';

import './enzyme-setup';
import ProjectPanel from './ProjectPanel';

describe('<ProjectPanel />', () => {
    let panel;
    beforeEach(() => {
        const wrapper = shallow(<ProjectPanel
            name="Test"
            description="Debugging the system."
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
        expect(panelHeading.text()).toEqual('Test');
    });

    it('readners a panel-body with the description', () => {
        const panelBody = panel.find('.panel-body');
        expect(panelBody.exists()).toBeTruthy();
        expect(panelBody.text()).toEqual('Debugging the system.');
    });
});
