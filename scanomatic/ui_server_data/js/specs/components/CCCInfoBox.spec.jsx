import { shallow } from 'enzyme';
import React from 'react';

import './enzyme-setup';
import CCCInfoBox from '../../src/components/CCCInfoBox';
import cccMetadata from '../fixtures/cccMetadata';

describe('<CCCInfoBox />', () => {
    const props = { cccMetadata };

    it('should render a <table />', () => {
        const wrapper = shallow(<CCCInfoBox {...props} />);
        expect(wrapper.find('table').exists()).toBeTruthy();
    });

    it('should show the CCC id', () => {
        const wrapper = shallow(<CCCInfoBox {...props} />);
        expect(wrapper.text()).toContain(cccMetadata.id);
    });

    it('should show the CCC access token', () => {
        const wrapper = shallow(<CCCInfoBox {...props} />);
        expect(wrapper.text()).toContain(cccMetadata.accessToken);
    });

    it('should show the CCC species', () => {
        const wrapper = shallow(<CCCInfoBox {...props} />);
        expect(wrapper.text()).toContain(cccMetadata.species);
    });

    it('should show the CCC reference', () => {
        const wrapper = shallow(<CCCInfoBox {...props} />);
        expect(wrapper.text()).toContain(cccMetadata.reference);
    });

    it('should show the CCC pinning format', () => {
        const wrapper = shallow(<CCCInfoBox {...props} />);
        expect(wrapper.text()).toContain(cccMetadata.pinningFormat.name);
    });

    it('should show the CCC fixture name', () => {
        const wrapper = shallow(<CCCInfoBox {...props} />);
        expect(wrapper.text()).toContain(cccMetadata.fixtureName);
    });
});
