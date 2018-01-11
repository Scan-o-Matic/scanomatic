import { mount } from 'enzyme';
import React from 'react';

import './enzyme-setup';
import ColonyImage from '../../src/components/ColonyImage';

describe('<ColonyImage/>', () => {
    const data = {
        image: [[0, 1], [1, 0]],
        imageMin: 0,
        imageMax: 1,
        blob: [[true, false], [false, true]],
        background: [[false, true], [true, false]],
    };

    it('should render 1 <canvas/>', () => {
        const wrapper = mount(<ColonyImage data={data} />);
        expect(wrapper.find('canvas').length).toEqual(1);
    });

    it('should render no <button />', () => {
        const wrapper = mount(<ColonyImage data={data} />);
        expect(wrapper.find('button').length).toEqual(0);
    });


    describe('when draw=true', () => {
        it('should render 2 <canvas/>', () => {
            const wrapper = mount(<ColonyImage data={data} draw/>);
            expect(wrapper.find('canvas').length).toEqual(2);
        });

        it('should render a "+", "-" and "Update" buttons', () => {
            const wrapper = mount(<ColonyImage data={data} draw/>);
            const buttons = wrapper.find('button')
            expect(buttons.length).toEqual(3);
            expect(buttons.at(0).text()).toEqual('+');
            expect(buttons.at(1).text()).toEqual('-');
            expect(buttons.at(2).text()).toEqual('Update');
        });

        it('should call the onUpdate callback when the "Update" button is clicked', () => {
            const onUpdate = jasmine.createSpy('onUpdate');
            const wrapper = mount(<ColonyImage data={data} onUpdate={onUpdate} draw/>);
            const button = wrapper.find('button.btn-update');
            button.simulate('click');
            expect(onUpdate).toHaveBeenCalled();
        });
    });
});
