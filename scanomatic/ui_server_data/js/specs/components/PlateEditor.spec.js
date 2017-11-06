import { shallow } from 'enzyme';
import React  from 'react';

import './enzyme-setup';

import PlateEditor from '../../ccc/components/PlateEditor';

describe('<PlateEditor />', () => {
    let props;

    beforeEach(() => {
       props = {
            accessToken: 'T0P53CR3T',
            cccId: 'CCC42',
            imageId: '1M4G3',
            plateId: 'PL4T3',
            pinFormat: [8, 12],
            onGriddingFinish: jasmine.createSpy('onGriddingFinish'),
            onColonyFinish: jasmine.createSpy('onColonyFinish'),
        };
    });

    describe('gridding step', () => {
        beforeEach(() => {
            props.step = 'gridding';
        });

        it('should set the title to "Gridding"', () => {
            const wrapper = shallow(<PlateEditor {...props} />);
            expect(wrapper.find('h3').text()).toEqual('Step 2: Gridding');
        });

        it('should render a <GriddingContainer />', () => {
            const wrapper = shallow(<PlateEditor {...props} />);
            expect(wrapper.find('GriddingContainer').exists()).toBeTruthy();
        });

        it('should call onGriddingFinish when <GriddingContainer /> finishes', () => {
            const wrapper = shallow(<PlateEditor {...props} />);
            wrapper.find('GriddingContainer').prop('onFinish')();
            expect(props.onGriddingFinish).toHaveBeenCalled();
        });
    });

    describe('colony step', () => {
        beforeEach(() => {
            props.step = 'colony';
            props.selectedColony = { row: 1, col: 2 };
        });

        it('should set the title to "Colony Detection"', () => {
            const wrapper = shallow(<PlateEditor {...props} />);
            expect(wrapper.find('h3').text()).toEqual('Step 3: Colony Detection');
        });

        it('should render a <ColonyEditorContainer />', () => {
            const wrapper = shallow(<PlateEditor {...props} />);
            expect(wrapper.find('ColonyEditorContainer').exists()).toBeTruthy();
        });

        it('should pass the current colony row/col to the <ColonyEditorContainer />', () => {
            const wrapper = shallow(<PlateEditor {...props} />);
            expect(wrapper.find('ColonyEditorContainer').prop('row')).toEqual(1);
            expect(wrapper.find('ColonyEditorContainer').prop('col')).toEqual(2);
        });

        it('should pass the current colony row/col to the <GriddingContainer />', () => {
            const wrapper = shallow(<PlateEditor {...props} />);
            expect(wrapper.find('GriddingContainer').prop('selectedColony'))
                .toEqual(props.selectedColony);
        });

        it('should call onColonyFinish when <ColonyEditorContainer /> finishes', () => {
            const wrapper = shallow(<PlateEditor {...props} />);
            wrapper.find('ColonyEditorContainer').prop('onFinish')();
            expect(props.onColonyFinish).toHaveBeenCalled();
        });

        it('should render a <PlateProgress />', () => {
            const wrapper = shallow(<PlateEditor {...props} />);
            expect(wrapper.find('PlateProgress').exists()).toBeTruthy();
        });

        it('should pass the total number of colony to the <PlateProgress />', () => {
            const wrapper = shallow(<PlateEditor {...props} />);
            expect(wrapper.find('PlateProgress').prop('max')).toEqual(8 * 12);
        });

        it('should pass the position of the current colony to the <PlateProgress />', () => {
            const wrapper = shallow(<PlateEditor {...props} />);
            expect(wrapper.find('PlateProgress').prop('now')).toEqual(10);
        });
    });
});
