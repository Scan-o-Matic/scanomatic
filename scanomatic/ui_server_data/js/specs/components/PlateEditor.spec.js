import { shallow } from 'enzyme';
import React  from 'react';

import './enzyme-setup';
import PlateEditor, { PlateStatusLabel } from '../../src/components/PlateEditor';
import cccMetadata from '../fixtures/cccMetadata';

describe('<PlateEditor />', () => {
    let props;

    beforeEach(() => {
       props = {
           cccMetadata,
           imageId: '1M4G3',
           imageName: 'myimage.tiff',
           plateId: 1,
           onClickNext: jasmine.createSpy('onClickNext'),
           onColonyFinish: jasmine.createSpy('onColonyFinish'),
           rowOffset: 1,
           colOffset: 2,
           onRowOffsetChange: jasmine.createSpy('onRowOffsetChange'),
           onColOffsetChange: jasmine.createSpy('onColOffsetChange'),
           onRegrid: jasmine.createSpy('onRegrid'),
           griddingError: 'XxX',
           griddingLoading: true,
           selectedColony: { row: 1, col: 2 },
           step: 'pre-processing',
        };
    });

    it('should render a bootstrap panel', () => {
        const wrapper = shallow(<PlateEditor {...props} />);
        expect(wrapper.find('div.panel').exists()).toBeTruthy();
    });

    it('should show the image title and plate number in the panel heading', () => {
        const wrapper = shallow(<PlateEditor {...props} />);
        expect(wrapper.find('div.panel-heading').text())
            .toContain('myimage.tiff, Plate 1');
    });

    describe('pre-processing', () => {
        xit('should render an <AnimatedProgressBar />', () => {
        });
    });

    describe('gridding', () => {
        it('should render a <PlateContainer />', () => {
            const wrapper = shallow(<PlateEditor {...props} step="gridding" />);
            expect(wrapper.find('PlateContainer').exists()).toBeTruthy();
        });

        it('should renedr a <Gridding />', () => {
            const wrapper = shallow(<PlateEditor {...props} step="gridding" />);
            expect(wrapper.find('Gridding').exists()).toBeTruthy();
        });

        it('should render a <button /> "Next"', () => {
            const wrapper = shallow(<PlateEditor {...props} step="gridding" />);
            expect(wrapper.find('button.btn-next').text()).toEqual('Next');
        });

        it('should disable the <button /> if griddingLoading is true', () => {
            const wrapper = shallow(
                <PlateEditor
                    {...props}
                    step="gridding"
                    griddingLoading
                    griddingError={null}
                />
            );
            expect(wrapper.find('.btn-next').prop('disabled')).toBeTruthy();
        });

        it('should disable the <button /> if griddingError is not empty', () => {
            const wrapper = shallow(
                <PlateEditor
                    {...props}
                    step="gridding"
                    griddingLoading={false}
                    griddingError="Error"
                />
            );
            expect(wrapper.find('.btn-next').prop('disabled')).toBeTruthy();
        });

        it('should enable the <button /> if griddingLoading is false and griddingError is null', () => {
            const wrapper = shallow(
                <PlateEditor
                    {...props}
                    step="gridding"
                    griddingLoading={false}
                    griddingError={null}
                />
            );
            expect(wrapper.find('.btn-next').prop('disabled')).toBeFalsy();
        });

        it('should pass onRowOffsetChange to <Gridding />', () => {
            const wrapper = shallow(<PlateEditor {...props} step="gridding" />);
            expect(wrapper.find('Gridding').prop('onRowOffsetChange'))
                .toBe(props.onRowOffsetChange);
        });

        it('should pass onColOffsetChange to <Gridding />', () => {
            const wrapper = shallow(<PlateEditor {...props} step="gridding" />);
            expect(wrapper.find('Gridding').prop('onColOffsetChange'))
                .toBe(props.onColOffsetChange);
        });

        it('should pass onRegrid to <Gridding />', () => {
            const wrapper = shallow(<PlateEditor {...props} step="gridding" />);
            expect(wrapper.find('Gridding').prop('onRegrid'))
                .toBe(props.onRegrid);
        });

        it('should pass griddingError to <Gridding />', () => {
            const wrapper = shallow(<PlateEditor {...props} step="gridding" />);
            expect(wrapper.find('Gridding').prop('error')).toEqual("XxX");
        });

        it('should pass griddingLoading to <Gridding />', () => {
            const wrapper = shallow(<PlateEditor {...props} step="gridding" />);
            expect(wrapper.find('Gridding').prop('loading')).toEqual(true);
        });

        it('should set the title to "Gridding"', () => {
            const wrapper = shallow(<PlateEditor {...props} step="gridding" />);
            expect(wrapper.find('h3').text()).toEqual('Step 2: Gridding');
        });
    });

    describe('colony-detection', () => {
        it('should set the title to "Colony Detection"', () => {
            const wrapper = shallow(<PlateEditor {...props} step="colony-detection" />);
            expect(wrapper.find('h3').text()).toEqual('Step 3: Colony Detection');
        });

        it('should render a <ColonyEditorContainer />', () => {
            const wrapper = shallow(<PlateEditor {...props} step="colony-detection" />);
            expect(wrapper.find('ColonyEditorContainer').exists()).toBeTruthy();
        });

        it('should pass the current colony row/col to the <ColonyEditorContainer />', () => {
            const wrapper = shallow(<PlateEditor {...props} step="colony-detection" />);
            expect(wrapper.find('ColonyEditorContainer').prop('row')).toEqual(1);
            expect(wrapper.find('ColonyEditorContainer').prop('col')).toEqual(2);
        });

        it('should pass the current colony row/col to the <PlateContainer />', () => {
            const wrapper = shallow(<PlateEditor {...props} step="colony-detection" />);
            expect(wrapper.find('PlateContainer').prop('selectedColony'))
                .toEqual(props.selectedColony);
        });

        it('should call onColonyFinish when <ColonyEditorContainer /> finishes', () => {
            const wrapper = shallow(<PlateEditor {...props} step="colony-detection" />);
            wrapper.find('ColonyEditorContainer').prop('onFinish')();
            expect(props.onColonyFinish).toHaveBeenCalled();
        });

        it('should render a <PlateProgress />', () => {
            const wrapper = shallow(<PlateEditor {...props} step="colony-detection" />);
            expect(wrapper.find('PlateProgress').exists()).toBeTruthy();
        });

        it('should pass the total number of colony to the <PlateProgress />', () => {
            const wrapper = shallow(<PlateEditor {...props} step="colony-detection" />);
            expect(wrapper.find('PlateProgress').prop('max')).toEqual(6);
        });

        it('should pass the position of the current colony to the <PlateProgress />', () => {
            const wrapper = shallow(<PlateEditor {...props} step="colony-detection" />);
            expect(wrapper.find('PlateProgress').prop('now')).toEqual(4);
        });
    });
});

describe('<PlateStatusLabel />', () => {
    describe('step=pre-processing', () => {
        it('should have class label-default', () => {
            const wrapper = shallow(<PlateStatusLabel step="pre-processing" />);
            expect(wrapper.prop('className')).toContain('label-default');
        });

        it('should have text Pre-processing...', () => {
            const wrapper = shallow(<PlateStatusLabel step="pre-processing" />);
            expect(wrapper.text()).toContain('Pre-processing...');
        });
    });

    describe('step=gridding, griddingError=null', () => {
        it('should have class label-info', () => {
            const wrapper = shallow(<PlateStatusLabel step="gridding" />);
            expect(wrapper.prop('className')).toContain('label-default');
        });

        it('should have text Gridding...', () => {
            const wrapper = shallow(<PlateStatusLabel step="gridding" />);
            expect(wrapper.text()).toContain('Gridding...');
        });
    });

    describe('step=gridding, griddingError!=null', () => {
        it('should have class label-danger', () => {
            const wrapper = shallow(<PlateStatusLabel step="gridding" griddingError="xxx" />);
            expect(wrapper.prop('className')).toContain('label-danger');
        });

        it('should have text Gridding error', () => {
            const wrapper = shallow(<PlateStatusLabel step="gridding" griddingError="xxx" />);
            expect(wrapper.text()).toContain('Gridding error');
        });
    });

    describe('step=colony-detection, now=42, total=96', () => {
        it('should have class label-primary', () => {
            const wrapper = shallow(<PlateStatusLabel step="colony-detection" now={42} max={96} />);
            expect(wrapper.prop('className')).toContain('label-primary');
        });

        it('should have text 42/96', () => {
            const wrapper = shallow(<PlateStatusLabel step="colony-detection" now={42} max={96} />);
            expect(wrapper.text()).toContain('42/96');
        });
    });

    describe('step=done', () => {
        it('should have class label-success', () => {
            const wrapper = shallow(<PlateStatusLabel step="done" />);
            expect(wrapper.prop('className')).toContain('label-success');
        });

        it('should have text Done!', () => {
            const wrapper = shallow(<PlateStatusLabel step="done" />);
            expect(wrapper.text()).toContain('Done!');
        });
    });
});
