import { shallow } from 'enzyme';
import React from 'react';

import './enzyme-setup';
import ColonyEditor from '../../src/components/ColonyEditor';


describe('<ColonyEditor />', () => {
    const data = {
        image: [[0, 1], [1, 0]],
        blob: [[true, false], [false, true]],
        background: [[false, true], [true, false]],
    };
    const onSet = jasmine.createSpy('onSet');
    const onSkip = jasmine.createSpy('onSkip');
    const onUpdate = jasmine.createSpy('onUpdate');
    const onCellCountChange = jasmine.createSpy('onCellCountChange');
    let wrapper;


    beforeEach(() => {
        onSet.calls.reset();
        onSkip.calls.reset();
        onUpdate.calls.reset();
        onCellCountChange.calls.reset();
        wrapper = shallow(
            <ColonyEditor
                data={data}
                cellCount={1234}
                cellCountError={false}
                onSet={onSet}
                onSkip={onSkip}
                onUpdate={onUpdate}
                onCellCountChange={onCellCountChange}
            />
        );
    });

    it('should render a <ColonyFeatures/>', () => {
        expect(wrapper.find('ColonyFeatures').exists()).toBe(true);
    });

    it('should render a "Fix" <button/>', () => {
        const button = wrapper.find('button.btn-fix');
        expect(button.exists()).toBe(true);
        expect(button.text()).toEqual("Fix");
    });

    it('should render a <ColonyImage/>', () => {
        expect(wrapper.find('ColonyImage').exists()).toBe(true);
    });

    it('should render a "Set" <button />', () => {
        const button = wrapper.find('button.btn-set');
        expect(button.exists()).toBe(true);
        expect(button.text()).toEqual('Set');

    });

    it('should render a "Skip" <button />', () => {
        const button = wrapper.find('button.btn-skip')
        expect(button.exists()).toBeTruthy();
        expect(button.text()).toEqual('Skip');
    });

    it('should render an <input /> for the cell count', () => {
        const input = wrapper.find('input[name="cell-count"]');
        expect(input.exists()).toBeTruthy();
        expect(input.prop('value')).toEqual(1234);
    });

    it('should not mark the <input /> as error if cellCountError is false', () => {
        wrapper.setProps({ cellCountError: false });
        const input = wrapper.find('input[name="cell-count"]');
        expect(input.parent().prop('className')).not.toContain('has-error');
    });

    it('should mark the <input /> as error if cellCountError is true', () => {
        wrapper.setProps({ cellCountError: true });
        const input = wrapper.find('input[name="cell-count"]');
        expect(input.parent().prop('className')).toContain('has-error');
    });

    it('should call the onCellCountChange callback when the input is changed', () => {
        const input = wrapper.find('input[name="cell-count"]');
        input.simulate('change', { target: { value: '666' } });
        expect(onCellCountChange).toHaveBeenCalledWith(666);
    });

    it('should set `draw` to true when the "fix" button is clicked', () => {
        const button = wrapper.find('button.btn-fix');
        button.simulate('click');
        const imageCanvas = wrapper.find('ColonyImage');
        expect(imageCanvas.prop('draw')).toBe(true);
    });

    it('should call the onSet callback when the "Set" button is clicked', () => {
        const button = wrapper.find('button.btn-set');
        button.simulate('click');
        expect(onSet).toHaveBeenCalledWith();
    });

    it('should call the onSkip callback when the "Skip" button is clicked', () => {
        const button = wrapper.find('button.btn-skip');
        button.simulate('click');
        expect(onSkip).toHaveBeenCalled();
    });

    describe('when the blob is updated', () => {
        const updatedData = { blob: [[false, false], [true, true]] };

        beforeEach(() => {
            wrapper.find('button.btn-fix').simulate('click');
            wrapper.find('ColonyImage').simulate('update', updatedData);
        });

        it('should set `draw` to false', () => {
            expect(wrapper.find('ColonyImage').prop('draw')).toBe(false);
        });

        it('should call the onUpdate callback', () => {
            expect(onUpdate).toHaveBeenCalledWith(updatedData);
        });

    });
});
