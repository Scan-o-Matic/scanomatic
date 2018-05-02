import { shallow } from 'enzyme';
import React from 'react';

import './enzyme-setup';
import ScanningJobFeatureExtractDialogue from './ScanningJobFeatureExtractDialogue';

describe('<ScanningJobFeatureExtractDialogue />', () => {
    let wrapper;
    const onExtractFeatures = jasmine.createSpy('onExtractFeatures');
    const onCancel = jasmine.createSpy('onCancel');

    const props = {
        onExtractFeatures,
        onCancel,
    };

    beforeEach(() => {
        onExtractFeatures.calls.reset();
        onCancel.calls.reset();
        wrapper = shallow(<ScanningJobFeatureExtractDialogue {...props} />);
    });

    describe('keep qc checkbox', () => {
        it('renders', () => {
            const checkbox = wrapper.find('input.keep-qc');
            expect(checkbox.exists()).toBeTruthy();
            expect(checkbox.prop('type')).toEqual('checkbox');
        });

        it('defaults to unchecked', () => {
            const checkbox = wrapper.find('input.keep-qc');
            expect(checkbox.prop('checked')).toEqual(false);
        });

        it('toggles when clicked', () => {
            let checkbox = wrapper.find('input.keep-qc');
            checkbox.simulate('change', { target: { checked: true } });
            wrapper.update();
            checkbox = wrapper.find('input.keep-qc');
            expect(checkbox.prop('checked')).toEqual(true);
        });
    });

    describe('Extract Features button', () => {
        it('renders as primary action', () => {
            const btn = wrapper.find('.feature-extract-button');
            expect(btn.exists()).toBeTruthy();
            expect(btn.hasClass('btn')).toBeTruthy();
            expect(btn.hasClass('btn-primary')).toBeTruthy();
        });

        it('calls onExtractFeatures', () => {
            const btn = wrapper.find('.feature-extract-button');
            btn.simulate('click');
            expect(onExtractFeatures).toHaveBeenCalledWith(false);
        });

        it('calls onExtractFeatures with updated keep qc value', () => {
            const checkbox = wrapper.find('input.keep-qc');
            checkbox.simulate('change', { target: { checked: true } });
            const btn = wrapper.find('.feature-extract-button');
            btn.simulate('click');
            expect(onExtractFeatures).toHaveBeenCalledWith(true);
        });
    });

    describe('Cancel button', () => {
        it('renders', () => {
            const btn = wrapper.find('.feature-extract-button');
            expect(btn.exists()).toBeTruthy();
            expect(btn.hasClass('btn')).toBeTruthy();
        });

        it('calls onCancel', () => {
            const btn = wrapper.find('.cancel-button');
            btn.simulate('click');
            expect(onCancel).toHaveBeenCalled();
        });
    });
});