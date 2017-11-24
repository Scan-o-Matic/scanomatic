import { shallow } from 'enzyme';
import React from 'react';

import '../components/enzyme-setup';
import RootContainer from '../../ccc/containers/RootContainer';
import * as API from '../../ccc/api';
import cccMetadata from '../fixtures/cccMetadata';
import FakePromise from '../helpers/FakePromise';


describe('<RootContainer />', () => {
    beforeEach(() => {
        spyOn(API, 'InitiateCCC').and.returnValue(new FakePromise());
        spyOn(API, 'GetFixtures').and.returnValue(new FakePromise());
        spyOn(API, 'GetPinningFormats').and.returnValue(new FakePromise());
        spyOn(API, 'finalizeCalibration').and.returnValue(new FakePromise());
    });

    it('should render <Root />', () => {
        const wrapper = shallow(<RootContainer />);
        expect(wrapper.find('Root').exists()).toBeTruthy();
    });

    it('should pass an empty cccMetadata', () => {
        const wrapper = shallow(<RootContainer />);
        expect(wrapper.prop('cccMetadata')).toBeFalsy();
    });

    it('should update the error on onError', () => {
        const wrapper = shallow(<RootContainer />);
        wrapper.prop('onError')('foobar');
        wrapper.update();
        expect(wrapper.prop('error')).toEqual('foobar');
    });

    const {
        species, reference, fixtureName, pinningFormat,
    } = cccMetadata;

    const cccData = {
        identifier: cccMetadata.id,
        access_token: cccMetadata.accessToken,
    };

    it('should call InitiateCCC on onInitializeCCC', () => {
        const wrapper = shallow(<RootContainer />);
        wrapper.prop('onInitializeCCC')(species, reference, fixtureName, pinningFormat);
        expect(API.InitiateCCC).toHaveBeenCalledWith(species, reference);
    });

    it('should set the error prop if initializing the CCC fails', () => {
        API.InitiateCCC.and.returnValue(FakePromise.reject('You broke biology'));
        const wrapper = shallow(<RootContainer />);
        wrapper.prop('onInitializeCCC')(species, reference, fixtureName, pinningFormat);
        wrapper.update();
        expect(wrapper.prop('error'))
            .toContain('Error initializing calibration: You broke biology');
    });

    describe('when Initializing the calibration succeeds', () => {
        beforeEach(() => {
            API.InitiateCCC.and.returnValue(FakePromise.resolve(cccData));
        });

        function initializeCCC(wrapper) {
            wrapper.prop('onInitializeCCC')(species, reference, fixtureName, pinningFormat);
        }

        it('should populate the cccMetadata prop', () => {
            const wrapper = shallow(<RootContainer />);
            initializeCCC(wrapper);
            wrapper.update();
            expect(wrapper.prop('cccMetadata')).toEqual(cccMetadata);
        });

        it('should clear the error prop', () => {
            const wrapper = shallow(<RootContainer />);
            wrapper.setState({ error: 'foobar' });
            initializeCCC(wrapper);
            wrapper.update();
            expect(wrapper.prop('error')).toBeFalsy();
        });

        it('should finalize the CCC on onFinalizeCCC', () => {
            const wrapper = shallow(<RootContainer />);
            initializeCCC(wrapper);
            wrapper.prop('onFinalizeCCC')();
            expect(API.finalizeCalibration)
                .toHaveBeenCalledWith(cccMetadata.id, cccMetadata.accessToken);
        });

        it('should set the error if finalization fails', () => {
            API.finalizeCalibration.and.returnValue(FakePromise.reject('Wobbly'));
            const wrapper = shallow(<RootContainer />);
            initializeCCC(wrapper);
            wrapper.prop('onFinalizeCCC')();
            wrapper.update();
            expect(wrapper.prop('error')).toEqual('Finalization error: Wobbly');
        });

        it('should set finalized to true if finalization succeeds', () => {
            API.finalizeCalibration.and.returnValue(FakePromise.resolve());
            const wrapper = shallow(<RootContainer />);
            initializeCCC(wrapper);
            wrapper.prop('onFinalizeCCC')();
            wrapper.update();
            expect(wrapper.prop('finalized')).toBeTruthy();
        });
    });
});
