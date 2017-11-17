import React from 'react';
import { shallow } from 'enzyme';

import '../components/enzyme-setup';
import PolynomialConstructionContainer from
    '../../ccc/containers/PolynomialConstructionContainer';
import * as API from '../../ccc/api';

describe('<PolynomialConstructionContainer />', () => {
    const props = {
        accessToken: 'T0P53CR3T',
        cccId: 'CCC42',
    };

    const results = {
        polynomial_power: 5,
        polynomial_coefficients: [1, 1, 2, 0.4],
        calculated_sizes: [15, 20],
        measured_sizes: [2, 3],
    };

    it('should render a <PolynomialConstruction />', () => {
        const wrapper = shallow(
            <PolynomialConstructionContainer {...props} />
        );
        expect(wrapper.find('PolynomialConstruction').exists()).toBeTruthy();
    });

    it('should set properties of PolynomialConstruction from state', () => {
        const wrapper = shallow(
            <PolynomialConstructionContainer {...props} />
        );
        const state = {
            power: -2,
            polynomial: 'y = e ^ x',
            resultsData: 'imaginenative',
            error: 'nope',
        };
        wrapper.setState(state);
        const poly = wrapper.find('PolynomialConstruction');
        expect(poly.prop('power')).toEqual(state.power);
        expect(poly.prop('polynomial')).toEqual(state.polynomial);
        expect(poly.prop('data')).toEqual(state.resultsData);
    });

    it('should start without results or error', () => {
        const wrapper = shallow(
            <PolynomialConstructionContainer {...props} />
        );
        expect(wrapper.state('error')).toBe(null);
        expect(wrapper.state('polynomial')).toBe(null);
        expect(wrapper.state('resultsData')).toBe(null);
    });

    it('should dispatch api call', () => {
        const wrapper = shallow(
            <PolynomialConstructionContainer {...props} />
        );
        const poly = wrapper.find('PolynomialConstruction');
        const promise = new Promise(() => {});
        spyOn(API, 'SetNewCalibrationPolynomial')
            .and.returnValue(promise);
        poly.prop('onConstruction')();
        expect(API.SetNewCalibrationPolynomial).toHaveBeenCalledWith(
            props.cccId, 5, props.accessToken
        );
    });

    it('should clear error on new results', (done) => {
        const wrapper = shallow(
            <PolynomialConstructionContainer {...props} />
        );
        const poly = wrapper.find('PolynomialConstruction');
        const promise = Promise.resolve(results);
        spyOn(API, 'SetNewCalibrationPolynomial')
            .and.returnValue(promise);
        wrapper.setState({error: 'test'});
        poly.prop('onConstruction')();
        promise.then(() => {
            expect(wrapper.state('error')).toBe(null);
            done();
        });
    });

    it('should set new results', (done) => {
        const wrapper = shallow(
            <PolynomialConstructionContainer {...props} />
        );
        const poly = wrapper.find('PolynomialConstruction');
        const promise = Promise.resolve(results);
        spyOn(API, 'SetNewCalibrationPolynomial')
            .and.returnValue(promise);
        poly.prop('onConstruction')();
        promise.then(() => {
            expect(wrapper.state('polynomial'))
                .toEqual({
                    power: results.polynomial_power,
                    coefficients: results.polynomial_coefficients,
                });
            expect(wrapper.state('resultsData'))
                .toEqual({
                    calculated: results.calculated_sizes,
                    independentMeasurements: results.measured_sizes,
                });
            done();
        });

    });

    it('should set error', (done) => {
        const wrapper = shallow(
            <PolynomialConstructionContainer {...props} />
        );
        const poly = wrapper.find('PolynomialConstruction');
        const promise = Promise.reject('foo');
        spyOn(API, 'SetNewCalibrationPolynomial')
            .and.returnValue(promise);
        poly.prop('onConstruction')()
            .then(() => {
                expect(wrapper.state('error')).toEqual('foo');
                done();
        });
    });

    it('should clear error', () => {
        const wrapper = shallow(
            <PolynomialConstructionContainer {...props} />
        );
        const poly = wrapper.find('PolynomialConstruction');
        wrapper.setState({error: 'SOS'});
        expect(wrapper.state('error')).toEqual('SOS');
        poly.prop('onClearError')();
        expect(wrapper.state('error')).toBe(null);
    });
});
