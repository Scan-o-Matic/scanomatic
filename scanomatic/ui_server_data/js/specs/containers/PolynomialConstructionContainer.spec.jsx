import React from 'react';
import { shallow } from 'enzyme';

import '../components/enzyme-setup';
import PolynomialConstructionContainer from
    '../../src/containers/PolynomialConstructionContainer';
import * as API from '../../src/api';
import cccMetadata from '../fixtures/cccMetadata';

describe('<PolynomialConstructionContainer />', () => {
    const onFinalizeCCC = jasmine.createSpy('onFinalizeCCC');
    const props = { cccMetadata, onFinalizeCCC };

    const results = {
        polynomial_coefficients: [1, 1, 2, 0.4],
        calculated_sizes: [15, 20],
        measured_sizes: [2, 3],
        correlation: {
            slope: 3,
            intercept: 44,
            stderr: -3,
        },
        colonies: {
            source_values: [[1, 2], [5.5]],
            source_value_counts: [[100, 1], [44]],
            target_values: [123, 441],
            max_source_counts: 100,
            min_source_values: 1,
            max_source_values: 5.5,
        },
    };

    beforeEach(() => {
        spyOn(API, 'SetNewCalibrationPolynomial')
            .and.returnValue(new Promise(() => {}));
    });

    it('should render a <PolynomialConstruction />', () => {
        const wrapper = shallow(<PolynomialConstructionContainer {...props} />);
        expect(wrapper.find('PolynomialConstruction').exists()).toBeTruthy();
    });

    it('should set 5 as the default polynomial degree', () => {
        const wrapper = shallow(<PolynomialConstructionContainer {...props} />);
        expect(wrapper.prop('degreeOfPolynomial')).toEqual(5);
    });

    it('should update the polynomial degree on onDegreeOfPolynomialChange', () => {
        const wrapper = shallow(<PolynomialConstructionContainer {...props} />);
        wrapper.prop('onDegreeOfPolynomialChange')({ target: { value: '42' } });
        wrapper.update();
        expect(wrapper.prop('degreeOfPolynomial')).toEqual(42);
    });

    it('should pass onFinalizeCCC to <PolynomialContruction />', () => {
        const wrapper = shallow(<PolynomialConstructionContainer {...props} />);
        expect(wrapper.find('PolynomialConstruction').prop('onFinalizeCCC'))
            .toBe(onFinalizeCCC);
    });

    it('should set properties of PolynomialConstruction from state', () => {
        const wrapper = shallow(<PolynomialConstructionContainer {...props} />);
        const state = {
            polynomial: {
                coefficients: [1, 2, 3, 4, 5],
                colonies: 42,
            },
            resultsData: {
                calculated: [1, 2, 3],
                independentMeasurements: [3, 4, 5],
            },
            correlation: {
                slope: 3,
                intercept: 44,
                stderr: -3,
            },
            colonies: {
                pixelValues: [[1, 2], [5.5]],
                pixelCounts: [[100, 1], [44]],
                independentMeasurements: [123, 441],
                minPixelValue: 1,
                maxPixelValue: 5.5,
                maxCount: 100,
            },
            error: 'nope',
        };
        wrapper.setState(state);
        const poly = wrapper.find('PolynomialConstruction');
        expect(poly.prop('polynomial')).toEqual(state.polynomial);
        expect(poly.prop('resultsData')).toEqual(state.resultsData);
        expect(poly.prop('correlation')).toEqual(state.correlation);
        expect(poly.prop('colonies')).toEqual(state.colonies);
        expect(poly.prop('error')).toEqual(state.error);
    });

    it('should start without results or error', () => {
        const wrapper = shallow(<PolynomialConstructionContainer {...props} />);
        expect(wrapper.state('error')).toBe(null);
        expect(wrapper.state('polynomial')).toBe(null);
        expect(wrapper.state('resultsData')).toBe(null);
        expect(wrapper.state('correlation')).toBe(null);
    });

    it('should dispatch api call', () => {
        const wrapper = shallow(<PolynomialConstructionContainer {...props} />);
        const poly = wrapper.find('PolynomialConstruction');
        poly.prop('onConstruction')();
        expect(API.SetNewCalibrationPolynomial).toHaveBeenCalledWith(
            props.cccMetadata.id,
            5,
            props.cccMetadata.accessToken,
        );
    });

    it('should build the polynomial with the updated degree', () => {
        const wrapper = shallow(<PolynomialConstructionContainer {...props} />);
        wrapper.prop('onDegreeOfPolynomialChange')({ target: { value: '42' } });
        wrapper.prop('onConstruction')();
        expect(API.SetNewCalibrationPolynomial)
            .toHaveBeenCalledWith(props.cccMetadata.id, 42, props.cccMetadata.accessToken);
    });

    it('should clear error on new results', (done) => {
        const wrapper = shallow(<PolynomialConstructionContainer {...props} />);
        const poly = wrapper.find('PolynomialConstruction');
        const promise = Promise.resolve(results);
        API.SetNewCalibrationPolynomial.and.returnValue(promise);
        wrapper.setState({ error: 'test' });
        poly.prop('onConstruction')()
            .then(() => {
                expect(wrapper.state('error')).toBe(null);
                done();
            });
    });

    it('should set new results', (done) => {
        const wrapper = shallow(<PolynomialConstructionContainer {...props} />);
        const poly = wrapper.find('PolynomialConstruction');
        const promise = Promise.resolve(results);
        API.SetNewCalibrationPolynomial.and.returnValue(promise);
        poly.prop('onConstruction')();
        promise.then(() => {
            expect(wrapper.state('polynomial'))
                .toEqual({
                    coefficients: results.polynomial_coefficients,
                    colonies: results.calculated_sizes.length,
                });
            expect(wrapper.state('resultsData'))
                .toEqual({
                    calculated: results.calculated_sizes,
                    independentMeasurements: results.measured_sizes,
                });
            expect(wrapper.state('correlation'))
                .toEqual(results.correlation);
            done();
        });
    });

    it('should set error', (done) => {
        const wrapper = shallow(<PolynomialConstructionContainer {...props} />);
        const poly = wrapper.find('PolynomialConstruction');
        const promise = Promise.reject('foo');
        API.SetNewCalibrationPolynomial.and.returnValue(promise);
        poly.prop('onConstruction')()
            .then(() => {
                expect(wrapper.state('error')).toEqual('foo');
                done();
            });
    });

    it('should clear error', () => {
        const wrapper = shallow(<PolynomialConstructionContainer {...props} />);
        const poly = wrapper.find('PolynomialConstruction');
        wrapper.setState({ error: 'SOS' });
        expect(wrapper.state('error')).toEqual('SOS');
        poly.prop('onClearError')();
        expect(wrapper.state('error')).toBe(null);
    });
});
