import React from 'react';
import { shallow } from 'enzyme';

import './enzyme-setup';
import PolyResults, { PolynomialEquation, ScientificNotation }
    from '../../ccc/components/PolyResults';
import * as P from '../../ccc/components/PolyResults';

describe('<PolyResults />', () => {
    const props = {
        error: null,
        onClearError: jasmine.createSpy('onClearError'),
        polynomial: {
            power: 4,
            coefficients: [1, 0, 0, 1, 0],
        },
        data: {
            calculated: [55, 56],
            independentMeasurements: [33, 102],
        },
    };

    beforeEach(() => {
        props.onClearError.calls.reset();
    });

    describe('while having an error', () => {
        it('renders an alert', () => {
            const err = 'awesomesauce!';
            const wrapper = shallow(<PolyResults {...props} error={err} />);
            expect(wrapper.find('div.alert').exists()).toBeTruthy();
        });

        it('doesnt render any results', () => {
            const err = 'awesomesauce!';
            const wrapper = shallow(<PolyResults {...props} error={err} />);
            expect(wrapper.find('div.results').exists()).not.toBeTruthy();
        });

        it('the alert displays the error', () => {
            const err = 'awesomesauce!';
            const wrapper = shallow(<PolyResults {...props} error={err} />);
            expect(wrapper.find('div.alert').text()).toContain(err);
        });

        it('the alert has a close button', () => {
            const err = 'awesomesauce!';
            const wrapper = shallow(<PolyResults {...props} error={err} />);
            expect(wrapper.find('div.alert').find('button').exists())
                .toBeTruthy();
        });

        it('the alert has a close button invokes onClearError', () => {
            const err = 'awesomesauce!';
            const wrapper = shallow(<PolyResults {...props} error={err} />);
            wrapper.find('div.alert').find('button').simulate('click');
            expect(props.onClearError).toHaveBeenCalled();
        });
    });

    describe('while not having an error', () => {
        it('doesnt renders an alert', () => {
            const wrapper = shallow(<PolyResults {...props} />);
            expect(wrapper.find('div.alert').exists()).not.toBeTruthy();
        });

        it('renders the results', () => {
            const wrapper = shallow(<PolyResults {...props} />);
            expect(wrapper.find('div.results').exists()).toBeTruthy();
        });

        it('sets the title', () => {
            const wrapper = shallow(<PolyResults {...props} />);
            expect(wrapper.find('div.results').find('h3').text())
                .toEqual('Cell Count Calibration Polynomial');

        });

        it('renders a list-group with two items', () => {
            const wrapper = shallow(<PolyResults {...props} />);
            const lg = wrapper.find('div.results').find('ul.list-group')
            expect(lg.exists()).toBeTruthy();
            expect(lg.find('li.list-group-item').length).toBe(2);
        });

        it('renders the polynomial equation', () => {
            const wrapper = shallow(<PolyResults {...props} />);
            const item = wrapper.find('div.results').find('ul.list-group')
                .find('li.list-group-item').at(0);
            expect(item.find('h4.list-group-item-heading').text())
                .toEqual('Polynomial');
            expect(item.find('PolynomialEquation').exists()).toBeTruthy();
        });

        it('calls passes the coefficients to the polynomial equation', () => {
            const wrapper = shallow(<PolyResults {...props} />);
            const polyEq = wrapper.find('div.results').find('ul.list-group')
                .find('li.list-group-item').at(0)
                .find('PolynomialEquation');
            expect(polyEq.prop('coefficients'))
                .toEqual(props.polynomial.coefficients);
        });
    });

    describe('while having no polynomial and no error', () => {

        it('doesnt renders an alert', () => {
            const wrapper = shallow(
                <PolyResults {...props} polynomial={null} />);
            expect(wrapper.find('div.alert').exists()).not.toBeTruthy();
        });

        it('renders the results', () => {
            const wrapper = shallow(
                <PolyResults {...props} polynomial={null} />);
            expect(wrapper.find('div.results').exists()).not.toBeTruthy();
        });
    });
});

describe('<PolynomialEquation />', () => {
    const props = {
        coefficients: [1, 0, 2, 0],
    }

    it('renders an equation', () => {
        const wrapper = shallow(<PolynomialEquation {...props} />);
        expect(wrapper.find('p.math').exists()).toBeTruthy();
        expect(wrapper.find('p.math').find('span.variable').first().text())
            .toEqual('y');
        expect(wrapper.find('p.math').text())
            .toContain(' = ');
    });

    it('renders the scientific precisions', () => {
        const wrapper = shallow(<PolynomialEquation {...props} />);
        const items = wrapper.find('ScientificNotation');
        expect(items.exists()).toBeTruthy();
        expect(items.length).toBe(2);
        expect(items.at(0).prop('value')).toEqual(1);
        expect(items.at(0).prop('precision')).toEqual(4);
        expect(items.at(1).prop('value')).toEqual(2);
        expect(items.at(1).prop('precision')).toEqual(4);
    });

    it('renders the correct x-powers', ()=>{
        //Neither x^0 or x^1 should have sups
        const wrapper = shallow(
            <PolynomialEquation {...props} coefficients={[1, 0, 1, 1, 1]} />);
        const items = wrapper.find('sup');
        expect(items.exists()).toBeTruthy();
        expect(items.length).toBe(2);
        expect(items.at(0).text()).toEqual('4');
        expect(items.at(1).text()).toEqual('2');
    });

    it('renders x:es', ()=>{
        //x^0 has no x but y is at 0
        const wrapper = shallow(
            <PolynomialEquation {...props} coefficients={[1, 0, 1, 1, 1]} />);
        const items = wrapper.find('span.variable');
        expect(items.exists()).toBeTruthy();
        expect(items.length).toBe(4);
        expect(items.at(0).text()).toEqual('y');
        expect(items.at(1).text()).toEqual('x');
        expect(items.at(2).text()).toEqual('x');
        expect(items.at(3).text()).toEqual('x');
    });

    it('renders +:es between terms', ()=>{
        //x^0 has no x but y is at 0
        const wrapper = shallow(<PolynomialEquation {...props} />);
        expect(wrapper.find('p.math').text())
            .toContain('+');
    });

    it('renders y=0 if no non-zero coeffs', () => {
        const wrapper = shallow(
            <PolynomialEquation {...props} coefficients={[0, 0, 0]} />);
        expect(wrapper.find('p.math').find('span.variable').first().text())
            .toEqual('y');
        expect(wrapper.find('p.math').text())
            .toContain(' = 0');
    });

    it('renders y=`intercept` if only intercept', () => {
        const wrapper = shallow(
            <PolynomialEquation {...props} coefficients={[0, 0, 5]} />);
        expect(wrapper.find('p.math').find('span.variable').first().text())
            .toEqual('y');
        expect(wrapper.find('ScientificNotation').at(0).prop('value'))
            .toEqual(5);
        expect(wrapper.find('p.math').text())
            .not.toContain('x');
    });

});

describe('<ScientificNotation />', () => {
    it('renders a span', () => {
        const wrapper = shallow(
            <ScientificNotation value={0} precision={1} />);
        expect(wrapper.find('span').exists()).toBeTruthy();
    });

    it('renders 0', () => {
        const wrapper = shallow(
            <ScientificNotation value={0} precision={1} />);
        expect(wrapper.find('span').first().text()).toEqual('0');
    });

    it('renders moderate numbers with precision', () => {
        const wrapper = shallow(
            <ScientificNotation value={10.32} precision={3} />);
        expect(wrapper.find('span').first().text()).toEqual('10.3');
    });

    it('renders larger numbers with precision', () => {
        const wrapper = shallow(
            <ScientificNotation value={9999} precision={2} />);
        expect(wrapper.html())
            .toEqual('<span>10×<span>10<sup>3</sup></span></span>');
    });

    it('renders largest numbers with precision', () => {
        const wrapper = shallow(
            <ScientificNotation value={-123456} precision={3} />);
        expect(wrapper.html())
            .toEqual('<span>-0.123×<span>10<sup>6</sup></span></span>');
    });

    it('renders small numbers with precision', () => {
        const wrapper = shallow(
            <ScientificNotation value={-0.0001} precision={3} />);
        expect(wrapper.html())
            .toEqual('<span>-0.100×<span>10<sup>-3</sup></span></span>');
    });
    
    it('only does powers of 3', () => {
        const wrapper = shallow(
            <ScientificNotation value={12345} precision={3} />);
        expect(wrapper.find('sup').text()).toEqual('3');
    });
});