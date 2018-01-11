import React from 'react';
import { shallow } from 'enzyme';

import './enzyme-setup';
import PolynomialResultsInfo, {
    PolynomialEquation, ScientificNotation, numberAsScientific
} from '../../src/components/PolynomialResultsInfo';

describe('<PolynomialResultsInfo />', () => {
    const props = {
        polynomial: {
            power: 4,
            coefficients: [1, 0, 0, 1, 0],
            colonies: 96,
        },
    };

    it('doesnt renders an alert', () => {
        const wrapper = shallow(<PolynomialResultsInfo {...props} />);
        expect(wrapper.find('div.alert').exists()).not.toBeTruthy();
    });

    it('renders the results', () => {
        const wrapper = shallow(<PolynomialResultsInfo {...props} />);
        expect(wrapper.find('div.results').exists()).toBeTruthy();
    });

    it('sets the title', () => {
        const wrapper = shallow(<PolynomialResultsInfo {...props} />);
        expect(wrapper.find('div.results').find('h3').text())
            .toEqual('Cell Count Calibration Polynomial');

    });

    it('renders a list-group with two items', () => {
        const wrapper = shallow(<PolynomialResultsInfo {...props} />);
        const lg = wrapper.find('div.results').find('ul.list-group')
        expect(lg.exists()).toBeTruthy();
        expect(lg.find('li.list-group-item').length).toBe(2);
    });

    it('renders the polynomial equation', () => {
        const wrapper = shallow(<PolynomialResultsInfo {...props} />);
        const item = wrapper.find('div.results').find('ul.list-group')
            .find('li.list-group-item').at(0);
        expect(item.find('h4.list-group-item-heading').text())
            .toEqual('Polynomial');
        expect(item.find('PolynomialEquation').exists()).toBeTruthy();
    });

    it('calls passes the coefficients to the polynomial equation', () => {
        const wrapper = shallow(<PolynomialResultsInfo {...props} />);
        const polyEq = wrapper.find('div.results').find('ul.list-group')
            .find('li.list-group-item').at(0)
            .find('PolynomialEquation');
        expect(polyEq.prop('coefficients'))
            .toEqual(props.polynomial.coefficients);
    });

    it('renders how many colonies were used to create it', () => {
        const wrapper = shallow(<PolynomialResultsInfo {...props} />);
        const item = wrapper.find('div.results').find('ul.list-group')
            .find('li.list-group-item').at(1);
        expect(item.find('h4.list-group-item-heading').text())
            .toEqual('Colonies included');
        expect(item.text())
            .toContain(`${props.polynomial.colonies} colonies`);
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

describe('numberAsScientific', () => {
    it('returns a coefficient and exponent', () => {
        const sci = numberAsScientific(33)
        expect(sci.coefficient).not.toBe(undefined);
        expect(sci.exponent).not.toBe(undefined);
    });

    it('calculates 42 as expected', () => {
        expect(numberAsScientific(42)).toEqual({
            exponent: 1,
            coefficient: 4.2
        });
    });

    it('calculates 0.0034 as expected', () => {
        expect(numberAsScientific(0.0034)).toEqual({
            exponent: -3,
            coefficient: 3.4
        });
    });

    it('preserves the sign', () => {
        expect(numberAsScientific(-42).coefficient).toEqual(-4.2);
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

    it('renders moderatly small numbers with precision', () => {
        const wrapper = shallow(
            <ScientificNotation value={0.32} precision={3} />);
        expect(wrapper.find('span').first().text()).toEqual('0.320');
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
            .toEqual('<span>-1.23×<span>10<sup>5</sup></span></span>');
    });

    it('renders small numbers with precision', () => {
        const wrapper = shallow(
            <ScientificNotation value={-0.0001} precision={3} />);
        expect(wrapper.html())
            .toEqual('<span>-1.00×<span>10<sup>-4</sup></span></span>');
    });

});
