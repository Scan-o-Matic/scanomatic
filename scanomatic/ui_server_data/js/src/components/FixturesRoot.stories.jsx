import React from 'react';
import { storiesOf } from '@storybook/react';
import { action } from '@storybook/addon-actions';
import FixturesRoot from './FixturesRoot';
import '../../../style/bootstrap.css';

const actions = {
    onSelectScanner: action('select-scanner'),
    onScanOneImage: action('scan-one'),
};

storiesOf('FixturesRoot', module)
    .addDecorator(story => (
        <div className="container">
            <div className="row">
                <div className="col-md-offset-1 col-md-10">
                    {story()}
                </div>
            </div>
        </div>
    ))
    .add('No scanners shows alert', () => (
        <FixturesRoot {...actions} />
    ))
    .add('Select scanner', () => (
        <FixturesRoot
            scanners={[
                {
                    identifier: 'scanner001',
                    name: 'Dirty Dove',
                    owned: false,
                    power: true,
                },
            ]}
            {...actions}
        />
    ))
    .add('Scanner selected', () => (
        <FixturesRoot
            scanners={[
                {
                    identifier: 'scanner001',
                    name: 'Dirty Dove',
                    owned: false,
                    power: true,
                },
                {
                    identifier: 'scanner002',
                    name: 'Paranoid Penguin',
                    owned: false,
                    power: true,
                },
            ]}
            scannerId="scanner002"
            {...actions}
        />
    ))
    .add('Non-existing scanner selected', () => (
        <FixturesRoot
            scanners={[
                {
                    identifier: 'scanner001',
                    name: 'Dirty Dove',
                    owned: false,
                    power: true,
                },
                {
                    identifier: 'scanner002',
                    name: 'Paranoid Penguin',
                    owned: false,
                    power: true,
                },
            ]}
            scannerId="scanner003"
            {...actions}
        />
    ))
    .add('Offline scanner selected', () => (
        <FixturesRoot
            scanners={[
                {
                    identifier: 'scanner001',
                    name: 'Dirty Dove',
                    owned: false,
                    power: true,
                },
                {
                    identifier: 'scanner002',
                    name: 'Paranoid Penguin',
                    owned: false,
                    power: false,
                },
            ]}
            scannerId="scanner002"
            {...actions}
        />
    ))
    .add('Occupied scanner selected', () => (
        <FixturesRoot
            scanners={[
                {
                    identifier: 'scanner001',
                    name: 'Dirty Dove',
                    owned: false,
                    power: true,
                },
                {
                    identifier: 'scanner002',
                    name: 'Paranoid Penguin',
                    owned: true,
                    power: true,
                },
            ]}
            scannerId="scanner002"
            {...actions}
        />
    ));
