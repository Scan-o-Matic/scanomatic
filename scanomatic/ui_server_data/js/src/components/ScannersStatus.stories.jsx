import React from 'react';
import { storiesOf } from '@storybook/react';
import ScannersStatus from './ScannersStatus';
import '../../../style/bootstrap.css';

const milliPerDay = 1000 * 3600 * 24;
const milliPerHour = 1000 * 3600;

storiesOf('ScannersStatus', module)
    .addDecorator(story => (
        <div className="container">
            <div className="row">
                <div className="col-md-offset-1 col-md-10">
                    {story()}
                </div>
            </div>
        </div>
    ))
    .add('No scanners', () => (
        <ScannersStatus
            scanners={[]}
            jobs={[]}
        />
    ))
    .add('Scanners no jobs', () => (
        <ScannersStatus
            scanners={[
                {
                    identifier: 'scanner001',
                    name: 'Resourceful Red robin',
                    owned: false,
                    power: true,
                },
                {
                    identifier: 'scanner002',
                    name: 'Lazy Lark',
                    owned: false,
                    power: false,
                },
                {
                    identifier: 'scanner003',
                    name: 'Boiserous Bowerbird',
                    owned: true,
                    power: true,
                },
            ]}
            jobs={[]}
        />
    ))
    .add('Scanners and jobs', () => (
        <ScannersStatus
            scanners={[
                {
                    identifier: 'scanner001',
                    name: 'Resourceful Red robin',
                    owned: false,
                    power: true,
                },
                {
                    identifier: 'scanner002',
                    name: 'Lazy Lark',
                    owned: false,
                    power: false,
                },
                {
                    identifier: 'scanner003',
                    name: 'Boiserous Bowerbird',
                    owned: true,
                    power: true,
                },
            ]}
            jobs={[
                {
                    id: 'job001',
                    name: 'testing stuff',
                    scannerId: 'scanner001',
                    started: new Date().getTime() - (2 * milliPerDay),
                    end: new Date().getTime() + milliPerDay,
                },
                {
                    id: 'job002',
                    name: 'over and done',
                    scannerId: 'scanner003',
                    started: new Date().getTime() - (2 * milliPerDay),
                    stopped: new Date().getTime() - (milliPerHour),
                    end: new Date().getTime() + milliPerDay,
                },
                {
                    id: 'job003',
                    name: 'old one',
                    scannerId: 'scanner003',
                    started: new Date().getTime() - (5 * milliPerDay),
                    end: new Date().getTime() - (2 * milliPerDay) - (2 * milliPerHour),
                },
            ]}
        />
    ));
