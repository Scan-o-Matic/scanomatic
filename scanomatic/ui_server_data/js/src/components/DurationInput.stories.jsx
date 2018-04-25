import React from 'react';
import { storiesOf } from '@storybook/react';
import { action } from '@storybook/addon-actions';
import DurationInput from './DurationInput';
import '../../../style/bootstrap.css';


storiesOf('DurationInput', module)
    .addDecorator(story => (
        <div className="row">
            <div className="col-md-offset-1 col-md-10">
                {story()}
            </div>
        </div>
    ))
    .add('Null duration', () => (
        <DurationInput
            onChange={action('change')}
        />
    ))
    .add('With errors', () => (
        <DurationInput
            duration={12345000}
            error="Really not a good duration"
            onChange={action('change')}
        />
    ))
    .add('With negative time', () => (
        <DurationInput
            duration={-1234}
            onChange={action('change')}
        />
    ));
