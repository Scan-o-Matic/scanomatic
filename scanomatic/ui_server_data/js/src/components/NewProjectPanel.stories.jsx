import React from 'react';
import { storiesOf } from '@storybook/react';
import { action } from '@storybook/addon-actions';
import NewProjectPanel from './NewProjectPanel';
import '../../../style/bootstrap.css';
import '../../../style/experiment.css';


storiesOf('NewProjectPanel', module)
    .addDecorator(story => (
        <div className="row">
            <div className="col-md-offset-1 col-md-10">
                {story()}
            </div>
        </div>
    ))
    .add('Empty values', () => (
        <NewProjectPanel
            name=""
            description=""
            error={null}
            onChange={action('change')}
            onSubmit={action('submit')}
            onCancel={action('cancel')}
        />
    ))
    .add('with error', () => (
        <NewProjectPanel
            name="Some name"
            description="This is the description. It describes."
            error="Something is wrong..."
            onChange={action('change')}
            onSubmit={action('submit')}
            onCancel={action('cancel')}
        />
    ));
