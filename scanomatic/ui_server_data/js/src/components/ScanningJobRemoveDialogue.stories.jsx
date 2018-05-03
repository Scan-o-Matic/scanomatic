import React from 'react';
import { storiesOf } from '@storybook/react';
import { action } from '@storybook/addon-actions';
import ScanningJobRemoveDialogue from './ScanningJobRemoveDialogue';
import '../../../style/bootstrap.css';
import '../../../style/experiment.css';


storiesOf('ScanningJobRemoveDialogue', module)
    .addDecorator(story => (
        <div className="row">
            <div className="col-md-offset-1 col-md-7">
                <div className="panel">
                    {story()}
                </div>
            </div>
        </div>
    ))
    .add('only view', () => (
        <ScanningJobRemoveDialogue
            name="My scan job"
            onCancel={action('cancel')}
            onConfirm={action('confirm')}
        />
    ));
