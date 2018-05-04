import React from 'react';
import { storiesOf } from '@storybook/react';
import { action } from '@storybook/addon-actions';
import ScanningJobFeatureExtractDialogue from './ScanningJobFeatureExtractDialogue';
import '../../../style/bootstrap.css';
import '../../../style/experiment.css';


storiesOf('ScanningJobFeatureExtractDialogue', module)
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
        <ScanningJobFeatureExtractDialogue
            projectPath="/root/12309sdaf09124dssfdw"
            onCancel={action('cancel')}
            onConfirm={action('confirm')}
        />
    ));
