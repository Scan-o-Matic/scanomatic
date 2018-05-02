import React from 'react';
import { storiesOf } from '@storybook/react';
import { action } from '@storybook/addon-actions';
import NewExperimentPanel from './NewExperimentPanel';
import '../../../style/bootstrap.css';
import '../../../style/project.css';


storiesOf('NewExperimentPanel', module)
    .addDecorator(story => (
        <div className="row">
            <div className="col-md-offset-1 col-md-10">
                {story()}
            </div>
        </div>
    ))
    .add('new experiment', () => (
        <NewExperimentPanel
            projectName="I am project"
            name=""
            description=""
            duration={0}
            interval={0}
            scannerId="haha"
            onChange={action('change')}
            onSubmit={action('submit')}
            onCancel={action('cancel')}
            scanners={[
                {
                    name: 'Tox',
                    owned: false,
                    power: true,
                    identifier: 'hoho',
                },
                {
                    name: 'Npm',
                    owned: true,
                    power: false,
                    identifier: 'haha',
                },
            ]}
        />
    ))
    .add('with errors', () => (
        <NewExperimentPanel
            projectName="I am project"
            name=""
            description=""
            duration={0}
            interval={0}
            scannerId="haha"
            onChange={action('change')}
            onSubmit={action('submit')}
            onCancel={action('cancel')}
            scanners={[
                {
                    name: 'Tox',
                    owned: false,
                    power: true,
                    identifier: 'hoho',
                },
                {
                    name: 'Npm',
                    owned: true,
                    power: false,
                    identifier: 'haha',
                },
            ]}
            errors={new Map([
                ['general', 'No repsonse from server.'],
                ['name', 'Can not be blank'],
                ['description', 'Dont be lazy, write something'],
                ['duration', 'I do not like days, I want nights'],
                ['interval', 'Must be at least 5 minutes'],
                ['scannerId', 'Can not be empty'],
            ])}
        />
    ))
    .add('new experiment with only duration error', () => (
        <NewExperimentPanel
            projectName="I am project"
            name=""
            description=""
            duration={0}
            interval={1200000}
            scannerId="haha"
            onChange={action('change')}
            onSubmit={action('submit')}
            onCancel={action('cancel')}
            scanners={[
                {
                    name: 'Tox',
                    owned: false,
                    power: true,
                    identifier: 'hoho',
                },
                {
                    name: 'Npm',
                    owned: true,
                    power: false,
                    identifier: 'haha',
                },
            ]}
            errors={new Map([
                ['duration', 'There should be at least 25h / day'],
            ])}
        />
    ));
