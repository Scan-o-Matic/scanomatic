import React from 'react';
import { storiesOf } from '@storybook/react';
import { action } from '@storybook/addon-actions';
import ProjectPanel from './ProjectPanel';
import Duration from '../Duration';
import '../../../style/bootstrap.css';
import '../../../style/project.css';


storiesOf('ProjectPanel', module)
    .addDecorator(story => (
        <div className="row">
            <div className="col-md-offset-1 col-md-10">
                {story()}
            </div>
        </div>
    ))
    .add('default expanded view', () => (
        <ProjectPanel
            name="I am project"
            description="The rapid horizontal transmission of many antibiotic resistance genes between bacterial host cells on conjugative plasmids is a major cause of the accelerating antibiotic resistance crisis. Preventing understanding and targeting conjugation, there currently are no experimental platforms for fast and cost-efficient screening of genetic effects on antibiotic resistance transmission by conjugation. We introduce a novel experimental framework to screen for conjugation based horizontal transmission of antibiotic resistance between >60,000 pairs of cell populations in parallel. Plasmid-carrying donor strains are constructed in high throughput."
            onNewExperiment={action('newExperiment')}
        />
    ))
    .add('new experiment', () => (
        <ProjectPanel
            name="I am project"
            description="The rapid horizontal transmission of many antibiotic resistance genes between bacterial host cells on conjugative plasmids is a major cause of the accelerating antibiotic resistance crisis. Preventing understanding and targeting conjugation, there currently are no experimental platforms for fast and cost-efficient screening of genetic effects on antibiotic resistance transmission by conjugation. We introduce a novel experimental framework to screen for conjugation based horizontal transmission of antibiotic resistance between >60,000 pairs of cell populations in parallel. Plasmid-carrying donor strains are constructed in high throughput."
            newExperiment={{
                project: 'I am project',
                name: '',
                description: '',
                duration: new Duration(),
                interval: new Duration(),
                scannerId: 'haha',
            }}
            newExperimentActions={{
                onNameChange: action('nameChange'),
                onDescriptionChange: action('descriptionChange'),
                onDurationDaysChange: action('durationDaysChange'),
                onDurationHoursChange: action('durationHoursChange'),
                onDurationMinutesChange: action('durationMinutesChange'),
                onIntervalChange: action('intervalChange'),
                onScannerChange: action('scannerChange'),
                onSubmit: action('submit'),
                onCancel: action('cancel'),
            }}
            onNewExperiment={action('newExperiment')}
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
    .add('new experiment with errors', () => (
        <ProjectPanel
            name="I am project"
            description="The rapid horizontal transmission of many antibiotic resistance genes between bacterial host cells on conjugative plasmids is a major cause of the accelerating antibiotic resistance crisis. Preventing understanding and targeting conjugation, there currently are no experimental platforms for fast and cost-efficient screening of genetic effects on antibiotic resistance transmission by conjugation. We introduce a novel experimental framework to screen for conjugation based horizontal transmission of antibiotic resistance between >60,000 pairs of cell populations in parallel. Plasmid-carrying donor strains are constructed in high throughput."
            newExperiment={{
                project: 'I am project',
                name: '',
                description: '',
                duration: new Duration(),
                interval: new Duration(),
                scannerId: 'haha',
            }}
            newExperimentActions={{
                onNameChange: action('nameChange'),
                onDescriptionChange: action('descriptionChange'),
                onDurationDaysChange: action('durationDaysChange'),
                onDurationHoursChange: action('durationHoursChange'),
                onDurationMinutesChange: action('durationMinutesChange'),
                onIntervalChange: action('intervalChange'),
                onScannerChange: action('scannerChange'),
                onSubmit: action('submit'),
                onCancel: action('cancel'),
            }}
            onNewExperiment={action('newExperiment')}
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
            newExperimentErrors={{
                general: 'No repsonse from server.',
                name: 'Can not be blank',
                durationDays: 'I do not like days, I want nights',
                durationHours: 'There should be at least 25h / day',
                durationMinutes: 'I prefer decimal time',
                interval: 'Must be at least 5 minutes',
                scanner: 'Can not be empty',
            }}
        />
    ))
    .add('new experiment with only hours error', () => (
        <ProjectPanel
            name="I am project"
            description="The rapid horizontal transmission of many antibiotic resistance genes between bacterial host cells on conjugative plasmids is a major cause of the accelerating antibiotic resistance crisis. Preventing understanding and targeting conjugation, there currently are no experimental platforms for fast and cost-efficient screening of genetic effects on antibiotic resistance transmission by conjugation. We introduce a novel experimental framework to screen for conjugation based horizontal transmission of antibiotic resistance between >60,000 pairs of cell populations in parallel. Plasmid-carrying donor strains are constructed in high throughput."
            newExperiment={{
                project: 'I am project',
                name: '',
                description: '',
                duration: new Duration(),
                interval: new Duration(),
                scannerId: 'haha',
            }}
            newExperimentActions={{
                onNameChange: action('nameChange'),
                onDescriptionChange: action('descriptionChange'),
                onDurationDaysChange: action('durationDaysChange'),
                onDurationHoursChange: action('durationHoursChange'),
                onDurationMinutesChange: action('durationMinutesChange'),
                onIntervalChange: action('intervalChange'),
                onScannerChange: action('scannerChange'),
                onSubmit: action('submit'),
                onCancel: action('cancel'),
            }}
            onNewExperiment={action('newExperiment')}
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
            newExperimentErrors={{
                durationHours: 'There should be at least 25h / day',
            }}
        />
    ));
